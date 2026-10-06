import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "kg_chat"))

import kg_question_answerer
from kg_question_answerer import KgQuestionAnswerer, LOOKUP_FAILED_REPLY, QueryPlan, looks_like_question

N2MU = "http://cltl.nl/leolani/n2mu/"
WORLD = "http://cltl.nl/leolani/world/"
TALK = "http://cltl.nl/leolani/talk/"
ACTIVITY = N2MU + "chat17.3"
MENTION = TALK + "chat17_utterance3_char0-40"
AGENT_MENTION = TALK + "chat17_utterance4_char0-30"


def _rows_for(query):
    if "GROUP BY ?activity" in query:
        return [{"activity": ACTIVITY, "last_mentioned": "2026-10-04 09:00:00"}]
    if "?activity ?p ?o ?o_label" in query:
        return [
            {"activity": ACTIVITY, "p": "http://www.w3.org/2000/01/rdf-schema#label", "o": "ate", "o_label": None},
            {"activity": ACTIVITY, "p": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "o": N2MU + "activity", "o_label": None},
            {"activity": ACTIVITY, "p": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "o": N2MU + "take_food", "o_label": None},
            {"activity": ACTIVITY, "p": "http://www.w3.org/1999/02/22-rdf-syntax-ns#type", "o": N2MU + "nl/eckg/EventSeries", "o_label": None},
            {"activity": ACTIVITY, "p": N2MU + "patient", "o": WORLD + "pasta", "o_label": "pasta"},
            {"activity": ACTIVITY, "p": N2MU + "time/vagueTime", "o": N2MU + "time/2026-10-03T00:00:00", "o_label": "yesterday"},
            {"activity": ACTIVITY, "p": N2MU + "id", "o": "3", "o_label": None},
            {"activity": ACTIVITY, "p": "http://groundedannotationframework.org/gaf#denotedIn", "o": MENTION, "o_label": None},
            {"activity": ACTIVITY, "p": "http://groundedannotationframework.org/gaf#denotedIn", "o": AGENT_MENTION, "o_label": None},
        ]
    if "?mention ?ts ?text" in query:
        return [
            {"mention": MENTION, "ts": "2026-10-04 09:00:00.1", "text": "Yesterday I ate pasta",
             "source_label": "Jan", "value": "http://groundedannotationframework.org/grasp/factuality#POSITIVE"},
            {"mention": AGENT_MENTION, "ts": "2026-10-04 09:01:00.1", "text": "Did you eat pasta again?",
             "source_label": "agent", "value": None},
        ]
    raise AssertionError("unexpected query " + query)


class FakeLLM:
    def __init__(self, plan, answer):
        self.plan = plan
        self.answer = answer
        self.calls = []

    def __call__(self, messages, json_mode=False):
        self.calls.append((messages, json_mode))
        if json_mode:
            return json.dumps(self.plan)
        if messages[0]["content"].startswith("Shorten"):
            return "You had pasta yesterday."
        return self.answer


class KgQuestionAnswererTest(unittest.TestCase):
    def _answerer(self, llm, rows=_rows_for, max_answer_words=80):
        self.queries = []

        def run_query(query):
            self.queries.append(query)
            return rows(query)

        return KgQuestionAnswerer("http://kg", "Jan", run_query=run_query, complete=llm,
                                  max_answer_words=max_answer_words)

    def test_looks_like_question(self):
        self.assertTrue(looks_like_question("What did I eat yesterday"))
        self.assertTrue(looks_like_question("remind me when I walked"))
        self.assertTrue(looks_like_question("I ate pasta, is that ok?"))
        self.assertFalse(looks_like_question("I ate pasta yesterday."))

    def test_report_is_not_planned(self):
        llm = FakeLLM({"kg_question": True}, "unused")
        self.assertIsNone(self._answerer(llm).try_answer("I ate pasta yesterday."))
        self.assertEqual([], llm.calls)

    def test_non_kg_question_falls_through(self):
        llm = FakeLLM({"kg_question": False}, "unused")
        self.assertIsNone(self._answerer(llm).try_answer("Is pasta bad for me?"))
        self.assertEqual([], self.queries)

    def test_answers_from_human_facts_only(self):
        llm = FakeLLM({"kg_question": True, "activity_types": ["take_food", "bogus"],
                       "keywords": ["pas\"ta"], "since": "2026-10-01"},
                      "Yesterday you had pasta.")
        reply = self._answerer(llm).try_answer("What did I eat yesterday?")

        self.assertEqual("Yesterday you had pasta.", reply)
        activity_query = self.queries[0]
        self.assertIn("n2mu:take_food", activity_query)
        self.assertNotIn("bogus", activity_query)
        self.assertIn('"pas ta"', activity_query)
        self.assertIn('>= "2026-10-01"', activity_query)
        self.assertIn('= "jan"', activity_query)

        records = llm.calls[-1][0][0]["content"]
        self.assertIn('take food "ate"', records)
        self.assertIn("patient: pasta", records)
        self.assertIn("time: yesterday (2026-10-03)", records)
        self.assertIn("mentioned 2026-10-04", records)
        self.assertIn('"Yesterday I ate pasta"', records)
        self.assertIn("factuality: positive", records)
        self.assertNotIn("EventSeries", records)
        self.assertNotIn("; id:", records)
        self.assertNotIn("Did you eat pasta again", records)

    def test_long_answer_is_summarised(self):
        llm = FakeLLM({"kg_question": True}, " ".join(["word"] * 20))
        reply = self._answerer(llm, max_answer_words=10).try_answer("What did I eat?")

        self.assertEqual("You had pasta yesterday.", reply)

    def test_narrow_plan_without_matches_is_broadened(self):
        def rows(query):
            if "GROUP BY ?activity" in query and "take_drink" in query:
                return []
            return _rows_for(query)

        llm = FakeLLM({"kg_question": True, "activity_types": ["take_drink"]}, "Pasta.")
        self.assertEqual("Pasta.", self._answerer(llm, rows).try_answer("What did I drink?"))
        self.assertNotIn("take_drink", self.queries[1])

    def test_no_facts_still_answers(self):
        llm = FakeLLM({"kg_question": True}, "I don't have that noted yet.")
        reply = self._answerer(llm, rows=lambda q: []).try_answer("When did I last walk?")

        self.assertEqual("I don't have that noted yet.", reply)
        self.assertIn("(no matching records)", llm.calls[-1][0][0]["content"])

    def test_query_failure_is_answered(self):
        def rows(query):
            raise ConnectionError("GraphDB down")

        llm = FakeLLM({"kg_question": True}, "unused")
        self.assertEqual(LOOKUP_FAILED_REPLY, self._answerer(llm, rows).try_answer("What did I eat?"))

    def test_queries_do_not_redeclare_rdflib_prefixes(self):
        # rdflib's SPARQLStore prepends its own bindings; GraphDB rejects a prefix declared twice.
        from rdflib import Graph
        bound = {prefix for prefix, _ in Graph().namespace_manager.namespaces()}

        answerer = self._answerer(FakeLLM({}, ""))
        answerer.retrieve(QueryPlan("q", ["take_food"], ["pasta"], "2026-10-01"))

        self.assertEqual(3, len(self.queries))
        for query in self.queries:
            declared = set(re.findall(r"PREFIX\s+(\w*):", query))
            self.assertFalse(declared & bound, declared & bound)
            # Every prefixed name used is declared in the query itself.
            used = set(re.findall(r"(?<![\w<])([A-Za-z]\w*):\w", query.split("WHERE", 1)[-1])) - {"http", "https"}
            self.assertLessEqual(used, declared, used - declared)


if __name__ == "__main__":
    unittest.main()
