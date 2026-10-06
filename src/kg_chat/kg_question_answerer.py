"""
kg_question_answerer.py
=========================

Answers the human's own questions about what they reported earlier ("what did I eat yesterday?",
"when did I last go for a walk?", "how have I been sleeping this week?") from the knowledge graph,
instead of letting the plain LLM agent guess or the SRL extractor mistake the question for a new
report. Used by chat_sessions.KgChatSession.say() (see its `question_answerer` parameter), ahead of
the normal annotate-and-push flow.

Per human utterance, at most three steps:

  1. plan() -- a cheap heuristic (looks_like_question()) and, only if that passes, one LLM call
     that decides whether this is a question about the human's own recorded history, and if so
     turns it into a small, structured query plan: activity types, keywords and a start date.
     The LLM never writes SPARQL itself.
  2. retrieve() -- fixed, parameterized SPARQL queries (ACTIVITY_QUERY, DETAIL_QUERY,
     MENTION_QUERY) for the activities the HUMAN reported (grasp:wasAttributedTo the human -- the
     agent's own turns are pushed to the graph too, as expectations, and are not facts about the
     human), most recently mentioned first. A plan whose types/keywords match nothing is retried
     once without them, so a wrongly guessed activity type doesn't hide the answer.
  3. answer() -- one LLM call that paraphrases the retrieved facts as a plain-language reply to
     the question, and, if that reply is longer than `max_answer_words`, a second one that
     summarises it to fit.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from chat_sessions import DEFAULT_MODEL, _call_openai, _openai_client

logger = logging.getLogger(__name__)

# Above this many words, the paraphrased answer is summarised (see KgQuestionAnswerer.answer()).
DEFAULT_MAX_ANSWER_WORDS = 80

# How many activities (most recently mentioned first) are passed to the LLM as facts.
DEFAULT_MAX_FACTS = 40

# events_from_chat/data_type.ActivityType's values, as local names in the graph (spaces turned
# into underscores, see chat_sessions.DEFAULT_GAP_ACTIVITY_TYPES).
QUERYABLE_ACTIVITY_TYPES = (
    "exercise", "measurement", "sleep", "take_food", "take_drink", "take_medicine", "social",
    "diet", "treatment", "physical_condition", "social_condition", "mental_condition", "symptom",
    "disease", "other",
)

LOOKUP_FAILED_REPLY = ("Sorry, I couldn't look that up in what you've told me before right now. "
                       "Could you ask me again in a moment?")

# How many of the running conversation's last messages give the planner and the answer the
# context of a follow-up question ("and the week before?").
CONTEXT_MESSAGES = 4

# The utterance text of a mention, as shown to the LLM, is cut off after this many characters.
MAX_UTTERANCE_CHARS = 200

N2MU = "http://cltl.nl/leolani/n2mu/"
N2MU_TIME = N2MU + "time/"
RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
RDFS_LABEL = "http://www.w3.org/2000/01/rdf-schema#label"
GAF_DENOTED_IN = "http://groundedannotationframework.org/gaf#denotedIn"
GRASP_FACTUALITY = "http://groundedannotationframework.org/grasp/factuality#"

# Role predicates that are bookkeeping, not something the human told.
_IGNORED_PREDICATES = {N2MU + "id"}

# A phrase-less activity is labelled with its own id (e.g. "chat1791202938.6").
_ID_LABEL = re.compile(r"chat\d+\.\d+")

# The coach itself, as it appears as a role filler (leolaniFriends:agent).
_COACH_LABEL = "agent"

_QUESTION_WORDS = (
    "what", "when", "where", "which", "who", "whom", "whose", "why", "how", "did", "do", "does",
    "have", "has", "had", "was", "were", "is", "are", "am", "can", "could", "would", "will",
)
_QUESTION_PHRASES = ("tell me", "remind me", "show me", "do you know", "do you remember", "list ")

# rdflib's SPARQLStore prepends its own PREFIX declarations (rdf, rdfs, prov, dc, foaf, ...) to
# every query, and GraphDB rejects a query that declares a prefix twice ("Multiple prefix
# declarations for prefix 'rdf'"). So only prefixes rdflib does not bind are declared here, and
# rdf:/rdfs: terms are written out as full IRIs.
_PREFIXES = """
PREFIX n2mu: <http://cltl.nl/leolani/n2mu/>
PREFIX gaf: <http://groundedannotationframework.org/gaf#>
PREFIX grasp: <http://groundedannotationframework.org/grasp#>
PREFIX sem: <http://semanticweb.cs.vu.nl/2009/11/sem/>
"""

# The activities the human reported, filtered by ${filters}, most recently mentioned first.
# sem:hasBeginTimeStamp is a "YYYY-MM-DD HH:MM:SS.ffffff" string, so a string comparison with a
# "YYYY-MM-DD" date is a date comparison.
ACTIVITY_QUERY = _PREFIXES + """
SELECT ?activity (MAX(STR(?ts)) AS ?last_mentioned) WHERE {
    ?activity a n2mu:activity ;
              gaf:denotedIn ?mention .
    ?mention grasp:wasAttributedTo ?source ;
             sem:hasBeginTimeStamp ?ts .
    ?source <http://www.w3.org/2000/01/rdf-schema#label> ?source_label .
    FILTER(LCASE(STR(?source_label)) = "%(human)s")
    %(filters)s
}
GROUP BY ?activity
ORDER BY DESC(?last_mentioned)
LIMIT %(limit)d
"""

DETAIL_QUERY = _PREFIXES + """
SELECT ?activity ?p ?o ?o_label WHERE {
    VALUES ?activity { %(activities)s }
    ?activity ?p ?o .
    OPTIONAL { ?o <http://www.w3.org/2000/01/rdf-schema#label> ?o_label }
}
"""

MENTION_QUERY = _PREFIXES + """
SELECT ?mention ?ts ?text ?source_label ?value WHERE {
    VALUES ?mention { %(mentions)s }
    ?mention sem:hasBeginTimeStamp ?ts ;
             grasp:wasAttributedTo ?source .
    ?source <http://www.w3.org/2000/01/rdf-schema#label> ?source_label .
    OPTIONAL { ?mention <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?text . FILTER(isLiteral(?text)) }
    OPTIONAL { ?mention grasp:hasAttribution ?attribution .
               ?attribution <http://www.w3.org/1999/02/22-rdf-syntax-ns#value> ?value . }
}
"""


@dataclass
class QueryPlan:
    """What plan() made of a question: the activity types and keywords to look for (either
    matches; neither given means "everything"), and the earliest date the activities may have
    been mentioned on (None: no lower bound)."""
    question: str
    activity_types: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    since: Optional[str] = None


@dataclass
class Fact:
    """One activity the human reported, flattened for the answer prompt."""
    activity: str
    label: Optional[str]
    types: List[str]
    roles: Dict[str, List[str]]
    mentioned: List[str]
    utterances: List[str]
    polarity: Optional[str]

    def describe(self) -> str:
        types = "/".join(t.replace("_", " ") for t in self.types) or "activity"
        parts = [f'{types} "{self.label}"' if self.label else types]
        parts.extend(f"{role}: {', '.join(values)}" for role, values in self.roles.items())
        if self.polarity:
            parts.append(f"factuality: {self.polarity.lower()}")
        line = f"- mentioned {', '.join(self.mentioned) or 'on an unknown date'}: " + "; ".join(parts)
        if self.utterances:
            line += " | said: " + " / ".join(f'"{u}"' for u in self.utterances)
        return line


def looks_like_question(utterance: str) -> bool:
    """Cheap pre-check so plain reports never cost the extra planning LLM call."""
    text = (utterance or "").strip().lower()
    if not text:
        return False
    if "?" in text:
        return True
    first_word = re.split(r"[^a-z']+", text, maxsplit=1)[0]
    return first_word in _QUESTION_WORDS or any(phrase in text for phrase in _QUESTION_PHRASES)


def _sparql_string(value: str) -> str:
    """`value` safe to put between double quotes in a SPARQL query."""
    return re.sub(r'["\\\n\r]', " ", value).strip().lower()


def _local_name(uri: str) -> str:
    return re.split(r"[/#]", uri.rstrip("/"))[-1]


def _recent_context(messages: Optional[List[Dict]]) -> List[Dict]:
    if not messages:
        return []
    return [m for m in messages[-CONTEXT_MESSAGES:] if m.get("role") != "system"]


class KgQuestionAnswerer:
    def __init__(self, kg_address: str, human: str, model: str = DEFAULT_MODEL,
                 max_answer_words: int = DEFAULT_MAX_ANSWER_WORDS,
                 max_facts: int = DEFAULT_MAX_FACTS,
                 run_query: Optional[Callable[[str], List[Dict]]] = None,
                 complete: Optional[Callable[..., str]] = None):
        """
        run_query -- callable(sparql) -> rows (dicts of strings, None for unbound), defaults to
                     kg_gap_finder.run_query() against `kg_address`.
        complete  -- callable(messages, json_mode=False) -> str, defaults to an OpenAI chat
                     completion with `model`.
        Both are injectable so the answerer can be tested without GraphDB or OpenAI.
        """
        self.kg_address = kg_address
        self.human = human
        self.model = model
        self.max_answer_words = max_answer_words
        self.max_facts = max_facts
        self._run_query = run_query or self._endpoint_query
        self._complete = complete or self._openai_complete
        self._graph = None

    # ------------------------------------------------------------------ #
    # Entry point
    # ------------------------------------------------------------------ #

    def try_answer(self, utterance: str, messages: Optional[List[Dict]] = None) -> Optional[str]:
        """The reply to `utterance` if it is a question about the human's own record, None if it
        is not, in which case the caller handles the utterance as usual. ChatTimeoutError from an
        LLM call propagates.

        A failed knowledge graph query still answers (LOOKUP_FAILED_REPLY): handing the question
        back to the caller would extract it as if it were a report ("What did I eat?" -> a new
        "eat" activity) and ask about that instead."""
        plan = self.plan(utterance, messages)
        if plan is None:
            return None

        try:
            facts = self.retrieve(plan)
        except Exception:
            logger.exception("Failed to query %s for %r", self.kg_address, utterance)
            return LOOKUP_FAILED_REPLY

        logger.info("Answering %r from %s fact(s) (types=%s keywords=%s since=%s)",
                    utterance, len(facts), plan.activity_types, plan.keywords, plan.since)

        return self.answer(plan, facts, messages)

    # ------------------------------------------------------------------ #
    # 1. Plan
    # ------------------------------------------------------------------ #

    def plan(self, utterance: str, messages: Optional[List[Dict]] = None) -> Optional[QueryPlan]:
        if not looks_like_question(utterance):
            return None

        today = datetime.now()
        system = (
            f"You route messages in a chat between a lifestyle coach and {self.human}, a person "
            "with Type 2 diabetes. Everything they told the coach in earlier conversations "
            "(food, drinks, exercise, sleep, medication, measurements, symptoms, mood, social "
            "life, ...) is stored as activities in a knowledge graph.\n\n"
            f"Decide whether {self.human}'s LAST message asks to be told something about their "
            "own past activities or conditions as recorded there, e.g. \"what did I eat "
            "yesterday?\", \"when did I last exercise?\", \"how have I slept this week?\", "
            "\"what did I tell you about my medication?\".\n"
            "It is NOT such a question when it asks for a recap of the current conversation, asks "
            "for general information or advice (\"is pasta bad for me?\"), asks about the coach, "
            "or is a rhetorical question within a report.\n\n"
            "Answer with a JSON object with these keys:\n"
            '- "kg_question": true or false\n'
            '- "activity_types": the relevant types, a subset of '
            f"{list(QUERYABLE_ACTIVITY_TYPES)}, or [] if unclear or any type fits\n"
            '- "keywords": up to 5 short lower-case words for the specific things asked about '
            '(e.g. "pasta", "walk", "insulin"), or [] for none\n'
            '- "since": the earliest date the answer could have been mentioned on, as '
            f'YYYY-MM-DD, or null for no limit. Today is {today:%A %Y-%m-%d}.'
        )
        prompt = ([{"role": "system", "content": system}]
                  + _recent_context(messages[:-1] if messages else None)
                  + [{"role": "user", "content": utterance}])
        raw = _call_openai("checking whether you asked about your own records",
                           self._complete, prompt, json_mode=True)

        try:
            data = json.loads(raw or "{}")
        except json.JSONDecodeError:
            logger.warning("Planner returned no JSON: %r", raw)
            return None
        if not isinstance(data, dict) or data.get("kg_question") is not True:
            return None

        activity_types = [t for t in (data.get("activity_types") or [])
                          if isinstance(t, str) and t in QUERYABLE_ACTIVITY_TYPES]
        keywords = [_sparql_string(k) for k in (data.get("keywords") or []) if isinstance(k, str)]
        since = data.get("since")
        if not (isinstance(since, str) and re.fullmatch(r"\d{4}-\d{2}-\d{2}", since)):
            since = None

        return QueryPlan(utterance, activity_types, [k for k in keywords if k][:5], since)

    # ------------------------------------------------------------------ #
    # 2. Retrieve
    # ------------------------------------------------------------------ #

    def retrieve(self, plan: QueryPlan) -> List[Fact]:
        activities = self._find_activities(plan, narrow=True)
        if not activities and (plan.activity_types or plan.keywords):
            activities = self._find_activities(plan, narrow=False)
        if not activities:
            return []

        return self._describe_activities(activities)

    def _find_activities(self, plan: QueryPlan, narrow: bool) -> List[str]:
        filters = []
        if plan.since:
            filters.append(f'FILTER(STR(?ts) >= "{plan.since}")')

        matches = []
        if narrow and plan.activity_types:
            types = " ".join(f"n2mu:{t}" for t in plan.activity_types)
            matches.append(f"EXISTS {{ VALUES ?type {{ {types} }} ?activity a ?type . }}")
        if narrow and plan.keywords:
            keyword_tests = " || ".join(f'CONTAINS(LCASE(STR(?l)), "{k}")' for k in plan.keywords)
            matches.append("EXISTS { { ?activity <http://www.w3.org/2000/01/rdf-schema#label> ?l } UNION "
                           "{ ?activity ?role ?filler . ?filler <http://www.w3.org/2000/01/rdf-schema#label> ?l } "
                           f"FILTER({keyword_tests}) }}")
        if matches:
            filters.append(f"FILTER({' || '.join(matches)})")

        query = ACTIVITY_QUERY % {"human": _sparql_string(self.human),
                                  "filters": "\n    ".join(filters), "limit": self.max_facts}

        return [row["activity"] for row in self._run_query(query) if row.get("activity")]

    def _describe_activities(self, activities: List[str]) -> List[Fact]:
        details = self._run_query(DETAIL_QUERY % {"activities": " ".join(f"<{a}>" for a in activities)})

        facts = {a: Fact(a, None, [], {}, [], [], None) for a in activities}
        mentions_of = {a: [] for a in activities}
        for row in details:
            fact = facts.get(row["activity"])
            if fact is None:
                continue
            predicate, value, value_label = row["p"], row["o"], row.get("o_label")
            if predicate == RDFS_LABEL:
                if not _ID_LABEL.fullmatch(value):
                    fact.label = fact.label or value
            elif predicate == RDF_TYPE:
                # n2mu:take_food etc.; not the generic n2mu:activity, and not n2mu:nl/eckg/...
                local = value[len(N2MU):] if value.startswith(N2MU) else None
                if local and "/" not in local and local != "activity" and local not in fact.types:
                    fact.types.append(local)
            elif predicate == GAF_DENOTED_IN:
                if value not in mentions_of[fact.activity]:
                    mentions_of[fact.activity].append(value)
            elif predicate.startswith(N2MU) and predicate not in _IGNORED_PREDICATES:
                role = "time" if predicate.startswith(N2MU_TIME) else _local_name(predicate)
                shown = self._show_value(value, value_label, is_time=role == "time")
                values = fact.roles.setdefault(role.replace("_", " "), [])
                if shown.lower() not in (v.lower() for v in values):
                    values.append(shown)

        self._add_mentions(facts, mentions_of)

        return [facts[a] for a in activities]

    def _add_mentions(self, facts: Dict[str, Fact], mentions_of: Dict[str, List[str]]):
        all_mentions = sorted({m for ms in mentions_of.values() for m in ms})
        if not all_mentions:
            return

        rows = self._run_query(MENTION_QUERY % {"mentions": " ".join(f"<{m}>" for m in all_mentions)})
        by_mention = {}
        for row in rows:
            entry = by_mention.setdefault(row["mention"], {"ts": row["ts"], "text": row.get("text"),
                                                           "source": row.get("source_label"),
                                                           "polarity": None})
            value = row.get("value") or ""
            if value.startswith(GRASP_FACTUALITY) and _local_name(value) in ("POSITIVE", "NEGATIVE"):
                entry["polarity"] = _local_name(value)

        human = self.human.strip().lower()
        for activity, mentions in mentions_of.items():
            fact = facts[activity]
            for mention in mentions:
                entry = by_mention.get(mention)
                if not entry or (entry["source"] or "").strip().lower() != human:
                    continue
                day = (entry["ts"] or "")[:10]
                if day and day not in fact.mentioned:
                    fact.mentioned.append(day)
                text = (entry["text"] or "").strip()
                if text and text not in fact.utterances:
                    fact.utterances.append(text[:MAX_UTTERANCE_CHARS] + ("..." if len(text) > MAX_UTTERANCE_CHARS else ""))
                fact.polarity = entry["polarity"] or fact.polarity
            fact.mentioned.sort()

    @staticmethod
    def _show_value(value: str, label: Optional[str], is_time: bool) -> str:
        shown = label or _local_name(value).replace("_", " ")
        if shown == _COACH_LABEL:
            return "the coach (you)"
        if is_time:
            date = re.match(r"\d{4}-\d{2}-\d{2}", _local_name(value))
            if date and date.group(0) not in shown:
                shown = f"{shown} ({date.group(0)})"
        return shown

    # ------------------------------------------------------------------ #
    # 3. Answer
    # ------------------------------------------------------------------ #

    def answer(self, plan: QueryPlan, facts: List[Fact], messages: Optional[List[Dict]] = None) -> str:
        today = datetime.now()
        facts_text = "\n".join(f.describe() for f in facts) if facts else "(no matching records)"
        system = (
            f"You are a lifestyle coach chatting with {self.human}, a person with Type 2 "
            f"diabetes. {self.human} asked you about something they told you in earlier "
            "conversations. Answer ONLY from the records below, which are what they told you, "
            "each with the date they mentioned it and their own words. A record's time may be "
            "relative to the date it was mentioned (\"yesterday\" said on 2026-03-02 means "
            "2026-03-01). A factuality of \"negative\" means it did NOT happen.\n"
            "Talk to them directly (\"you\"), in warm, plain, natural language: no record "
            "formats, IDs, category names like \"take_food\", or lists. Say when things happened "
            "in a natural way (\"last Tuesday\", \"on 3 March\"). If the records do not answer the "
            "question, say honestly that you don't have that noted yet, and ask whether they "
            "want to tell you now. Do not give advice and do not make anything up.\n\n"
            f"Today is {today:%A %Y-%m-%d}.\n\nRecords:\n{facts_text}"
        )
        prompt = ([{"role": "system", "content": system}]
                  + _recent_context(messages[:-1] if messages else None)
                  + [{"role": "user", "content": plan.question}])
        reply = (_call_openai("looking up the answer to your question", self._complete, prompt) or "").strip()

        if len(reply.split()) > self.max_answer_words:
            reply = self.summarise(plan.question, reply)

        return reply

    def summarise(self, question: str, answer: str) -> str:
        prompt = [
            {"role": "system", "content": (
                f"Shorten the coach's answer to {self.human}'s question to at most "
                f"{self.max_answer_words} words. Keep the facts that answer the question most "
                "directly and the most recent ones; group similar things together instead of "
                "listing each. Keep the warm, direct tone, speak to them as \"you\", and do not "
                "add anything that is not in the answer. Reply with the shortened answer only."
            )},
            {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
        ]
        summary = (_call_openai("summarising the answer to your question", self._complete, prompt) or "").strip()

        return summary or answer

    # ------------------------------------------------------------------ #
    # Defaults for run_query / complete
    # ------------------------------------------------------------------ #

    def _endpoint_query(self, query: str) -> List[Dict]:
        from chat_sessions import _load_kg_dependencies

        kg_gap_finder = _load_kg_dependencies()["kg_gap_finder"]
        if self._graph is None:
            self._graph = kg_gap_finder.load_graph_from_endpoint(self.kg_address)

        return kg_gap_finder.run_query(self._graph, query)

    def _openai_complete(self, messages: List[Dict], json_mode: bool = False) -> str:
        kwargs = {"response_format": {"type": "json_object"}} if json_mode else {}
        response = _openai_client().chat.completions.create(model=self.model, messages=messages, **kwargs)

        return response.choices[0].message.content
