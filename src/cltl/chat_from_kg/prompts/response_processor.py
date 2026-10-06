import re

from prompts.instruct import Instruct

# Mirrors kg_gap_finder.py's own NAMESPACE/AGENT_ROLE_PREDICATES, and
# events_from_chat/events_to_capsules.py's AGENT_LIKE_ROLES (duplicated rather than imported --
# response_processor.py otherwise has no dependency on either module, just on the gap-row
# *shape* kg_gap_finder.py's find_*() functions produce; see kg_gap_finder.py's own
# PREDICATE_GROUPS comment for why these predicates are treated as one interchangeable group
# rather than independently gap-checked). A gap on any of these predicates is who
# performed/experienced/was affected by the activity -- in this single-human chat, that's
# essentially always the person speaking, so get_prompt_for_kg_gap() asks them to confirm it
# rather than asking an open "who did this" question (see get_prompt_for_agent_gap()).
NAMESPACE = "http://cltl.nl/leolani/n2mu/"
AGENT_PREDICATES = {
    NAMESPACE + "agent", NAMESPACE + "agent_patient", NAMESPACE + "participant", NAMESPACE + "experiencer",
}

# * _statement_novelty
# * _entity_novelty
# * _negation_conflicts
# * _complement_conflict
# * _subject_gaps
# * _complement_gaps
# * _overlaps
# * _trust


def local_name(uri: str) -> str:
    """Human-ish label for a URI: the part after the last '/' or '#', with underscores and
    camelCase word boundaries (e.g. n2mu/time/recurringTime's "recurringTime") turned into
    spaces, lowercased. Passed straight through unchanged if it doesn't look like a URI (e.g. a
    literal value, which kg_gap_finder.py rows also carry in subject/predicate/object position)."""
    if not uri or "://" not in uri:
        return uri
    tail = uri.rsplit("#", 1)[-1].rsplit("/", 1)[-1]
    tail = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", tail)
    return tail.replace("_", " ").lower()


class PromptProcessor():

    def __init__(self, language="English"):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """


        self._instruct = Instruct(language)

    def get_no_answer_prompt(self, question):
        prompt = [self._instruct.get_instruct_for_no_answer(), {"role": "system", "content": "I have no answer for this question"+question["utterance"]}]
        return prompt

    def get_answer_prompt(self, question, answer):
        prompt = [self._instruct.get_instruct_for_answer(), {"role": "system", "content": answer}]
        return prompt

    def get_thought_prompt(self, statement, thought):
        prompt = [self._instruct.get_instruct_for_statement(), {"role": "system", "content": statement+thought}]
        return prompt

    def get_utterance_from_statement (self, statement):
        if "utterance" in statement:
            utterance = statement["utterance"]
        else:
            utterance = ""
        return utterance

    def get_triple_text_from_statement (self, statement):
        triple = statement["triple"]
        triple_text = triple["_subject"]["_label"]+", "+triple["_predicate"]["_label"]+", "+triple["_complement"]["_label"]
        return triple_text

    def get_perspective_from_statement  (self, statement):
        perspective = statement["perspective"]
        perspective_text = ""
        if not perspective['_certainty']=='UNDERSPECIFIED':
            perspective_text += perspective['_certainty'].tolower()+ ", "
        if perspective['_polarity']=='POSITIVE':
            perspective_text += 'believes'+ ", "
        elif perspective['_polarity']=='NEGATIVE':
            perspective_text += 'denies'+ ", "
        if not perspective['_sentiment']=='NEUTRAL':
            perspective_text += perspective['_sentiment']+ ", "
        if not perspective['_emotion']=='UNDERSPECIFIED':
            perspective_text += perspective['_emotion'].tolower()+ ", "
        return perspective_text

    def get_author_from_statement (self, statement):
        author = statement["author"]["label"]
        return author

    #### In the case of a GAP, there is a trigger triple in the eKG that contains a "known entity".
    #### If the known entity is the subject of the trigger triple it can either be the subject or the complement (object) in a triple with another predicate.
    #### If a subject_gap_subject the known entity is subject in both the trigger triple and the unknown triple.
    #### If a subject_gap_complement the known entity is subject in the trigger triple and the object in the unknown triple.
    #### If a complement_gap_complement  the known entity is the complement in both the trigger triple and the unknown triple.
    #### If a complement_gap_subject the known entity is object in the trigger triple and the subject in the unknown triple.

    def get_subject_gap_subject(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text = known_entity +", "+predicate+", "+ gap_type[0]
        return gap_text

    def get_subject_gap_complement(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text = gap_type[0]+", "+predicate+", "+known_entity
        return gap_text

    def get_complement_gap_subject(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text =gap_type[0]+", ", predicate+", "+known_entity
        return gap_text

    def get_complement_gap_complement(self, thought):
        known_entity = thought["_known_entity"]["_label"]
        predicate = thought["_predicate"]["_label"]
        gap_type = thought["_entity"]["_types"]
        gap_text = known_entity +", " + predicate+ ", " + gap_type[0]
        return gap_text

    def get_negation_conflict(self, conflict):
        provenance = conflict["_provenance"]
        author = provenance["_author"]["_label"]
        date = provenance["_date"]
        value = conflict["_polarity_value"]
        if value == "NEGATIVE":
            value = "denies"
        elif value == "POSITIVE":
            value = "claims"
        novelty_text = author + " "+ value+ " this on " + date
        return novelty_text


    # def get_cardinality_conflict(self, thought):
    #     novelty_text = author + " "+ value+ " me on " + date
    #     return novelty_text

    def get_provenance_from_statement_novelty(self, provenance):
        # {'_provenance': {
        #     '_author': {'_id': 'http://cltl.nl/leolani/friends/carl', '_label': 'carl', '_offset': None, '_confidence': 0.0,
        #                 # '_types': ['Source', 'Actor']}, '_date': '2017-10-24'}}
        author = provenance["_author"]["_label"]
        date = provenance["_date"]
        novelty_text = author + " told me this on "+ date
        return novelty_text

    def get_all_prompt_input_from_response(self, response):
        statement = response["statement"]
        statement_text = self.get_triple_text_from_statement(statement)
        statement_author = self.get_author_from_statement(statement)
        prompts = []
        thought = response["thoughts"]
        #print("THOUGHT", thought)
        novelties = thought["_statement_novelty"]
      #  print("novelties", novelties)
        for novelty in novelties:
            input = statement_author + " claims " + statement_text + ". Also " + self.get_provenance_from_statement_novelty(novelty["_provenance"])
            prompt = [self._instruct.get_instruct_for_novelty(), {"role": "user", "content": input}]
            prompts.append(prompt)

        # @TODO
        # novelties = thought["_entity_novelty"]
        # for novelty in novelties:
        #     input = statement_author+" claims "+ statement_text+". Also "+get_XXX_from_entity_novelty(novelty)
        #     prompt = [instruct.instruct_for_novelty, {"role": "user", "content": input}]
        #     prompts.append(prompt)

        conflicts = thought["_negation_conflicts"]
      #  print("conflicts", conflicts)
        for conflict in conflicts:
            input = statement_author + " claims " + statement_text + ". But " + self.get_negation_conflict(conflict)
            prompt = [self._instruct.get_instruct_for_novelty(), {"role": "user", "content": input}]
            prompts.append(prompt)

        # @TODO
        # conflicts = thought["_complement_conflict"]
        # for conflict in conflicts:
        #     input = statement_author+" claims "+ statement_text+". Also "+get_complement_conflict(conflict)
        #     inputs.append(input)

        gaps = thought["_subject_gaps"]
       # print("_subject_gaps", gaps)
        if gaps["_subject"]:
            for gap in gaps["_subject"]:
                input = self.get_subject_gap_subject(gap)
                prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
             #   print(prompt)
                prompts.append(prompt)

            if gaps["_complement"]:
                for gap in gaps["_complement"]:
                    input = self.get_subject_gap_complement(gap)
                    prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
              #      print(prompt)

                    prompts.append(prompt)

        gaps = thought["_complement_gaps"]
       # print("_complement_gaps", gaps)

        if gaps["_subject"]:
            for gap in gaps["_subject"]:
                input = self.get_complement_gap_subject(gap)
                prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
               # print(prompt)

                prompts.append(prompt)

            if gaps["_complement"]:
                for gap in gaps["_complement"]:
                    input = self.get_complement_gap_complement(gap)
                    prompt = [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": input}]
                #    print(prompt)

                    prompts.append(prompt)

        # @TODO
        # overlaps = thought["_overlaps"]
        # for overlap in overlaps:
        #     input = statement_author+" claims "+ statement_text+". Also "+get_overlap_statement(overlap)
        #     inputs.append(input)

        # @TODO
        # trust = thought["_trust"]
        # input = statement_author+" claims "+ statement_text+". Also "+get_trust_statement(trust)
        # inputs.append(input)
        return prompts

    #### kg_gap_finder.py builds its own gap-row schema (class/predicate/subject[/object_type|
    #### object]/subject_triples/...), unrelated to cltl.brain's "thought" schema the methods
    #### above are built for -- get_prompt_for_kg_gap() is the equivalent entry point for that
    #### schema instead, reusing the same subject-gap instruct/phrasing as get_subject_gap_subject()
    #### et al.

    def _label_for_subject(self, gap: dict) -> str:
        """Prefer the subject's own rdfs:label, already included in a kg_gap_finder.py row's
        subject_triples, over a raw URI/activity_id.

        A subject can end up with more than one rdfs:label: the real phrase from when the
        activity was first introduced (e.g. "cycling"), AND its bare activity_id (e.g.
        "chat9999.1") from a LATER turn that referred back to it without repeating a phrase --
        events_to_capsules.get_triples_with_types_and_activity_id() falls back to the
        activity_id itself as the RDF label whenever that turn's extraction has no "value" of
        its own. Any label that's just the subject's own id is skipped in favour of a real one.
        """
        subject_id = gap["subject"].rstrip("/").rsplit("/", 1)[-1]
        labels = [
            triple["object"] for triple in gap.get("subject_triples") or []
            if triple["predicate"] == "http://www.w3.org/2000/01/rdf-schema#label"
        ]
        real_labels = [label for label in labels if label != subject_id]
        if real_labels:
            return real_labels[0]
        if labels:
            return labels[0]
        return local_name(gap["subject"])

    def _format_peer_examples(self, gap: dict) -> str:
        """Render a kg_gap_finder.py gap row's `peer_examples` (find_predicate_gaps() et al. --
        the actual values peers do have for the missing predicate, most frequent first) as a
        short comma-separated list for the prompt, e.g. "in the morning (3x), 7pm (2x), after
        dinner". Bare (no "Nx") when a value only occurred once. Empty string if the row carries
        no peer_examples (e.g. an older kg_gap_finder, or a hand-built gap row in a test)."""
        examples = gap.get("peer_examples") or []
        if not examples:
            return ""
        return ", ".join(
            f"{example['value']} ({example['count']}x)" if example["count"] > 1 else example["value"]
            for example in examples
        )

    def _format_known_context(self, gap: dict) -> str:
        """Render a gap row's `known_context` (populated only by intent_gap_finder.py's
        next_intent_gap() -- see its _known_context()) as a short phrase describing facts about
        this SAME event that are already known, e.g. who did it and when. Fed into
        get_prompt_for_kg_gap() so the resulting question can weave those in (e.g. "yesterday")
        instead of asking about them again -- and so a weaker LLM backend has real material to
        build a natural question from, instead of just the bare subject/predicate/type triple,
        which is also how the raw predicate name (e.g. "patient") can end up leaking into the
        question. Empty string if the row carries no known_context at all (e.g. a plain
        kg_gap_finder.py row, which never sets this field)."""
        context = gap.get("known_context") or {}
        parts = []
        if context.get("agent"):
            parts.append("it was done by the person you're talking to -- address them as \"you\"")
        if context.get("time"):
            parts.append(f"it happened: {context['time']}")
        return "; ".join(parts)

    def get_prompt_for_agent_gap(self, gap: dict, human: str):
        """Build a [instruct, user-message] prompt asking `human` to confirm they were the agent
        of `gap`'s subject, instead of the open "who did this" question get_prompt_for_kg_gap()
        would otherwise build. Used for a gap on `agent`/`agent_patient` (see AGENT_PREDICATES)
        -- in this single-human chat, the activity's agent is essentially always the person
        speaking, so it's a yes/no confirmation, not a genuine unknown."""
        subject_label = self._label_for_subject(gap)
        gap_text = f"{human}, agent, {subject_label}"
        return [self._instruct.get_instruct_for_agent_confirmation(), {"role": "user", "content": gap_text}]

    def get_prompt_for_kg_gap(self, gap: dict, kind: str, human: str = None):
        """Build a [instruct, user-message] prompt (the same shape get_all_prompt_input_from_response()
        produces) for one kg_gap_finder.py gap row, so an LLM can turn it into a natural
        follow-up question.

        :param gap: one row from kg_gap_finder's report -- a find_predicate_gaps() /
            find_predicate_object_gaps() / find_predicate_object_instances_gaps() result. If it
            carries a non-empty `peer_examples` (the actual values peers do have for the missing
            predicate, most frequent first -- see kg_gap_finder.py), those are woven into the
            prompt so the resulting question can suggest concrete options (e.g. "did you do this
            in the morning, like usual, or at a different time?") instead of asking blind.
        :param kind: which of those three the row came from --
            "predicate" (object unknown), "predicate_object_type" (object's TYPE known, e.g.
            "a Wine"), or "predicate_object_instance" (the exact expected object known).
        :param human: the name of the person chatting, if known. When `gap`'s predicate is
            `agent`/`agent_patient` (AGENT_PREDICATES) and `human` is given, this delegates to
            get_prompt_for_agent_gap() instead -- see there for why.

        If `gap` carries a non-empty `question_template` (intent_gap_finder.py's own hand-
        authored example question for this requirement, e.g. "What do you have for lunch?", with
        {activity}/{patient} already filled in), that takes priority over everything else here:
        the LLM is asked to paraphrase THAT question fluently (weaving in `known_context` --
        see _format_known_context()) instead of inventing one from the bare subject/predicate/
        type triple -- see get_instruct_for_templated_gap().
        """
        if human is not None and gap["predicate"] in AGENT_PREDICATES:
            return self.get_prompt_for_agent_gap(gap, human)

        question_template = gap.get("question_template")
        if question_template:
            content = question_template
            context_text = self._format_known_context(gap)
            if context_text:
                content += f". Already known about this same event: {context_text}"
            return [self._instruct.get_instruct_for_templated_gap(), {"role": "user", "content": content}]

        subject_label = self._label_for_subject(gap)
        predicate_label = local_name(gap["predicate"])
        if kind == "predicate":
            gap_type = "something"
        elif kind == "predicate_object_type":
            gap_type = local_name(gap["object_type"])
        elif kind == "predicate_object_instance":
            gap_type = local_name(gap["object"])
        else:
            raise ValueError(f"Unknown gap kind {kind!r}, expected 'predicate', 'predicate_object_type' or 'predicate_object_instance'")
        gap_text = f"{subject_label}, {predicate_label}, {gap_type}"

        examples_text = self._format_peer_examples(gap)
        if examples_text:
            gap_text += f". Examples from similar peers: {examples_text}"
            return [self._instruct.get_instruct_for_subject_gap_with_examples(), {"role": "user", "content": gap_text}]

        context_text = self._format_known_context(gap)
        if context_text:
            gap_text += f". Already known about this same event: {context_text}"
            return [self._instruct.get_instruct_for_subject_gap_with_context(), {"role": "user", "content": gap_text}]

        return [self._instruct.get_instruct_for_subject_gap(), {"role": "user", "content": gap_text}]

    #### Handling the human's ANSWER to a get_prompt_for_agent_gap() confirmation question --
    #### chat_sessions.KgChatSession._handle_confirmation_reply() uses these two together:
    #### get_prompt_for_confirmation_response() first, to classify what the human said, then
    #### get_prompt_for_gap_filled_ack() once the resulting triple has actually been pushed to
    #### the KG, to acknowledge it (or, on a plain denial, chat_sessions.py falls back to
    #### get_prompt_for_kg_gap() itself, called without `human=` this time).

    def get_prompt_for_confirmation_response(self, question: str, reply: str):
        """Build a [instruct, user-message] prompt asking the LLM to classify a human's reply to
        a get_prompt_for_agent_gap() yes/no confirmation question as one of CONFIRM / DENY /
        "CORRECT: <value>" (see get_instruct_for_confirmation_response() for the exact contract).
        `question` is that confirmation question's own text (the reply LLMTripleReplier.reply()
        returned for it), so the classifier has it as context for what's being answered."""
        content = f"Question asked: {question}\nReply: {reply}"
        return [self._instruct.get_instruct_for_confirmation_response(), {"role": "user", "content": content}]

    def get_prompt_for_gap_filled_ack(self, gap: dict, value: str):
        """Build a [instruct, user-message] prompt for a brief natural-language acknowledgement
        after `gap` has just been filled -- confirmed or corrected -- with `value`."""
        subject_label = self._label_for_subject(gap)
        predicate_label = local_name(gap["predicate"])
        content = f"{subject_label}, {predicate_label}, {value}"
        return [self._instruct.get_instruct_for_gap_filled_ack(), {"role": "user", "content": content}]

    #### Handling the human's answer to an intent_gap_finder.py-driven follow-up question, for
    #### any gap that carries a "fill_role" (see intent_gap_finder._make_gap()'s own docstring on
    #### exactly which requirements set one) -- chat_sessions.KgChatSession._handle_intent_answer_reply()
    #### uses these together with get_prompt_for_gap_filled_ack() (ANSWER) or
    #### get_prompt_for_gap_declined_ack() (DECLINE) above: get_prompt_for_intent_answer_response()
    #### first, to classify what the human said, then whichever ack applies once that's resolved
    #### (an UNRELATED reply gets neither -- see chat_sessions.py for why).

    def get_prompt_for_intent_answer_response(self, question: str, reply: str):
        """Build a [instruct, user-message] prompt asking the LLM to classify a human's reply to
        an intent-driven follow-up question as one of "ANSWER: <value>" / DECLINE / UNRELATED
        (see get_instruct_for_intent_answer_response() for the exact contract). `question` is
        that follow-up's own text, so the classifier has it as context for what's being asked."""
        content = f"Question asked: {question}\nReply: {reply}"
        return [self._instruct.get_instruct_for_intent_answer_response(), {"role": "user", "content": content}]

    def get_prompt_for_gap_declined_ack(self, gap: dict):
        """Build a [instruct, user-message] prompt for a brief natural-language acknowledgement
        after the human indicated (get_prompt_for_intent_answer_response()'s DECLINE) that they
        don't have/didn't do/don't know whatever `gap` was asking about -- the requirement is
        simply dropped, not asked again."""
        subject_label = self._label_for_subject(gap)
        predicate_label = local_name(gap["predicate"])
        content = f"{subject_label}, {predicate_label}"
        return [self._instruct.get_instruct_for_gap_declined_ack(), {"role": "user", "content": content}]
