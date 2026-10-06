# _statement_novelty
# _entity_novelty
# _negation_conflicts
# _complement_conflict
# _subject_gaps
# _complement_gaps
# _overlaps
# _trust

class Instruct():

    def __init__(self, language="English"):
        # type: () -> None
        """
        Generate natural language based on structured data

        Parameters
        ----------
        """
        self._language = language

    def get_instruct_for_statement (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a phrase, followed by a perspective, followed by \"that\" and a triple with a subject, predicate and object.\
         You need to paraphrase the input in plain " + self._language + ". \
         Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
        }
        return instruct

    def get_instruct_for_answer (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: the question, followed by the answer.\
         You need to paraphrase the input in plain " + self._language + ". \
         Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
        }
        return instruct

    def get_instruct_for_no_answer (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         You were given a question but you do not have an answer from your memory.\
         You are given the question as input and need to explain that you have no answer in plain " + self._language + ". \
         Only reply with the short paraphrase of the input and only use the subject and object from the triple in your reply as given. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         The response should be just the paraphrased text and nothing else."
        }
        return instruct


    def get_instruct_for_subject_gap (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of subject.\
         You need to paraphrase the input in plain " + self._language + " as a question for the object. \
         Use who for the type person, where for the type location, when for the type time and what for everything else. \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         Do not use the words agent, patient or experiencer in the paraphrase. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct


    def get_instruct_for_subject_gap_with_examples(self):
        """Same as get_instruct_for_subject_gap(), but the input also carries example values
        peers have for the missing predicate (see kg_gap_finder.py's `peer_examples` and
        prompts.response_processor._format_peer_examples()) -- used so the resulting question
        can suggest concrete, evidence-backed options instead of asking blind."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of subject, \
         followed by example values that similar peers reported for that predicate. \
         You need to paraphrase the triple in plain " + self._language + " as a question for the object, \
         and naturally weave in one or two of the example values as suggestions to make the question \
         concrete and easy to answer -- without assuming or stating that they apply to the subject. \
         Use who for the type person, where for the type location, when for the type time and what for everything else. \
         Only reply with the short paraphrase of the input. \
         When responding use the names or labels from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         Do not use the words agent, patient or experiencer in the paraphrase. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct


    def get_instruct_for_subject_gap_with_context(self):
        """Same as get_instruct_for_subject_gap(), but the input also carries facts about this
        SAME event that are already known (see prompts.response_processor._format_known_context()
        -- currently populated only by intent_gap_finder.py's patient/date checks), so the
        resulting question can naturally weave them in (e.g. mention "yesterday", address the
        person as "you") instead of asking about them again -- and so a weaker LLM backend has
        real material to build a natural question from instead of just the bare triple, which is
        also how the raw predicate name (e.g. "patient") can otherwise leak into the question."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of subject, \
         followed by facts about this same event that are already known. \
         You need to paraphrase the triple in plain " + self._language + " as a single natural \
         question about the missing object only, naturally weaving in the already-known facts \
         instead of asking about them again -- e.g. mention a known time like \"yesterday\", and \
         address the person as \"you\" when they are already known to be the one who did it. \
         Use who for the type person, where for the type location, when for the type time and what for everything else. \
         Only reply with the short paraphrase of the input. \
         When responding use the names or labels from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         Do not use the words agent, patient or experiencer in the paraphrase. \
         Do not ask again about any fact already given as known. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct


    def get_instruct_for_templated_gap(self):
        """Used when a gap row carries a `question_template` -- one of intent_gap_finder.py's
        own hand-authored example questions (patient_question/qualification_question/
        date_question/location_question in intents/*.json), already filled in with
        {activity}/{patient} (see intent_gap_finder._render_question_template()). Rather than
        asking the LLM to invent a question from a bare subject/predicate/type triple (see
        get_instruct_for_subject_gap()), this asks it to paraphrase the GIVEN candidate question
        fluently -- so the intent author's own exact wording/phrasing is what's actually asked,
        not something the LLM reinvents -- while still weaving in whatever's already known about
        the same event (see prompts.response_processor._format_known_context()) instead of
        asking about that again."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a candidate question to ask, optionally followed by facts \
         about this same event that are already known. \
         Paraphrase the candidate question fluently and naturally in plain " + self._language + ", \
         keeping its exact meaning and exactly what it asks for -- do not invent a different \
         question, and do not ask about anything extra. \
         If facts about this same event are already known, naturally weave them into the \
         question instead of asking about them again -- e.g. mention a known time like \
         \"yesterday\", and address the person as \"you\" when they are already known to be the \
         one who did it. \
         Only reply with the short paraphrase. \
         Do not give an explanation. \
         Do not use the words agent, patient or experiencer in the paraphrase. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct


    def get_instruct_for_agent_confirmation(self):
        """Used by prompts.response_processor.get_prompt_for_agent_gap() for a gap on the
        `agent`/`agent_patient` role: instead of asking who did something (get_instruct_for_subject_gap()
        with an open "who" question), ask the named person to confirm it was them -- this is a
        single-human chat, so the activity's agent is essentially always the person speaking."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: the name of a person, the word \"agent\", and an activity. \
         You need to phrase this in plain " + self._language + " as a short yes/no question, \
         addressed to that person, confirming that they themselves were the one doing the activity \
         -- e.g. \"Just to confirm, was that you who went for a run?\". \
         Only reply with the short confirmation question. \
         Do not give an explanation. \
         Do not explain what the input is. \
         The response should be just the question and nothing else."
               }
        return instruct


    def get_instruct_for_confirmation_response(self):
        """Used by prompts.response_processor.get_prompt_for_confirmation_response() to
        classify a human's reply to a get_instruct_for_agent_confirmation() yes/no question --
        see chat_sessions.KgChatSession._classify_confirmation_reply(), which parses the exact
        three-way output contract described here."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you a yes/no confirmation question that was asked, and the person's reply \
         to it. Classify the reply as exactly one of: \
         CONFIRM -- the person confirmed the statement is correct (e.g. yes, that's right, correct). \
         DENY -- the person denied it and did NOT give the correct information instead. \
         CORRECT: <value> -- the person denied it AND gave the correct information in the same \
         reply; <value> is that corrected information, stated as briefly and literally as \
         possible in a few words, e.g. \"CORRECT: my son\". \
         Respond with exactly one line, exactly one of these three forms, and nothing else -- \
         no explanation, no extra punctuation."
               }
        return instruct


    def get_instruct_for_gap_filled_ack(self):
        """Used by prompts.response_processor.get_prompt_for_gap_filled_ack() once a gap has
        just been filled (confirmed or corrected) and pushed to the KG -- a brief acknowledgement
        instead of another question."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you a short fact that the person just confirmed or corrected. \
         Reply with a brief, warm acknowledgement in plain " + self._language + ", at most one \
         short sentence (e.g. \"Got it, thanks!\"). Do not repeat the fact back in full. \
         Do not ask a new question. \
         The response should be just the acknowledgement and nothing else."
               }
        return instruct


    def get_instruct_for_intent_answer_response(self):
        """Used by prompts.response_processor.get_prompt_for_intent_answer_response() to
        classify a human's reply to an intent-driven follow-up question (see
        intent_gap_finder.next_intent_gap()) -- see
        chat_sessions.KgChatSession._classify_intent_answer_reply(), which parses the exact
        three-way output contract described here. Unlike get_instruct_for_confirmation_response()
        (a yes/no question specifically), this question could be answered with any short phrase
        (e.g. "yoghurt with fresh fruit", "30 minutes"), so the UNRELATED case matters here in a
        way it doesn't there -- a reply that neither answers nor declines (e.g. a clarifying
        question back, or a change of subject) must never be pushed as if it were the answer."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you a question that was asked, and the person's reply to it. Classify the \
         reply as exactly one of: \
         ANSWER: <value> -- the reply gives ANY concrete information that answers the question, \
         even if it's brief, informally phrased, or contains typos (e.g. \"Coupld of days\" IS an \
         answer to \"how long\" -- treat it as ANSWER: a couple of days, do not let a typo or \
         casual phrasing make you classify it as DECLINE or UNRELATED); <value> is that answer, \
         corrected for obvious typos, stated as briefly and literally as possible in a few words, \
         e.g. \"ANSWER: yoghurt with fresh fruit\". \
         DECLINE -- the person EXPLICITLY indicated they don't have, didn't do, or don't know this \
         -- only for a clear negative like no, nothing, not really, I don't know, none -- never \
         for a reply that actually contains an answer, however short. \
         UNRELATED -- the reply does not answer or decline the question at all -- e.g. it asks a \
         clarifying question back, or changes the subject. \
         When genuinely unsure between ANSWER and DECLINE, prefer ANSWER -- a wrong DECLINE \
         silently throws away real information the person just gave you. \
         Respond with exactly one line, exactly one of these three forms, and nothing else -- \
         no explanation, no extra punctuation."
               }
        return instruct


    def get_instruct_for_gap_declined_ack(self):
        """Used by prompts.response_processor.get_prompt_for_gap_declined_ack() once the human
        has indicated (via get_instruct_for_intent_answer_response()'s DECLINE) that they don't
        have/didn't do/don't know whatever an intent-driven gap question asked about -- a brief
        acknowledgement that drops the requirement instead of asking it again."""
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you a short subject and topic that the person just indicated they don't \
         have, didn't do, or don't know. Reply with a brief, warm acknowledgement in plain " + self._language + ", \
         at most one short sentence (e.g. \"No worries, thanks for letting me know.\"). \
         Do not ask a new question about it. \
         The response should be just the acknowledgement and nothing else."
               }
        return instruct


    def get_instruct_for_object_gap (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of object.\
         You need to paraphrase the input in plain  " + self._language + "  as a question for the object. \
         Use who for the type person, where for the type location, when for the type time and what for everything else. \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct

    def get_instruct_for_novelty (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of object and a phrase.\
         You need to paraphrase the input in plain  " + self._language + ". \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
               }
        return instruct

    def get_instruct_for_conflict (self):
        instruct = {"role": "system", "content": "You are an intelligent assistant. \
         I will give you as input: a triple with a subject, a predicate and a type of object and a phrase that expresses a conflict.\
         You need to paraphrase the input in plain  " + self._language + ". \
         Only reply with the short paraphrase of the input. \
         When responding use the names from the triple and be specific. \
         Do not give an explanation. \
         Do not explain what the subject and object is. \
         The response should be just the paraphrased text and nothing else."
           }
        return instruct
