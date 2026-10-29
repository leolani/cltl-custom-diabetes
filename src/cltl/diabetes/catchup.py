"""The conversation logic of cltl-kg-driven-chat, wrapped for the event bus.

This is the flow of cltl-kg-driven-chat's notebooks/kg_catchup_intent_chat.ipynb,
minus its GUI:

  1. Find the last conversation with the human in the knowledge graph and the
     topics (activity types) discussed back then (catch_up_from_kg).
  2. Open with the gap in time, a summary of those topics and a question about
     the topic that most needs catching up on (SaturationTracker).
  3. Answer every following utterance through KgIntentChatSession: extract
     events from it, push them to the knowledge graph, and ask the follow-up
     questions defined in intents/*.json; once those are exhausted, go back to
     the next catch-up topic until every topic is saturated.
  4. A question about the human's own record ("what did I eat yesterday?") is
     answered from the knowledge graph instead (kg_question_answerer).

The code doing the actual work lives in src/cltl/kg_chat (copied from
cltl-kg-driven-chat/notebooks) and src/cltl/{chat_from_kg,events_from_chat,
gaps_from_kg} (copied from cltl-kg-driven-chat/src). It locates its own
src/cltl and intents/ directories by walking up from the working directory,
so the process has to run from this module's root directory, the same
requirement src/main.py already has for config/.
"""
import logging
import os
import re
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from cltl.diabetes import place
from cltl.diabetes.api import Conversation, ConversationFactory, Reply
from cltl.kg_chat.graphdb_links import graphdb_visualization_url

logger = logging.getLogger(__name__)

def _load_kg_chat():
    # Imported on first use: chat_sessions reads OPENAI_API_KEY at import time.
    from cltl.kg_chat import catch_up_from_kg, chat_sessions, kg_question_answerer

    return chat_sessions, catch_up_from_kg, kg_question_answerer


class CatchUpConversation(Conversation):
    def __init__(self, scenario_id: str, human: str, ltm, kg_address: str, intents,
                 log_dir: Path, fallback_gap_days: int, model: Optional[str],
                 max_answer_words: Optional[int] = None,
                 place_same_threshold: float = place.DEFAULT_SAME_THRESHOLD,
                 place_ask_threshold: float = place.DEFAULT_ASK_THRESHOLD):
        chat_sessions, catch_up, kg_question_answerer = _load_kg_chat()
        self._chat_sessions = chat_sessions
        self._catch_up = catch_up

        self.scenario_id = scenario_id
        self.human = human
        self._kg_address = kg_address
        self._log_dir = log_dir
        # What open() asked about the place (place.ASK or place.CONFIRM), so
        # the next utterance is taken as the answer to it; None otherwise.
        self._place_question = None
        self._place_decision = place.PlaceDecision(place.KNOWN)

        self.current_date = datetime.now()
        fallback_date = self.current_date - timedelta(days=fallback_gap_days)
        self.last_conversation_date = catch_up.find_last_conversation_date(
            human, ltm, self.current_date, fallback_date)
        self.catch_up_topics = catch_up.find_catch_up_topics(
            ltm, self.current_date, self.last_conversation_date)
        logger.info("Catch-up for %s in scenario %s: last conversation %s, topics %s",
                    human, scenario_id, self.last_conversation_date,
                    [(t["activity_type"], t["expected_count"]) for t in self.catch_up_topics])

        model = model or chat_sessions.DEFAULT_MODEL
        self._model = model
        llm = chat_sessions.openai_agent(model)
        self._places = place.PlaceResolver(
            ltm, human, lambda messages: chat_sessions._call_openai("working out where you are", llm, messages),
            same_threshold=place_same_threshold, ask_threshold=place_ask_threshold)
        self.tracker = catch_up.SaturationTracker(self.catch_up_topics, human=human, model=model)
        agent_fn = catch_up.wrap_agent_fn_with_saturation_loop(
            chat_sessions.openai_agent(model), self.tracker)
        # None leaves the human's questions to the normal flow.
        question_answerer = kg_question_answerer.KgQuestionAnswerer(
            kg_address, human, model=model, max_answer_words=max_answer_words
        ) if max_answer_words else None

        self.session = chat_sessions.KgIntentChatSession(
            # A fresh id per conversation, as in kg_catchup_intent_chat.ipynb:
            # activities are minted under the chat id, so reusing one would
            # merge them with an earlier conversation's.
            chat=int(time.time()),
            human=human,
            kg_address=kg_address,
            log_dir=str(log_dir / "kg_logs"),
            intents=intents,
            agent_fn=agent_fn,
            on_new_subject=self.tracker.record_new_activity,
            question_answerer=question_answerer,
            brain=ltm,
            # The chat goes into the scenario's context, shared with the
            # perceptions of the scenario (see DiabetesService._add_context).
            context_id=scenario_id,
        )

    def open(self) -> Optional[str]:
        """The introduction, which also settles where the human is (see
        cltl.diabetes.place): it says they are at the same place as last time,
        asks whether they are, or asks where they are."""
        decision = self._decide_place()
        place_question, place_remark, fallback = None, None, "How have you been since we last spoke?"
        if decision.kind == place.SAME:
            self._set_place(decision.place_uri)
            place_remark = f"that you can see they are at {decision.name} again"
        elif decision.kind == place.CONFIRM:
            place_question = f"whether they are at {decision.name} again"
            fallback = f"Are you at {decision.name} again?"
        elif decision.kind == place.ASK:
            place_question = "where they are right now (e.g. at home, at work), since you don't know that yet"
            fallback = "Where are you right now?"
        self._place_question = decision.kind if place_question else None
        self._place_decision = decision

        try:
            opening = self.tracker.opening_question(self.current_date, self.last_conversation_date,
                                                    place_question=place_question, place_remark=place_remark)
        except self._chat_sessions.ChatTimeoutError as e:
            logger.warning("Timed out %s, opening with a plain greeting instead", e.source)
            opening = f"Hi {self.human}, good to talk to you again! {fallback}"

        self.session.open_with(opening)

        return opening

    def respond(self, utterance: str) -> Reply:
        question, self._place_question = self._place_question, None
        if question == place.ASK:
            self._save_named_place(lambda city: self._places.name_place(utterance, city))
            return Reply(self._after_place_answer(utterance))
        if question == place.CONFIRM:
            return Reply(self._answer_confirmation(utterance))

        reply = self.session.say(utterance)

        return Reply(reply, self._graph_uri())

    def _graph_uri(self) -> Optional[str]:
        """The GraphDB visualisation of the activity the last utterance added
        triples about, or None if it added nothing (e.g. a question answered
        from the knowledge graph)."""
        human_entries = [entry for entry in self.session.turn_log if entry["speaker"] == self.human]
        if not human_entries or not human_entries[-1]["triples_pushed"]:
            return None

        # say() points last_subject_uris at the activities of this utterance,
        # including the one a follow-up answer added a role to. The visual
        # graph centres on one of them; the others are a click away when they
        # share a role, such as the human as agent.
        subjects = self.session.last_subject_uris
        if len(subjects) > 1:
            logger.info("Utterance added triples about %s, visualising %s", subjects, subjects[0])

        return graphdb_visualization_url(self._kg_address, subjects[0]) if subjects else None

    def _answer_confirmation(self, utterance: str) -> str:
        decision = self._place_decision
        answer = self._resolve(lambda city: self._places.resolve_confirmation(utterance, decision.name, city))
        if answer == "SAME":
            self._set_place(decision.place_uri)
        elif answer:
            self._save_named_place(lambda city: answer)
        else:
            # Not there, and not said where: ask once more.
            self._place_question = place.ASK
            self.session.record_human_turn(utterance)
            reply = "Where are you right now, then?"
            self.session.open_with(reply)
            return reply

        return self._after_place_answer(utterance)

    def _after_place_answer(self, utterance: str) -> str:
        """The answer about the place went to the context, not through event
        extraction (which would take "I'm at my daughter's" for an activity);
        the reply is the first catch-up question."""
        self.session.record_human_turn(utterance)
        try:
            reply = (self.tracker.next_question(messages=self.session._messages)
                     or self._chat_sessions.openai_agent(self._model)(self.session._messages))
        except self._chat_sessions.ChatTimeoutError as e:
            logger.warning("Timed out %s after the place answer", e.source)
            reply = "Thanks! How have you been since we last spoke?"
        # open_with() adds the agent's turn and pushes it, as for any agent turn.
        self.session.open_with(reply)

        return reply

    def close(self) -> None:
        if not self.session.turns:
            return

        self._chat_sessions.save_turns(self.session.turns, str(self._turns_path()))
        self._catch_up.save_intent_log(self.session, self.tracker, self.catch_up_topics,
                                       self.current_date, self.last_conversation_date,
                                       log_dir=str(self._log_dir / "intents_log"))

    def _decide_place(self) -> "place.PlaceDecision":
        try:
            return self._places.decide(self.scenario_id)
        except Exception:
            logger.exception("Failed to work out the place of scenario %s", self.scenario_id)
            return place.PlaceDecision(place.KNOWN)

    def _city_name(self) -> Optional[str]:
        return self._places.city_name(self._places.city(self._places.context_uri(self.scenario_id)))

    def _resolve(self, ask) -> Optional[str]:
        """`ask(city name)` -- an LLM call on the human's answer -- or None if it fails."""
        try:
            return ask(self._city_name())
        except Exception:
            logger.exception("Failed to understand the answer about the place in scenario %s", self.scenario_id)
            return None

    def _save_named_place(self, ask):
        name = self._resolve(ask)
        if not name:
            logger.info("No place in the answer, the place of scenario %s stays unknown", self.scenario_id)
            return
        try:
            self._places.save_named_place(self.scenario_id, name)
        except Exception:
            # The conversation goes on without it.
            logger.exception("Failed to save the place of scenario %s", self.scenario_id)

    def _set_place(self, place_uri: str):
        try:
            self._places.set_place(self.scenario_id, place_uri)
        except Exception:
            logger.exception("Failed to save the place of scenario %s", self.scenario_id)

    def _turns_path(self) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"chat{self.session.chat}_{self.human}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        return self._log_dir / "chat_logs" / f"{name}_turns_{stamp}.json"


class CatchUpConversationFactory(ConversationFactory):
    def __init__(self, kg_address: str, intents_dir: str, log_dir: str,
                 fallback_gap_days: int = 7, model: Optional[str] = None,
                 max_answer_words: Optional[int] = None,
                 place_same_threshold: float = place.DEFAULT_SAME_THRESHOLD,
                 place_ask_threshold: float = place.DEFAULT_ASK_THRESHOLD):
        # The kg-chat code reads the key at import time and calls sys.exit()
        # when it is missing, which would take down the service thread in the
        # middle of a conversation instead of at start-up.
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set; cltl-custom-diabetes needs it "
                             "for event extraction and reply generation")
        self._kg_address = kg_address
        self._intents_dir = intents_dir
        self._log_dir = Path(log_dir)
        self._fallback_gap_days = fallback_gap_days
        self._model = model
        self._max_answer_words = max_answer_words
        self._place_same_threshold = place_same_threshold
        self._place_ask_threshold = place_ask_threshold

        self._lock = threading.Lock()
        self._intents = None

    def create(self, scenario_id: str, human: str, ltm) -> Conversation:
        return CatchUpConversation(scenario_id, human, ltm, self._kg_address, self._load_intents(),
                                   self._log_dir, self._fallback_gap_days, self._model,
                                   max_answer_words=self._max_answer_words,
                                   place_same_threshold=self._place_same_threshold,
                                   place_ask_threshold=self._place_ask_threshold)

    def _load_intents(self):
        with self._lock:
            if self._intents is None:
                chat_sessions, _, _ = _load_kg_chat()
                # chat_sessions imports intent_gap_finder lazily, together
                # with the other knowledge graph dependencies.
                intent_gap_finder = chat_sessions._load_kg_dependencies()["intent_gap_finder"]
                self._intents = intent_gap_finder.load_intents(self._intents_dir)
                logger.info("Loaded %s intents from %s", len(self._intents), self._intents_dir)

            return self._intents


def connect_ltm(kg_address: str, log_dir: str, clear_brain: bool = False):
    """The one LongTermMemory for `kg_address`, shared by the service and every
    conversation. `clear_brain` empties the repository first."""
    if not kg_address or kg_address.startswith("$"):
        raise ValueError(f"[cltl.diabetes] kg_address is not set ({kg_address!r}); "
                         "set kg_server and kg_repository, or CLTL_KG_ADDRESS to the "
                         "GraphDB repository's SPARQL endpoint")

    _, catch_up, _ = _load_kg_chat()
    ltm = catch_up.connect_brain(kg_address, log_dir=str(Path(log_dir) / "kg_logs"), clear_all=clear_brain)
    if clear_brain:
        logger.warning("Cleared the knowledge graph at %s (clear_brain)", kg_address)
    logger.info("Connected to %s", kg_address)

    return ltm
