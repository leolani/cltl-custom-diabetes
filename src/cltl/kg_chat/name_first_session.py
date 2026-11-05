"""
name_first_session.py
=======================

The catch-up chat of kg_catchup_intent_chat.ipynb, for when the agent does not know yet who it is
talking to: a session for kg_chat_gui.run_gui() that

  1. opens by naming everyone in the knowledge graph the agent has talked with and asking the
     human's name (catch_up_from_kg.known_humans_opening());
  2. takes the name from the answer (catch_up_from_kg.name_in_utterance()) and queries the graph
     for what it remembers of that person (memory(): the last conversation, the catch-up topics,
     the topics talked about last time);
  3. starts the catch-up chat with them: a KgIntentChatSession driven by a SaturationTracker,
     opening with the catch-up question, exactly as the notebook did with a fixed HUMAN.

Until the name is known this object answers itself; after that, every attribute the GUI or the
notebook reads (turns, turn_log, kg_pushes, ...) comes from the real session, with the turns
about the name in front. `last_conversation_date` is set once the memory is in, which is how the
GUI knows to show the timeline (see kg_chat_gui.ChatWindow._on_reply_done()).
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from cltl.kg_chat import catch_up_from_kg as catch_up
from cltl.kg_chat.chat_sessions import (DEFAULT_MODEL, KgIntentChatSession, _today,
                                        openai_agent)

logger = logging.getLogger(__name__)

# Before the name is known, the human's turns are recorded under this speaker name.
UNKNOWN_HUMAN = "you"


class NameFirstCatchUpSession:
    def __init__(self, kg_address: str, brain, chat: int, intents_dir: Optional[str] = None,
                 log_dir: str = "kg_logs", current_date: Optional[datetime] = None,
                 fallback_gap_days: int = 7, model: str = DEFAULT_MODEL,
                 answer_questions: bool = True):
        self.kg_address = kg_address
        self.brain = brain
        self.chat = chat
        self.intents_dir = intents_dir
        self.log_dir = log_dir
        self.current_date = current_date or datetime.now()
        self.fallback_gap_days = fallback_gap_days
        self.model = model
        self.answer_questions = answer_questions
        self.date = _today()
        self.human = UNKNOWN_HUMAN

        # Set by _start(), once the name is known.
        self.session: Optional[KgIntentChatSession] = None
        self.tracker: Optional[catch_up.SaturationTracker] = None
        self.last_conversation_date: Optional[datetime] = None
        self.catch_up_topics: List[Dict] = []
        self.last_topics: List[str] = []

        self._pre_turns: List[Dict] = []
        self._pre_sources: List[str] = []

        self.known_humans = catch_up.find_known_humans(brain)
        self._add_turn("agent", catch_up.known_humans_opening(self.known_humans))

    # ------------------------------------------------------------------ #
    # What the GUI uses
    # ------------------------------------------------------------------ #

    def say(self, utterance: str) -> str:
        if self.session is not None:
            return self.session.say(utterance)

        self._add_turn(self.human, utterance)
        name = catch_up.name_in_utterance(utterance, self.known_humans, model=self.model)
        if not name:
            return self._agent_says(catch_up.NAME_RETRY_QUESTION)

        return self._start(name)

    @property
    def turns(self) -> List[Dict]:
        return self._pre_turns + (self.session.turns if self.session is not None else [])

    @property
    def reply_sources(self) -> List[str]:
        # The catch-up opening (open_with()) has no reply source of its own; it is an LLM reply.
        return self._pre_sources + (["default"] + self.session.reply_sources if self.session is not None else [])

    def __getattr__(self, name):
        # Only reached for attributes this object does not have itself: the real session's
        # (turn_log, kg_pushes, annotations, last_subject_uris, ...) once it exists.
        session = self.__dict__.get("session")
        if session is None:
            if name == "last_subject_uris":
                return []
            raise AttributeError(name)
        return getattr(session, name)

    # ------------------------------------------------------------------ #
    # Memory and the start of the catch-up
    # ------------------------------------------------------------------ #

    def memory(self, human: str) -> Dict:
        """What the knowledge graph has on `human`: the date of the last conversation (or
        `fallback_gap_days` ago when there was none), the catch-up topics with their saturation
        targets, the topics talked about in the last conversation, and when they last talked
        about each topic (the order of the questions once the catch-up is done)."""
        fallback = self.current_date - timedelta(days=self.fallback_gap_days)
        last_date = catch_up.find_last_conversation_date(human, self.brain, self.current_date, fallback)
        return {
            "human": human,
            "last_conversation_date": last_date,
            "catch_up_topics": catch_up.find_catch_up_topics(self.brain, self.current_date, last_date),
            "last_topics": catch_up.find_last_conversation_topics(human, self.brain),
            "topic_recency": catch_up.find_topic_recency(human, self.brain),
        }

    def _start(self, name: str) -> str:
        memory = self.memory(name)
        self.human = name
        first_meeting = name not in self.known_humans
        # find_catch_up_topics() counts everyone's activities, not just this person's: someone
        # the agent has never talked with has nothing to catch up on.
        self.catch_up_topics = [] if first_meeting else memory["catch_up_topics"]
        self.last_topics = [] if first_meeting else memory["last_topics"]
        logger.info("Catching up with %s: last conversation %s, last topics %s", name,
                    memory["last_conversation_date"], self.last_topics)

        self.tracker = catch_up.SaturationTracker(self.catch_up_topics, human=name, model=self.model,
                                                  topic_recency=memory["topic_recency"])
        question_answerer = None
        if self.answer_questions:
            from cltl.kg_chat.kg_question_answerer import KgQuestionAnswerer
            question_answerer = KgQuestionAnswerer(self.kg_address, name, model=self.model)
        session = KgIntentChatSession(
            chat=self.chat,
            human=name,
            kg_address=self.kg_address,
            log_dir=self.log_dir,
            intents_dir=self.intents_dir,
            agent_fn=catch_up.wrap_agent_fn_with_saturation_loop(openai_agent(self.model), self.tracker),
            on_new_subject=self.tracker.record_new_activity,
            question_answerer=question_answerer,
        )
        opening = self.tracker.opening_question(self.current_date, memory["last_conversation_date"],
                                                last_topics=self.last_topics,
                                                first_meeting=first_meeting)
        session.open_with(opening)

        self.session = session
        # Last: the GUI shows the timeline as soon as this is set.
        self.last_conversation_date = memory["last_conversation_date"]

        return opening

    def _add_turn(self, speaker: str, utterance: str) -> Dict:
        turn = {"chat": self.chat, "human": self.human, "date": self.date,
                "turn": len(self._pre_turns) + 1, "speaker": speaker, "utterance": utterance}
        self._pre_turns.append(turn)
        return turn

    def _agent_says(self, utterance: str) -> str:
        self._add_turn("agent", utterance)
        self._pre_sources.append("default")
        return utterance
