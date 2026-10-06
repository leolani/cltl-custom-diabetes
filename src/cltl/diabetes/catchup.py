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

The code doing the actual work lives unchanged in src/kg_chat (copied from
cltl-kg-driven-chat/notebooks) and src/cltl/{chat_from_kg,events_from_chat,
gaps_from_kg} (copied from cltl-kg-driven-chat/src). It locates its own
src/cltl and intents/ directories by walking up from the working directory,
so the process has to run from this module's root directory, the same
requirement src/main.py already has for config/.
"""
import logging
import os
import re
import sys
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from cltl.diabetes.api import Conversation, ConversationFactory

logger = logging.getLogger(__name__)

# src/kg_chat holds chat_sessions.py and catch_up_from_kg.py, which import each
# other by bare module name, as they did in the notebooks/ directory they were
# copied from.
_KG_CHAT_DIR = Path(__file__).resolve().parents[2] / "kg_chat"


def _load_kg_chat():
    if str(_KG_CHAT_DIR) not in sys.path:
        sys.path.insert(0, str(_KG_CHAT_DIR))
    import chat_sessions
    import catch_up_from_kg
    import kg_question_answerer

    return chat_sessions, catch_up_from_kg, kg_question_answerer


class CatchUpConversation(Conversation):
    def __init__(self, scenario_id: str, human: str, brain, kg_address: str, intents,
                 log_dir: Path, fallback_gap_days: int, model: Optional[str],
                 max_answer_words: Optional[int] = None):
        chat_sessions, catch_up, kg_question_answerer = _load_kg_chat()
        self._chat_sessions = chat_sessions
        self._catch_up = catch_up

        self.scenario_id = scenario_id
        self.human = human
        self._log_dir = log_dir

        self.current_date = datetime.now()
        fallback_date = self.current_date - timedelta(days=fallback_gap_days)
        self.last_conversation_date = catch_up.find_last_conversation_date(
            human, brain, self.current_date, fallback_date)
        self.catch_up_topics = catch_up.find_catch_up_topics(
            brain, self.current_date, self.last_conversation_date)
        logger.info("Catch-up for %s in scenario %s: last conversation %s, topics %s",
                    human, scenario_id, self.last_conversation_date,
                    [(t["activity_type"], t["expected_count"]) for t in self.catch_up_topics])

        model = model or chat_sessions.DEFAULT_MODEL
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
        )

    def open(self) -> Optional[str]:
        try:
            opening = self.tracker.opening_question(self.current_date, self.last_conversation_date)
        except self._chat_sessions.ChatTimeoutError as e:
            logger.warning("Timed out %s, opening with a plain greeting instead", e.source)
            opening = f"Hi {self.human}, good to talk to you again! How have you been since we last spoke?"

        self.session.open_with(opening)

        return opening

    def respond(self, utterance: str) -> Optional[str]:
        return self.session.say(utterance)

    def close(self) -> None:
        if not self.session.turns:
            return

        self._chat_sessions.save_turns(self.session.turns, str(self._turns_path()))
        self._catch_up.save_intent_log(self.session, self.tracker, self.catch_up_topics,
                                       self.current_date, self.last_conversation_date,
                                       log_dir=str(self._log_dir / "intents_log"))

    def _turns_path(self) -> Path:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"chat{self.session.chat}_{self.human}")
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        return self._log_dir / "chat_logs" / f"{name}_turns_{stamp}.json"


class CatchUpConversationFactory(ConversationFactory):
    def __init__(self, kg_address: str, intents_dir: str, log_dir: str,
                 fallback_gap_days: int = 7, model: Optional[str] = None,
                 max_answer_words: Optional[int] = None):
        # The kg-chat code reads the key at import time and calls sys.exit()
        # when it is missing, which would take down the service thread in the
        # middle of a conversation instead of at start-up.
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError("OPENAI_API_KEY is not set; cltl-custom-diabetes needs it "
                             "for event extraction and reply generation")
        if not kg_address or kg_address.startswith("$"):
            raise ValueError(f"[cltl.diabetes] kg_address is not set ({kg_address!r}); "
                             "set kg_server and kg_repository, or CLTL_KG_ADDRESS to the "
                             "GraphDB repository's SPARQL endpoint")

        self._kg_address = kg_address
        self._intents_dir = intents_dir
        self._log_dir = Path(log_dir)
        self._fallback_gap_days = fallback_gap_days
        self._model = model
        self._max_answer_words = max_answer_words

        self._lock = threading.Lock()
        self._brain = None
        self._intents = None

    def create(self, scenario_id: str, human: str) -> Conversation:
        brain, intents = self._resources()

        return CatchUpConversation(scenario_id, human, brain, self._kg_address, intents,
                                   self._log_dir, self._fallback_gap_days, self._model,
                                   max_answer_words=self._max_answer_words)

    def _resources(self):
        """Connect to the knowledge graph on first use, not at start-up, so the
        module comes up even while GraphDB is not (yet) reachable. A failed
        connection is not cached; the next conversation tries again."""
        with self._lock:
            if self._brain is None:
                chat_sessions, catch_up, _ = _load_kg_chat()
                self._brain = catch_up.connect_brain(self._kg_address,
                                                     log_dir=str(self._log_dir / "kg_logs"))
                # chat_sessions imports intent_gap_finder lazily, together
                # with the other knowledge graph dependencies.
                intent_gap_finder = chat_sessions._load_kg_dependencies()["intent_gap_finder"]
                self._intents = intent_gap_finder.load_intents(self._intents_dir)
                logger.info("Connected to %s, loaded %s intents from %s",
                            self._kg_address, len(self._intents), self._intents_dir)

            return self._brain, self._intents
