"""
Shared classes for the chat_session.ipynb / kg_chat_session.ipynb notebooks.

- ChatSession: a live or scripted conversation between a human (a diabetes patient) and an
  agent (a lifestyle coach), turned into the flat turn schema described in chat_session.ipynb.
- KgChatSession(ChatSession): the same, but every turn is also run through SRL extraction and
  pushed into a knowledge graph, with the agent's reply drawn from a knowledge-graph gap when
  one is available. See kg_chat_session.ipynb for the full per-turn flow.

KgChatSession's own dependencies (events_from_chat, chat_from_kg -- cltl.brain, rdflib,
transformers/torch via perspective/emotion_extraction.py, and OPENAI_API_KEY read at *import*
time) are only loaded on first use (_load_kg_dependencies(), called from KgChatSession.__init__),
so `from chat_sessions import ChatSession` alone -- as chat_session.ipynb does -- never pays for
any of that.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import openai
from openai import OpenAI

DEFAULT_MODEL = "gpt-5.1"

# Default per-request timeout (seconds) for every OpenAI call this module -- or its KG
# dependencies (LLM_EventExtraction, LLMTripleReplier) -- make. The SDK's own default has no
# ceiling low enough to catch a genuinely stuck connection in reasonable time: a live chat turn
# was once observed hanging for 5+ minutes with no error before this was added. Passed to every
# OpenAI client this module constructs, directly (_openai_client()) or via
# LLM_EventExtraction/LLMTripleReplier's own `timeout=` constructor parameter (see
# KgChatSession.__init__). See ChatTimeoutError/_call_openai() for what happens once a call
# actually does time out.
DEFAULT_OPENAI_TIMEOUT = 60.0

# Toggle for KgChatSession's per-turn diagnostic log (see its class docstring's "self.turn_log"
# paragraph and _print_log_entry()) -- printed as each entry is built. Flip to False for quiet
# runs; self.turn_log itself is always populated regardless of this flag.
LOG_TURNS = True

# KgChatSession.__init__'s default `gap_activity_types` -- gap-finding (kg_gap_finder queries and
# the follow-up questions built from them) only ever runs for an activity whose own type (the
# Activity.type an extraction gave it -- see _annotate_and_push()/_is_gap_eligible_type()) is in
# this set; anything else (an "other"/unclassified activity, or a type not meant to be probed for
# gaps at all) is simply never queried. These are local names as they appear in the knowledge
# graph's own rdf:type triples under the n2mu namespace (data_type.ActivityType's *values*, with
# any space turned into an underscore -- e.g. "physical condition" -> "physical_condition" -- to
# match events_to_capsules.py's own IRI-safe handling), not the Python enum's member names, which
# occasionally differ (its "exercise" member is spelled correctly; a "excercise" typo here would
# silently match nothing in the graph and exclude every real exercise activity from gap-finding).
# Set before constructing KgChatSession -- e.g. pass a different tuple, or None to disable the
# restriction entirely and consider every activity type -- there's no live control for this one
# (unlike gap_threshold's slider in kg_chat_gui.py): which activity types are worth probing for
# gaps is a modeling decision for the run, not something to flip mid-conversation.
DEFAULT_GAP_ACTIVITY_TYPES = (
    "exercise", "take_food", "take_drink", "symptom", "social_condition",
    "mental_condition", "physical_condition", "treatment", "diet", "medication", "measurement",
    "sleep",
)

# KgIntentChatSession._fetch_gap_queue()'s backstop: the same underlying intent requirement (see
# its own "requirement key" comment there) is asked about at most this many times per chat before
# the intent gives up on it for the rest of the session, regardless of which (possibly
# freshly-minted, unrelated) subject keeps surfacing it. A hard cap independent of
# _handle_intent_answer_reply()'s own direct-answer-capture (see chat_sessions.KgChatSession) --
# that fixes the common case (a short follow-up reply gets correctly attached to the pending gap
# instead of coreferenced onto a new, empty subject); this is the safety net for whatever it still
# misses (e.g. an "unrelated" classification that itself turns out to be another coreference
# miss), so the same question can't repeat indefinitely either way.
MAX_INTENT_GAP_ATTEMPTS = 2


# --------------------------------------------------------------------------- #
# Setup
# --------------------------------------------------------------------------- #

def _load_key() -> str:
    env = os.environ.get("OPENAI_API_KEY")
    if not env:
        raise SystemExit("OPENAI_API_KEY environment variable not set.")
    return env


_client = None

def _openai_client() -> OpenAI:
    global _client
    if _client is None:
        # max_retries=0: a timeout should surface immediately as openai.APITimeoutError, not
        # after the SDK silently retries (its own default: up to 2 more times) -- see
        # _call_openai() below, which is what actually turns that into a message the human sees.
        _client = OpenAI(api_key=_load_key(), timeout=DEFAULT_OPENAI_TIMEOUT, max_retries=0)
    return _client


class ChatTimeoutError(Exception):
    """Raised (by _call_openai(), below) when an OpenAI call this session depends on doesn't
    respond within its timeout. `source` is a short, plain-language description of what was
    being attempted -- e.g. "generating the agent's reply" -- meant to be shown directly to the
    human, which is exactly what ChatSession._timeout_reply() does with it."""

    def __init__(self, source: str):
        self.source = source
        super().__init__(f"timed out: {source}")


def _call_openai(source: str, fn, *args, **kwargs):
    """Call fn(*args, **kwargs) -- any OpenAI-backed call this module or its KG dependencies
    make (agent_fn, LLM_EventExtraction.annotate_new_turn, LLMTripleReplier.reply) -- and
    translate a request that timed out (openai.APITimeoutError, raised once the client's own
    `timeout=`/`max_retries=0` -- see DEFAULT_OPENAI_TIMEOUT -- gives up) into a
    ChatTimeoutError(source), so say() can turn it into a message the human actually sees
    instead of the turn just hanging or crashing with a raw SDK exception.

    Only openai.APITimeoutError is translated -- every other OpenAI error (a bad API key, a
    rate limit, a content-policy rejection, ...) is left to propagate as itself, since labelling
    those "a timeout" would misdescribe what actually happened."""
    try:
        return fn(*args, **kwargs)
    except openai.APITimeoutError as exc:
        raise ChatTimeoutError(source) from exc


def _today() -> str:
    """Default date in the same 'YYYY,Mon,DD' style used by the example chats
    (e.g. '2013,Jan,31')."""
    return datetime.now().strftime("%Y,%b,%d")


def default_system_prompt(human: str) -> str:
    """System prompt for the agent side: a lifestyle coach talking with `human`, a diabetes
    patient -- the same framing prompts.py uses for annotation ('a conversation between a
    diabetes patient and a lifestyle coach').

    Deliberately restrained, NOT advice-giving: this is only ever used for the FALLBACK reply,
    when there's no knowledge-graph gap to build a grounded follow-up question from (see
    chat_sessions.KgChatSession.say()'s "default" reply_sources tag) -- so all it has to go on is
    the bare conversation itself, with no knowledge-graph grounding behind it. Telling the human
    what to do from that alone would be exactly the kind of ungrounded advice this project wants
    the KG-driven flow (intent_gap_finder.py's own questions, or kg_gap_finder.py's) to handle
    instead -- this fallback's job is just to keep the conversation going, not to coach."""
    return (
        f"You are a lifestyle coach talking with {human}, a person with Type 2 diabetes, in a "
        "chat conversation about diet, exercise, sleep, stress and daily routines that affect "
        "their blood sugar management. Keep every reply short (1-2 sentences). Only ever do one "
        "of two things: give plain factual information, or ask a question to learn more detail "
        "about what they just said. Do not give advice, recommendations, or suggestions about "
        "what they should do."
    )


def openai_agent(model: str = DEFAULT_MODEL):
    """Build an agent_fn(messages) -> str backed by the OpenAI chat completions API.
    `messages` is the running list of {'role', 'content'} dicts (system/user/assistant)."""
    def _reply(messages):
        response = _openai_client().chat.completions.create(model=model, messages=messages)
        return response.choices[0].message.content
    return _reply


# --------------------------------------------------------------------------- #
# ChatSession
# --------------------------------------------------------------------------- #

class ChatSession:
    """Holds one conversation. say(utterance) records one human turn, gets one agent turn in
    reply, and appends both to self.turns in the flat schema (see chat_session.ipynb).
    run_interactive() drives that in a loop from typed input, so the whole live chat comes out
    as session.turns when you stop."""

    def __init__(self, chat, human, date=None, agent_fn=None, system_prompt=None):
        """
        chat         -- chat identifier (int or str), shared by every turn in this conversation.
        human        -- the human speaker's name, used both as metadata and as the 'speaker'
                         value on their turns.
        date         -- date the chat took place (defaults to today, '%Y,%b,%d').
        agent_fn     -- callable(messages) -> str that produces the agent's reply given the
                         running OpenAI-style message list. Defaults to openai_agent().
        system_prompt-- system prompt steering the agent; defaults to default_system_prompt(human).
        """
        self.chat = chat
        self.human = human
        self.date = date or _today()
        self.agent_fn = agent_fn or openai_agent()
        self.system_prompt = system_prompt or default_system_prompt(human)
        self._messages = [{"role": "system", "content": self.system_prompt}]
        self.turns = []

    def _add_turn(self, speaker, utterance):
        turn = {
            "chat": self.chat,
            "human": self.human,
            "date": self.date,
            "turn": len(self.turns) + 1,
            "speaker": speaker,
            "utterance": utterance,
        }
        self.turns.append(turn)
        return turn

    def say(self, utterance: str) -> str:
        """Record one human utterance, get the agent's reply, record that too. Returns the
        agent's reply text.

        If generating that reply times out (see ChatTimeoutError/_call_openai()), the agent's
        turn is _timeout_reply()'s message instead -- recorded in self.turns like any other
        reply, but NOT appended to self._messages, since it's a synthetic notice rather than
        something the assistant actually said; the human's own utterance stays in the running
        message history, so their next say() call naturally continues the conversation from
        where it stalled."""
        self._add_turn(self.human, utterance)
        self._messages.append({"role": "user", "content": utterance})

        try:
            reply = _call_openai("generating the agent's reply", self.agent_fn, self._messages)
        except ChatTimeoutError as exc:
            reply = self._timeout_reply(exc)
            self._add_turn("agent", reply)
            return reply

        self._messages.append({"role": "assistant", "content": reply})
        self._add_turn("agent", reply)
        return reply

    def _timeout_reply(self, exc: ChatTimeoutError) -> str:
        """The agent's turn text when producing a real reply timed out (see ChatTimeoutError)
        -- names the step that stalled and asks the human to just try again, instead of leaving
        them staring at a "thinking" indicator (e.g. kg_chat_gui.py's status label) that will
        never resolve on its own. Shared by KgChatSession, which can hit this for several
        different steps (see its own say())."""
        return (
            f"Sorry, there has been a timeout while {exc.source} -- I didn't get a response in "
            "time. Could you please enter your message again?"
        )

    def open_with(self, agent_utterance: str) -> None:
        """Let the agent speak first (turn 1), e.g. a greeting, before any human turn."""
        self._messages.append({"role": "assistant", "content": agent_utterance})
        self._add_turn("agent", agent_utterance)

    def run_interactive(self, quit_words=("quit", "exit", "goodbye", "bye", "stop")) -> list:
        """Live chat loop: prompts for the human's line via input(), prints the agent's
        reply, and keeps going until an empty line or one of `quit_words` is entered.
        Returns self.turns."""
        print(f"Chatting as {self.human} (chat {self.chat!r}, {self.date}). "
              f"Type 'quit' to stop.\n")
        while True:
            utterance = input(f"{self.human}: ").strip()
            if not utterance or utterance.lower() in quit_words:
                break
            reply = self.say(utterance)
            print(f"agent: {reply}\n")
        return self.turns

    def as_conversation(self) -> dict:
        """The nested {chat, human, date, turns:[{turn, speaker, utterance}, ...]} shape
        used as input to annotate_all_turns_in_conversation(), derived from self.turns."""
        return {
            "chat": self.chat,
            "human": self.human,
            "date": self.date,
            "turns": [
                {"turn": t["turn"], "speaker": t["speaker"], "utterance": t["utterance"]}
                for t in self.turns
            ],
        }


# --------------------------------------------------------------------------- #
# Scripted / offline demo helpers
# --------------------------------------------------------------------------- #

def mock_agent(messages):
    """A canned agent_fn(messages) -> str that needs no API key -- useful for testing
    simulate_chat()/ChatSession without network access or an OPENAI_API_KEY."""
    last_human_line = messages[-1]["content"]
    return f"Thanks for sharing that -- tell me more about \"{last_human_line[:40].strip()}...\""


def simulate_chat(chat, human, human_utterances, date=None, agent_fn=None, system_prompt=None) -> list:
    """Run a scripted conversation: the human speaks each line in `human_utterances` in
    turn, the agent replies after every one (via ChatSession.say), and the resulting flat
    list of turns is returned."""
    session = ChatSession(chat=chat, human=human, date=date, agent_fn=agent_fn, system_prompt=system_prompt)
    for utterance in human_utterances:
        session.say(utterance)
    return session.turns


def save_turns(turns: list, path: str) -> None:
    """Write `turns` as JSON to `path`, creating its parent directory first if it doesn't exist
    yet (e.g. `path="intents_log/kg_intent_turns.json"` when nothing has created `intents_log/`
    on this run yet -- a bare filename's parent is `.`, so this is a no-op in that case, same as
    before)."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(turns, f, indent=2)
    print(f"Wrote {len(turns)} turns to {path}")


# --------------------------------------------------------------------------- #
# KgChatSession
# --------------------------------------------------------------------------- #

_KG_DEPS = None

def _load_kg_dependencies():
    """Import LLM_EventExtraction, populate_ekg_from_annotations, kg_gap_finder and
    LLMTripleReplier on first use only (cached in _KG_DEPS after that), so importing this
    module -- or using plain ChatSession -- never requires events_from_chat/chat_from_kg's
    heavier dependencies (cltl.brain, rdflib, transformers/torch via
    perspective/emotion_extraction.py) or OPENAI_API_KEY to already be set
    (llm_event_triples_openai_pydantic.py reads it at import time) until a KgChatSession is
    actually constructed."""
    global _KG_DEPS
    if _KG_DEPS is not None:
        return _KG_DEPS

    import sys
    from pathlib import Path

    def _find_src_dir(*parts) -> Path:
        """Locate src/cltl/<parts...> from the current working directory, so its flat,
        non-package modules can import each other by bare name (they assume their own
        directory is on sys.path, not that they're used as a proper installed package)."""
        for base in (Path.cwd(), *Path.cwd().parents):
            candidate = base.joinpath("src", "cltl", *parts)
            if candidate.is_dir():
                return candidate
        raise FileNotFoundError(
            f"Couldn't locate src/cltl/{'/'.join(parts)} from the current working directory "
            f"({Path.cwd()}); run the notebook from within the kg-chat repo, or adjust "
            f"_find_src_dir()."
        )

    events_from_chat_dir = _find_src_dir("events_from_chat")
    chat_from_kg_dir = _find_src_dir("chat_from_kg")

    # events_from_chat/prompts.py and chat_from_kg/prompts/ are two unrelated, same-named flat
    # modules -- each directory's code assumes it's the only "prompts" on sys.path. Importing
    # both directories at once would make whichever lands first on sys.path silently shadow the
    # other's "prompts". Importing one directory's modules fully, then clearing its "prompts"
    # from sys.modules before adding the other directory, avoids that: each import below
    # resolves "prompts" to the right one.
    sys.path.insert(0, str(events_from_chat_dir))
    # Reads OPENAI_API_KEY at import time -- make sure it's set before this runs.
    from llm_event_triples_openai_pydantic import LLM_EventExtraction
    from populate_ekg import populate_ekg_from_annotations
    sys.path.remove(str(events_from_chat_dir))
    sys.modules.pop("prompts", None)

    sys.path.insert(0, str(chat_from_kg_dir))
    import kg_gap_finder
    import intent_gap_finder
    from llm_triple_replier import LLMTripleReplier
    from prompts.response_processor import AGENT_PREDICATES

    _KG_DEPS = {
        "LLM_EventExtraction": LLM_EventExtraction,
        "populate_ekg_from_annotations": populate_ekg_from_annotations,
        "kg_gap_finder": kg_gap_finder,
        "intent_gap_finder": intent_gap_finder,
        "LLMTripleReplier": LLMTripleReplier,
        "AGENT_PREDICATES": AGENT_PREDICATES,
    }
    return _KG_DEPS


# Role fields an SRLAnnotation carries, matching events_to_capsules.ROLE_FIELDS_WITH_TYPE plus
# "result" -- used only by _extraction_log_triples(), below, to build the human-readable
# "triples_pushed" view in KgChatSession.turn_log; NOT used to build the actual RDF pushed to
# the graph (that stays events_to_capsules.get_triples_with_types_and_activity_id()'s job).
_LOG_ROLE_FIELDS = (
    "agent", "patient", "agent_patient", "experiencer", "participant",
    "qualification", "instrument", "location", "result",
)


def _extraction_log_triples(extraction) -> list:
    """A simplified, human-readable [{"subject", "predicate", "object"}, ...] view of one
    SRLAnnotation's role fillers -- for KgChatSession.turn_log only. Deliberately NOT a
    reconstruction of the exact RDF actually pushed (see events_to_capsules.py for that): a role
    filler's raw phrase is shown as-is, not resolved to a pronoun's identity or linked via a
    real URI the way the real push does -- close enough to see, at a glance, what a turn told
    the knowledge graph, without duplicating (and risking drifting out of sync with)
    events_to_capsules.py's own triple-building logic just to log it.
    """
    activity = extraction.activity
    subject = activity.value or activity.activity_id
    triples = [
        {"subject": subject, "predicate": role, "object": filler.value}
        for role in _LOG_ROLE_FIELDS
        for filler in (getattr(extraction, role, None) or [])
    ]
    triples.extend(
        {"subject": subject, "predicate": "time", "object": time_span.value}
        for time_span in (extraction.time or [])
    )
    return triples


class KgChatSession(ChatSession):
    """A ChatSession that also runs SRL extraction and knowledge-graph population after every
    new turn, incrementally (see kg_chat_session.ipynb for the full per-turn flow).

    self.annotations holds every non-empty annotation entry produced so far ({chat, date, human,
    Input, Output} dicts, Output still the raw SRLAnnotation Pydantic objects). self.kg_pushes
    holds one populate_ekg_from_annotations() summary per turn actually pushed to the graph
    (turns with no extractions produce neither). self.reply_sources holds one "gap" / "default" /
    "kg_answer" / "timeout" tag per agent reply, so you can see which replies came from a
    knowledge-graph gap, a generic LLM reply, an answer to the human's question looked up in the
    knowledge graph (see `question_answerer`), or a step that timed out (see
    ChatTimeoutError/say()'s own docstring).

    Gap questions are drawn from self._gap_queues (per-activity-instance queues of gaps still to
    ask about) and never repeat a gap already recorded in self._asked_gap_keys -- see
    _next_gap()/_reply_from_gaps(). When the gap just asked about was an agent/agent_patient
    confirmation (see prompts.response_processor.get_prompt_for_agent_gap()), self._pending_confirmation
    holds it until the human's very next turn, which is then routed to
    _handle_confirmation_reply() instead of the normal annotate-and-push flow -- see say().

    self.last_subject_uris holds the KG subject URI(s) most recently touched by a push -- the
    human turn's own extraction (say()) or a confirmation-driven push (_push_gap_triple()) --
    kept unchanged on a turn that pushes nothing, so it always names the activity most relevant
    right now. Read by kg_chat_gui.py's graph panel to know which activity to display/link to;
    of no interest if you're not driving this session through that GUI.

    self.gap_activity_types (see DEFAULT_GAP_ACTIVITY_TYPES) restricts gap-finding to activities
    of an allow-listed type -- a turn's own new subjects are still pushed to the graph and still
    update self.last_subject_uris either way, but only the eligible ones (_is_gap_eligible_type())
    are ever passed to _reply_from_gaps()/_fetch_gap_queue(), so an activity of some other type
    never gets a gap-driven follow-up question. None disables the restriction (every type is
    eligible). self._subject_types (subject_uri -> its type's local name, populated as a side
    effect of _annotate_and_push()) is what that check reads.

    self.turn_log holds one diagnostic entry per turn (both the human's and the agent's, same
    turns/indexing as self.turns), each a
    {"turn", "speaker", "utterance", "triples_pushed", "gap_queries", "selected_gap"} dict:
      - triples_pushed: a simplified [{"subject", "predicate", "object"}, ...] view of what THIS
        turn told the knowledge graph (see _extraction_log_triples()) -- NOT the exact RDF (see
        events_to_capsules.py for that), just enough to see at a glance what was asserted.
      - gap_queries: one [{"subject", "threshold", "found", "after_dedup"}, ...] entry per FRESH
        kg_gap_finder query actually run while producing this turn's reply (empty if none was --
        e.g. an already-queued gap was used instead, see _next_gap(), or this is the agent's own
        turn, which never triggers gap-finding at all).
      - selected_gap: the {"subject", "predicate", "kind", "peer_coverage"} gap this turn's reply
        was about -- freshly found this turn, or (for a turn that answers a pending confirmation)
        the one being resolved -- or None if the reply wasn't gap-driven at all.
    Printed as each entry is built when LOG_TURNS is True (the module default); always populated
    into self.turn_log regardless. kg_chat_gui.py's save_session() writes it to a third JSON file
    alongside the turns/stats ones on quit, when driving this session through that GUI.
    """

    def __init__(self, chat, human, kg_address, log_dir="kg_logs", date=None, agent_fn=None,
                 system_prompt=None, extractor=None, replier=None, gap_threshold=0.3,
                 gap_activity_types=DEFAULT_GAP_ACTIVITY_TYPES, clear_all=False,
                 on_new_subject=None, question_answerer=None):
        super().__init__(chat=chat, human=human, date=date, agent_fn=agent_fn, system_prompt=system_prompt)
        deps = _load_kg_dependencies()
        self._populate_ekg_from_annotations = deps["populate_ekg_from_annotations"]
        self._kg_gap_finder = deps["kg_gap_finder"]
        self._agent_predicates = deps["AGENT_PREDICATES"]

        self.kg_address = kg_address
        self.log_dir = log_dir
        # Optional callable(subject_uri, activity_type_local_name) -- called exactly once per
        # genuinely NEW activity_id (see _annotate_and_push()'s own "is_new" check), the first
        # time it's ever seen a recognized type, never again on a later turn that just adds
        # another role to one already known. The only integration point catch_up_from_kg.py's
        # SaturationTracker needs with this module: it passes its own record_new_activity() here
        # to count, live, how many of each gap-period topic have actually been reported so far
        # this session -- see that module's own docstring. None (the default) is a no-op.
        self.on_new_subject = on_new_subject
        # Optional kg_question_answerer.KgQuestionAnswerer -- when given, a human question about
        # their own recorded history is answered from the knowledge graph instead of being
        # extracted and pushed as if it were a report (see say()). None (the default) disables it.
        self.question_answerer = question_answerer
        # DEFAULT_OPENAI_TIMEOUT -- see there, and ChatTimeoutError/_call_openai() -- for why.
        self.extractor = extractor or deps["LLM_EventExtraction"](timeout=DEFAULT_OPENAI_TIMEOUT)
        self.replier = replier if replier is not None else deps["LLMTripleReplier"](
            backend="openai", timeout=DEFAULT_OPENAI_TIMEOUT
        )
        self.gap_threshold = gap_threshold
        # None means "no restriction" (every activity type is gap-eligible); otherwise a set of
        # local names -- see DEFAULT_GAP_ACTIVITY_TYPES/_is_gap_eligible_type().
        self.gap_activity_types = None if gap_activity_types is None else set(gap_activity_types)
        # subject_uri -> its activity type's local name, e.g. "take_food" -- populated as a side
        # effect of _annotate_and_push() (only when the extraction actually carried a type), read
        # by _is_gap_eligible_type(). Never removed, so it also still has the answer for a
        # subject_uri from many turns ago.
        self._subject_types = {}
        # subject_uri -> its activity's own label/phrase, e.g. "headache" -- populated the same
        # way as self._subject_types (only when the extraction actually carried a non-empty
        # `activity.value`, so a later phrase-less reference to an already-known activity never
        # overwrites a real label with nothing). Unused by KgChatSession itself; read by
        # KgIntentChatSession/intent_gap_finder.find_intent() to disambiguate between several
        # intents that share one activity_type (e.g. several distinct symptoms, all typed
        # "symptom" -- see intent_gap_finder.py's own module docstring).
        self._subject_labels = {}
        self.annotations = []
        self.kg_pushes = []
        self.reply_sources = []
        # Per-activity-instance queues of (gap, kind) pairs still to turn into a follow-up
        # question, keyed by subject_uri, most-affected first -- and the set of (subject,
        # predicate) gaps already asked about, ever, so none is repeated. See _next_gap().
        self._gap_queues = {}
        self._asked_gap_keys = set()
        # (gap, kind, question_text) for an agent-confirmation question just asked, awaiting the
        # human's answer -- see _reply_from_gaps()/_handle_confirmation_reply(). None whenever
        # there isn't one pending.
        self._pending_confirmation = None
        # (gap, kind, question_text) for any OTHER intent-driven gap question just asked whose
        # gap carries a "fill_role" (see intent_gap_finder._make_gap()'s own docstring) -- awaiting
        # the human's answer, same idea as self._pending_confirmation but for non-agent-like
        # requirements (what/how much/where, not who) -- see _reply_from_gaps()/
        # _handle_intent_answer_reply(). Mutually exclusive with self._pending_confirmation (a
        # gap is routed to at most one of the two -- see _reply_from_gaps()). Always None for a
        # plain KgChatSession, since kg_gap_finder.py's own gaps never carry a "fill_role".
        self._pending_intent_answer = None
        # See the class docstring -- kept as [] until the first successful push.
        self.last_subject_uris = []
        # See the class docstring's "self.turn_log" paragraph. _pending_gap_queries/
        # _last_selected_gap/_pending_triples_pushed are transient, reset at the start of
        # _reply_from_gaps()/_handle_confirmation_reply() and read back by say() right after --
        # not part of the public per-session state.
        self.turn_log = []
        self._pending_gap_queries = []
        self._last_selected_gap = None
        self._pending_triples_pushed = []
        if clear_all:
            # Wipe the graph once, up front, instead of on every incremental push below.
            self._populate_ekg_from_annotations([], kg_address=kg_address, log_dir=log_dir, clear_all=True)

    def say(self, utterance: str) -> str:
        """See ChatSession.say(). Additionally: if any OpenAI-backed step involved in producing
        this turn's reply times out (confirmation classification/acknowledgement, SRL
        extraction, or gap-question/default reply generation -- see ChatTimeoutError), the
        agent's turn is _timeout_reply()'s message instead, naming that specific step, and
        nothing past that point runs for this turn (notably, the agent's own reply is never
        itself annotated/pushed below -- there's no real content worth extracting meaning from).
        reply_sources still gets exactly one entry either way, tagged "timeout" in that case.

        A timeout while CLASSIFYING a pending confirmation reply (as opposed to acknowledging
        it) restores self._pending_confirmation first -- see _handle_confirmation_reply() -- so
        the human's retry is still routed as answering that same confirmation, not treated as an
        unrelated fresh utterance. The same applies to self._pending_intent_answer/
        _handle_intent_answer_reply() (see there).

        Once a real reply exists, annotating/pushing the AGENT's own turn (the last step, purely
        bookkeeping for future gap-finding/extractor context) is handled separately: a timeout
        there is only logged to stdout, never allowed to discard an already-delivered reply.

        Builds one self.turn_log entry for the human's turn (see the class docstring's
        "self.turn_log" paragraph) either way -- including on a timeout, so every turn in
        self.turns has a matching entry -- and a second, gap-free one for the agent's own turn
        once/if it's successfully annotated.
        """
        human_turn = self._add_turn(self.human, utterance)
        self._messages.append({"role": "user", "content": utterance})

        # Reset here, not just inside _reply_from_gaps()/_handle_confirmation_reply(): if a
        # timeout hits BEFORE either of those runs (e.g. during the human turn's own
        # extraction), these would otherwise still carry stale values from a PREVIOUS turn's
        # _reply_from_gaps() call into this turn's log entry.
        self._pending_gap_queries = []
        self._last_selected_gap = None
        human_triples_pushed = []

        try:
            if self._pending_confirmation is not None:
                # This turn answers the confirmation question asked last agent turn -- handle it
                # as confirm/deny/deny+correct instead of running it through the general-purpose
                # SRL extractor, which isn't built to make sense of a bare "yes"/"no, my son"
                # reply (see _handle_confirmation_reply()/_push_gap_triple()). The KG side-effect
                # (if any) happens explicitly in there, not via _annotate_and_push().
                reply = self._handle_confirmation_reply(human_turn)
                source_tag = "gap"
                human_triples_pushed = self._pending_triples_pushed
            else:
                # Same idea, generalized to any OTHER intent-driven gap question (see
                # self._pending_intent_answer's own comment) -- but here the classifier can also
                # come back UNRELATED (the reply neither answers nor declines it, e.g. a
                # clarifying question back), in which case _handle_intent_answer_reply() has
                # already cleared the pending state and returns (False, None) to say "run this
                # turn through the normal flow below instead, exactly as if there had been no
                # pending intent answer at all."
                handled, intent_answer_reply = (
                    self._handle_intent_answer_reply(human_turn)
                    if self._pending_intent_answer is not None else (False, None)
                )
                kg_answer = None
                if not handled and self.question_answerer is not None:
                    # A question about the human's own record ("what did I eat yesterday?") is
                    # answered from the graph, and neither it nor its answer is extracted/pushed:
                    # the question reports nothing new, and the answer only restates what is
                    # already there (pushing it would mint duplicate activities).
                    kg_answer = self.question_answerer.try_answer(utterance, self._messages)
                if handled:
                    reply = intent_answer_reply
                    source_tag = "gap"
                    human_triples_pushed = self._pending_triples_pushed
                elif kg_answer is not None:
                    reply = kg_answer
                    source_tag = "kg_answer"
                else:
                    # Record + annotate + push the human turn FIRST: whether the agent's reply is
                    # a KG-grounded gap question depends on what (if anything) this turn just added.
                    _, new_subjects, human_triples_pushed = self._annotate_and_push(human_turn)
                    if new_subjects:
                        self.last_subject_uris = new_subjects
                    # last_subject_uris (above, for the graph panel) and triples_pushed cover
                    # EVERY new subject regardless of type; gap-finding itself only runs for the
                    # gap_activity_types-eligible ones -- see _is_gap_eligible_type().
                    gap_eligible_subjects = [u for u in new_subjects if self._is_gap_eligible_type(u)]
                    reply = self._reply_from_gaps(gap_eligible_subjects) if gap_eligible_subjects else None
                    source_tag = "gap" if reply is not None else "default"
                    if reply is None:
                        reply = _call_openai("generating the agent's reply", self.agent_fn, self._messages)
        except ChatTimeoutError as exc:
            reply = self._timeout_reply(exc)
            self.reply_sources.append("timeout")
            self._add_turn("agent", reply)
            self._log_turn(
                human_turn, triples_pushed=human_triples_pushed,
                gap_queries=self._pending_gap_queries, selected_gap=self._last_selected_gap,
            )
            return reply

        self.reply_sources.append(source_tag)
        self._log_turn(
            human_turn, triples_pushed=human_triples_pushed,
            gap_queries=self._pending_gap_queries, selected_gap=self._last_selected_gap,
        )

        self._messages.append({"role": "assistant", "content": reply})
        agent_turn = self._add_turn("agent", reply)
        if source_tag == "kg_answer":
            self._log_turn(agent_turn, triples_pushed=[])
            return reply
        try:
            _, _, agent_triples_pushed = self._annotate_and_push(agent_turn)
            self._log_turn(agent_turn, triples_pushed=agent_triples_pushed)
        except ChatTimeoutError as exc:
            # This is annotating the agent's OWN reply, already delivered above -- purely
            # bookkeeping for future gap-finding/extractor context, not something the human is
            # waiting on. Never discard a real, already-returned reply over a background miss
            # like this one; just note it so it's visible in the notebook's own output.
            print(f"[KgChatSession] timed out {exc.source} -- the agent's own turn was not "
                  f"annotated/pushed to the knowledge graph.")
            self._log_turn(agent_turn, triples_pushed=[])
        return reply

    def _log_turn(self, turn: dict, triples_pushed: list, gap_queries: list = None, selected_gap: dict = None) -> dict:
        """Build, append to self.turn_log, and (if LOG_TURNS) print one diagnostic entry for
        `turn` -- see the class docstring's "self.turn_log" paragraph. gap_queries/selected_gap
        default to "nothing happened" (always true for the agent's own turn, which never
        triggers gap-finding)."""
        entry = {
            "turn": turn["turn"], "speaker": turn["speaker"], "utterance": turn["utterance"],
            "triples_pushed": list(triples_pushed), "gap_queries": list(gap_queries or []),
            "selected_gap": selected_gap,
        }
        self.turn_log.append(entry)
        if LOG_TURNS:
            self._print_log_entry(entry)
        return entry

    @staticmethod
    def _print_log_entry(entry: dict) -> None:
        print(f"[turn {entry['turn']}] {entry['speaker']}: {entry['utterance']}")
        if entry["triples_pushed"]:
            print(f"    pushed {len(entry['triples_pushed'])} triple(s):")
            for t in entry["triples_pushed"]:
                print(f"      {t['subject']}  {t['predicate']}  =  {t['object']}")
        for q in entry["gap_queries"]:
            found = q["found"]
            print(
                f"    gap query: subject={q['subject']} threshold={q['threshold']:.2f} "
                f"found(B={found['predicate_gaps']}, D={found['predicate_object_gaps']}, "
                f"E={found['predicate_object_instances_gaps']}) after_dedup={q['after_dedup']}"
            )
        if entry["selected_gap"]:
            g = entry["selected_gap"]
            print(
                f"    selected gap: subject={g['subject']} predicate={g['predicate']} "
                f"kind={g['kind']} peer_coverage={g['peer_coverage']}"
            )

    def open_with(self, agent_utterance: str) -> None:
        super().open_with(agent_utterance)
        turn = self.turns[-1]
        try:
            _, _, triples_pushed = self._annotate_and_push(turn)
        except ChatTimeoutError as exc:
            print(f"[KgChatSession] timed out {exc.source} -- open_with()'s turn was not "
                  f"annotated/pushed to the knowledge graph.")
            triples_pushed = []
        self._log_turn(turn, triples_pushed=triples_pushed)

    def _annotate_and_push(self, turn: dict):
        """Annotate one new turn and, if it produced anything, push it to the knowledge graph
        right away. Returns (summary, subject_uris, triples_pushed): summary is the
        populate_ekg_from_annotations() result, subject_uris the KG subject URI of each
        activity/condition just asserted, triples_pushed a turn_log-shaped (see the class
        docstring/_extraction_log_triples()) view of what was pushed -- (None, [], []) if the
        turn produced no extractions.

        Raises ChatTimeoutError (see _call_openai()) if the extractor's own OpenAI call times
        out -- named "extracting meaning from your message" for the human's turn, "...from the
        agent's reply" for the agent's own (say() calls this for both)."""
        turn_for_extractor = {"turn": turn["turn"], "speaker": turn["speaker"], "utterance": turn["utterance"]}
        whose = "your message" if turn["speaker"] == self.human else "the agent's reply"
        entry = _call_openai(
            f"extracting meaning from {whose}",
            self.extractor.annotate_new_turn, self.chat, self.human, self.date, turn_for_extractor,
        )
        if entry is None:
            return None, [], []
        self.annotations.append(entry)

        subject_uris = [
            "http://cltl.nl/leolani/n2mu/" + extraction.activity.activity_id
            for extraction in entry["Output"]
            if extraction.activity and extraction.activity.activity_id
        ]
        # For _is_gap_eligible_type() -- the local name as it actually appears in the graph's own
        # rdf:type triples (events_to_capsules.py appends Activity.type.value as-is to the pushed
        # type list, and downstream capsule/RDF building turns any space in it into an underscore
        # the same way it does for every other label-derived URI, e.g. "physical condition" ->
        # ".../n2mu/physical_condition"). Left unset (never a KEY in self._subject_types) when an
        # extraction had no Activity.type at all -- _is_gap_eligible_type() treats that as NOT
        # eligible, same as an explicitly out-of-list type.
        for extraction in entry["Output"]:
            activity = extraction.activity
            if activity and activity.activity_id and activity.type:
                uri = "http://cltl.nl/leolani/n2mu/" + activity.activity_id
                is_new_subject = uri not in self._subject_types
                self._subject_types[uri] = activity.type.value.replace(" ", "_")
                # Fired exactly once per genuinely NEW activity_id -- never again on a later turn
                # that just adds another role to one already known -- see self.on_new_subject's
                # own comment in __init__.
                if is_new_subject and self.on_new_subject:
                    self.on_new_subject(uri, self._subject_types[uri])
            # self._subject_labels -- see its own comment in __init__. Only set when this
            # extraction actually carried a phrase of its own: a later, phrase-less reference to
            # an already-known activity (activity.value is None -- see
            # events_to_capsules.get_triples_with_types_and_activity_id()'s own comment on this)
            # must never overwrite an earlier real label with nothing.
            if activity and activity.activity_id and activity.value:
                uri = "http://cltl.nl/leolani/n2mu/" + activity.activity_id
                self._subject_labels[uri] = activity.value
        triples_pushed = [t for extraction in entry["Output"] for t in _extraction_log_triples(extraction)]

        # annotate_new_turn()'s "Output" is a list of SRLAnnotation Pydantic objects;
        # populate_ekg_from_annotations' capsule building (events_to_capsules.py) reads them as
        # plain dicts (event_data.get(...)), so dump them first.
        pushable_entry = {**entry, "Output": [extraction.model_dump(mode="json") for extraction in entry["Output"]]}
        summary = self._populate_ekg_from_annotations(
            [[pushable_entry]],  # one conversation, containing just this one new turn's entry
            kg_address=self.kg_address,
            log_dir=self.log_dir,
            clear_all=False,
        )
        self.kg_pushes.append(summary)
        return summary, subject_uris, triples_pushed

    def _is_gap_eligible_type(self, subject_uri) -> bool:
        """True if `subject_uri` may be used for gap-finding -- see self.gap_activity_types/
        DEFAULT_GAP_ACTIVITY_TYPES. self.gap_activity_types is None means the restriction is
        off (every subject is eligible); otherwise `subject_uri` must have a KNOWN type (recorded
        in self._subject_types by _annotate_and_push()) that's in that set -- a subject whose
        type was never captured at all (e.g. its extraction had no Activity.type) is NOT
        eligible, same as one with an explicitly out-of-list type."""
        if self.gap_activity_types is None:
            return True
        return self._subject_types.get(subject_uri) in self.gap_activity_types

    @staticmethod
    def _gap_key(gap: dict):
        """Identifies "the same gap" for _asked_gap_keys: subject + predicate, regardless of
        which of kg_gap_finder's three gap kinds (B/D/E) surfaced it or what the expected
        object/type was -- a follow-up about a subject's missing `location` is the same
        question whether it's phrased as "you have no location at all" (B) or "your location
        isn't of the type most peers have" (D/E), so asking it once is enough."""
        return gap["subject"], gap["predicate"]

    def _fetch_gap_queue(self, subject_uri):
        """Run kg_gap_finder ONCE against this session's knowledge graph for `subject_uri`, and
        return its gaps as a queue: most-affected (highest missing_count) first, with anything
        already in self._asked_gap_keys filtered out. This is the only place that actually
        launches a kg_gap_finder query for one instance -- _next_gap() works through the result
        one gap per turn, across as many turns as it takes, before calling this again for the
        same subject_uri.

        Also appends one entry to self._pending_gap_queries (see the class docstring's
        "self.turn_log" paragraph / _reply_from_gaps(), which resets that list before the whole
        _next_gap() call this participates in) -- so this is exactly, and only, where a
        turn_log "gap_queries" entry comes from: one real kg_gap_finder query, not the cheaper
        "drain an already-fetched queue" path _next_gap() otherwise takes."""
        report = self._kg_gap_finder.analyze(
            endpoint=self.kg_address, subject_uri=subject_uri,
            threshold=self.gap_threshold, print_output=False,
        )
        found = (
            [(g, "predicate") for g in report["predicate_gaps"]]
            + [(g, "predicate_object_type") for g in report["predicate_object_gaps"]]
            + [(g, "predicate_object_instance") for g in report["predicate_object_instances_gaps"]]
        )
        found = [(g, kind) for g, kind in found if self._gap_key(g) not in self._asked_gap_keys]
        found.sort(key=lambda pair: -pair[0]["missing_count"])
        self._pending_gap_queries.append({
            "subject": subject_uri,
            "threshold": self.gap_threshold,
            "found": {
                "predicate_gaps": len(report["predicate_gaps"]),
                "predicate_object_gaps": len(report["predicate_object_gaps"]),
                "predicate_object_instances_gaps": len(report["predicate_object_instances_gaps"]),
            },
            "after_dedup": len(found),
        })
        return found

    def _next_gap(self, subject_uris):
        """The next (gap, kind) to turn into a follow-up question, or None if there isn't one.

        For each subject_uri (in order), gaps already queued for it (self._gap_queues) are
        worked through in sequence -- one per call, oldest-fetched batch first -- before a fresh
        kg_gap_finder query is launched for that same instance again; a fresh query only ever
        happens once its previous queue has been fully drained. A gap already asked about (see
        self._asked_gap_keys) is skipped wherever it resurfaces and is never selected again,
        even via a different instance's queue or a later requery of the same one.
        """
        for subject_uri in subject_uris:
            queue = self._gap_queues.get(subject_uri)
            if not queue:  # None (never fetched) or already fully drained -- fetch a fresh batch
                queue = self._fetch_gap_queue(subject_uri)
                self._gap_queues[subject_uri] = queue
            while queue:
                gap, kind = queue.pop(0)
                if self._gap_key(gap) in self._asked_gap_keys:
                    continue  # asked meanwhile via a different instance's queue
                return gap, kind
        return None

    def _reply_from_gaps(self, subject_uris):
        """Turn the next knowledge-graph gap around subject_uris (see _next_gap()) into a
        natural-language follow-up question via LLMTripleReplier. Returns None (so the caller
        falls back to the default agent_fn reply) if there are no gaps, or no replier
        configured. A gap on the `agent`/`agent_patient` role is asked as a yes/no confirmation
        that `self.human` themselves was the agent, rather than an open "who did this" question
        -- see prompts.response_processor.get_prompt_for_kg_gap()'s `human` parameter -- and, in
        that case, arms self._pending_confirmation so the human's next turn is routed to
        _handle_confirmation_reply() instead of the normal annotate-and-push flow (see say()).

        A gap that carries a "fill_role" (see intent_gap_finder._make_gap()'s own docstring --
        never true for a kg_gap_finder.py gap, only ever for an intent_gap_finder.py one) arms
        self._pending_intent_answer instead, the same idea generalized to a what/how much/where
        requirement instead of a who one -- see _handle_intent_answer_reply(). The two are
        mutually exclusive: an agent-like gap is always routed to the confirmation path, never
        this one, regardless of whether it also happens to carry a "fill_role".

        Resets, then populates, self._pending_gap_queries/self._last_selected_gap for say()'s
        turn_log entry -- see the class docstring's "self.turn_log" paragraph. Both stay at their
        reset values ([] / None) if this returns None."""
        self._pending_gap_queries = []
        self._last_selected_gap = None
        if self.replier is None or not subject_uris:
            return None
        next_gap = self._next_gap(subject_uris)
        if next_gap is None:
            return None
        gap, kind = next_gap
        self._last_selected_gap = {
            "subject": gap["subject"], "predicate": gap["predicate"], "kind": kind,
            "peer_coverage": gap.get("peer_coverage"),
        }
        self._asked_gap_keys.add(self._gap_key(gap))
        prompt = self.replier._processor.get_prompt_for_kg_gap(gap, kind, human=self.human)
        reply = _call_openai("generating a knowledge-graph follow-up question", self.replier.reply, prompt)
        if gap["predicate"] in self._agent_predicates:
            self._pending_confirmation = (gap, kind, reply)
        elif gap.get("fill_role"):
            self._pending_intent_answer = (gap, kind, reply)
        return reply

    # ------------------------------------------------------------------- #
    # Handling the human's answer to a pending agent-confirmation question
    # ------------------------------------------------------------------- #

    def _classify_confirmation_reply(self, question: str, reply_text: str):
        """Classify a human reply to a pending yes/no agent-confirmation question (see
        _reply_from_gaps()/prompts.response_processor.get_prompt_for_agent_gap()) as one of:
          - ("confirm", None) -- the assumed statement is correct.
          - ("deny", None) -- denied, with no correct information given instead.
          - ("deny_correct", <value>) -- denied AND corrected in the same reply; <value> is that
            corrected information.
        Falls back to ("deny", None) for anything the classifier LLM doesn't return in one of
        the three expected forms (see get_instruct_for_confirmation_response()), or a
        "CORRECT:" with no usable value after it -- the safe choice, since it never pushes an
        unverified triple to the KG, only asks the human to clarify."""
        prompt = self.replier._processor.get_prompt_for_confirmation_response(question, reply_text)
        raw = (_call_openai(
            "interpreting your answer to the confirmation question", self.replier.reply, prompt
        ) or "").strip()
        upper = raw.upper()
        if upper.startswith("CONFIRM"):
            return "confirm", None
        if upper.startswith("CORRECT"):
            value = raw.split(":", 1)[1].strip() if ":" in raw else ""
            if value:
                return "deny_correct", value
        return "deny", None

    def _push_gap_triple(self, gap: dict, role_value: str, role_type: str, human_turn: dict,
                          role_name: str = None):
        """Build a synthetic Output entry that adds exactly one role filler (gap's predicate) to
        gap's subject activity -- a bare reference to that already-known activity_id, not a new
        activity -- and push it to the KG via the normal capsule pipeline
        (populate_ekg_from_annotations(), the same one every other turn's SRL extraction goes
        through). Used to fill a gap directly from the human's answer to a confirmation
        question, instead of relying on the general-purpose SRL extractor to make sense of a
        short "yes"/"no, my son"-style reply that names no activity of its own.

        role_value="I" is a deliberate special case: get_triples_with_types_and_activity_id()'s
        pronoun resolution (events_to_capsules._role_filler_object()) turns it into the KG
        identity of `self.human` -- the "I" pronoun resolves to whoever is speaking, and this
        entry's synthetic Input turn is always attributed to the human -- the same way it would
        if the human had literally said "I" in an extracted utterance, so confirming an assumed
        agent links to the human's own URI instead of a bare, unlinked "I" literal.

        role_name defaults to gap["predicate"]'s own local name (the confirmation-flow case,
        where that predicate is always a real, single RDF predicate URI, e.g. n2mu:agent) --
        pass it explicitly for a gap whose "predicate" field is a display string, not a real
        predicate (e.g. a qualification-aspect gap's own aspect name, "duration"), where the
        real predicate to write to is instead named by the gap's own "fill_role" (see
        intent_gap_finder._make_gap()) -- see _handle_intent_answer_reply().
        """
        activity_id = gap["subject"].rsplit("/", 1)[-1]
        role_name = role_name or gap["predicate"].rsplit("/", 1)[-1]
        output_entry = {
            "perspective": {"emotion": "neutral", "factuality": "confirm", "certainty": "certain"},
            "activity": {"activity_id": activity_id, "value": None, "offset": None, "length": None, "type": None},
            "agent": [], "patient": [], "agent_patient": [], "experiencer": [], "participant": [],
            "qualification": [], "instrument": [], "location": [], "result": [], "time": [], "time_resolved": [],
        }
        output_entry[role_name] = [{"value": role_value, "type": role_type, "offset": 0, "length": len(role_value)}]
        entry = {
            "chat": self.chat, "date": self.date, "human": self.human,
            "Input": {"turn": human_turn["turn"], "speaker": human_turn["speaker"], "utterance": human_turn["utterance"]},
            "Output": [output_entry],
        }
        self.annotations.append(entry)
        summary = self._populate_ekg_from_annotations(
            [[entry]], kg_address=self.kg_address, log_dir=self.log_dir, clear_all=False,
        )
        self.kg_pushes.append(summary)
        self.last_subject_uris = [gap["subject"]]
        triples_pushed = [{"subject": activity_id, "predicate": role_name, "object": role_value}]
        return summary, triples_pushed

    def _continue_after_ack(self, ack: str) -> str:
        """Append a natural, forward-driving follow-up after a bare acknowledgement (see
        _handle_confirmation_reply()'s confirm/deny_correct branches and
        _handle_intent_answer_reply()'s answer/decline branches) by calling self.agent_fn
        directly -- the exact same call say() would otherwise make once a turn falls through to
        the "default" reply case, just made explicitly here so an acknowledgement is never the
        WHOLE reply with nothing to keep the conversation moving.

        This is what fixes a real failure mode: without it, a reply like "Thanks for letting me
        know" (a plain decline ack) was the entire turn, and the conversation only continued once
        the human happened to say something new on their own -- which, from their side, looked
        like the agent had simply stopped. For a KgIntentChatSession whose agent_fn is
        catch_up_from_kg.wrap_agent_fn_with_saturation_loop(), calling it here immediately surfaces
        the NEXT still-uncovered catch-up topic (or nothing extra, once every topic really is
        covered) instead of waiting for another turn to notice. For a plain agent_fn it's simply
        the ordinary default reply -- harmless, and keeps any chat moving the same way.

        A timeout getting the follow-up is swallowed (logged, not raised) -- the acknowledgement
        itself is already valid and worth keeping; losing the bonus continuation on a slow request
        isn't worth discarding it or surfacing a scary timeout message for what is, from the
        human's point of view, an answer that already went through fine."""
        try:
            followup = _call_openai("continuing the conversation", self.agent_fn, self._messages)
        except ChatTimeoutError:
            print("[KgChatSession] timed out generating a follow-up after the acknowledgement -- "
                  "keeping the acknowledgement alone for this turn.")
            return ack
        followup = (followup or "").strip()
        if not followup or followup == ack:
            return ack
        return f"{ack} {followup}"

    def _handle_confirmation_reply(self, human_turn: dict) -> str:
        """Handle the human's answer to self._pending_confirmation (armed by _reply_from_gaps()):
          - confirm -> push the originally-assumed triple (subject, predicate, self.human) to
            the KG, filling the gap, acknowledge, and continue (_continue_after_ack()) -- never
            just the bare acknowledgement alone.
          - deny + correction (both in the same reply) -> push the CORRECTED triple instead,
            acknowledge, and continue the same way.
          - deny, no correction -> push nothing; fall back to the open (non-confirmation)
            question for the same gap, so the human can just answer it directly next (already
            forward-driving on its own -- no _continue_after_ack() needed here).

        If classifying the reply times out, self._pending_confirmation is restored (the human's
        answer was never actually read) before the ChatTimeoutError propagates to say(), so a
        retry is still routed here rather than treated as an unrelated fresh utterance. A timeout
        while acknowledging an already-pushed confirm/correction is NOT restored -- the KG write
        already happened; only the acknowledgement text itself is missing, and say() turns that
        ChatTimeoutError into a plain "please try again" message like any other.

        For say()'s turn_log entry (see the class docstring's "self.turn_log" paragraph): sets
        self._last_selected_gap to the gap being RESOLVED here (not freshly found -- it was
        selected on an earlier turn, see _reply_from_gaps()), self._pending_gap_queries to []
        (no fresh kg_gap_finder query happens on this path), and self._pending_triples_pushed to
        whatever _push_gap_triple() ends up pushing (empty on a plain denial)."""
        gap, kind, question = self._pending_confirmation
        self._pending_confirmation = None
        self._pending_gap_queries = []
        self._last_selected_gap = {
            "subject": gap["subject"], "predicate": gap["predicate"], "kind": kind,
            "peer_coverage": gap.get("peer_coverage"),
        }
        self._pending_triples_pushed = []
        try:
            verdict, correction = self._classify_confirmation_reply(question, human_turn["utterance"])
        except ChatTimeoutError:
            self._pending_confirmation = (gap, kind, question)
            raise

        if verdict == "confirm":
            _, self._pending_triples_pushed = self._push_gap_triple(
                gap, role_value="I", role_type="person", human_turn=human_turn
            )
            ack_prompt = self.replier._processor.get_prompt_for_gap_filled_ack(gap, self.human)
            ack = _call_openai("acknowledging your answer", self.replier.reply, ack_prompt)
            return self._continue_after_ack(ack)

        if verdict == "deny_correct":
            _, self._pending_triples_pushed = self._push_gap_triple(
                gap, role_value=correction, role_type="person", human_turn=human_turn
            )
            ack_prompt = self.replier._processor.get_prompt_for_gap_filled_ack(gap, correction)
            ack = _call_openai("acknowledging your correction", self.replier.reply, ack_prompt)
            return self._continue_after_ack(ack)

        # Plain denial, no usable correction -- ask the open question instead (no `human=`, so
        # get_prompt_for_kg_gap() does NOT take the confirmation branch this time).
        prompt = self.replier._processor.get_prompt_for_kg_gap(gap, kind)
        return _call_openai("asking a follow-up question", self.replier.reply, prompt)

    # ------------------------------------------------------------------- #
    # Handling the human's answer to a pending (non-agent-like) intent gap question
    # ------------------------------------------------------------------- #

    def _classify_intent_answer_reply(self, question: str, reply_text: str):
        """Classify a human reply to a pending intent-driven gap question (see
        _reply_from_gaps()'s self._pending_intent_answer branch) as one of:
          - ("answer", <value>) -- the reply answers the question; <value> is that answer.
          - ("decline", None) -- the human indicated they don't have/didn't do/don't know this.
          - ("unrelated", None) -- neither -- e.g. a clarifying question back, or a change of
            subject (see get_instruct_for_intent_answer_response()'s own docstring on why this
            third case matters here in a way it doesn't for a plain yes/no confirmation).
        Falls back to ("unrelated", None) for anything the classifier LLM doesn't return in one
        of the three expected forms, or an "ANSWER:" with no usable value after it -- the safe
        choice, since it never pushes an unverified value to the KG, only lets the turn run
        through the ordinary extraction pipeline instead."""
        prompt = self.replier._processor.get_prompt_for_intent_answer_response(question, reply_text)
        raw = (_call_openai(
            "interpreting your answer to the follow-up question", self.replier.reply, prompt
        ) or "").strip()
        upper = raw.upper()
        if upper.startswith("ANSWER"):
            value = raw.split(":", 1)[1].strip() if ":" in raw else ""
            if value:
                return "answer", value
        if upper.startswith("DECLINE"):
            return "decline", None
        return "unrelated", None

    def _handle_intent_answer_reply(self, human_turn: dict):
        """Handle the human's answer to self._pending_intent_answer (armed by _reply_from_gaps()
        for a gap that carries a "fill_role" -- see intent_gap_finder._make_gap()'s own
        docstring), generalizing _handle_confirmation_reply()'s idea (attach the answer directly
        to the pending gap's own subject instead of relying on the general-purpose SRL extractor
        to coreference a short follow-up reply back to it) to any what/how much/where
        requirement, not just a who one:
          - answer -> push the value onto the gap's own "fill_role" (see _push_gap_triple()),
            acknowledge (get_prompt_for_gap_filled_ack(), the same one the confirmation flow
            uses), and continue (_continue_after_ack()) rather than stopping at the bare ack.
          - decline -> push nothing, drop the requirement, acknowledge that instead of asking
            again (get_prompt_for_gap_declined_ack()), and continue the same way
            (_continue_after_ack()) -- a bare "no worries, thanks for letting me know" with
            nothing after it is exactly what used to make the conversation look like it had
            simply stopped, even though say() was still working correctly turn to turn. Unlike a
            plain agent-confirmation denial, this deliberately does NOT fall back to re-asking the
            SAME question: a generic what/how much/where requirement has no better "who" pivot to
            ask instead -- _continue_after_ack() is what supplies a genuinely NEW thread instead
            (see KgIntentChatSession's own MAX_INTENT_GAP_ATTEMPTS backstop for the belt-and-braces
            version of this, covering the cases this classifier itself might still get wrong).
          - unrelated -> push nothing, ask nothing here; returns (False, None) so say() runs the
            turn through the normal annotate-and-push/gap-finding flow instead, exactly as if
            there had been no pending intent answer at all (e.g. "what body function?" -- a
            clarifying question back, not an answer or a decline).

        Returns (handled, reply): (True, <text>) for answer/decline, (False, None) for unrelated.
        If classifying the reply times out, self._pending_intent_answer is restored (the human's
        answer was never actually read) before the ChatTimeoutError propagates to say(), same
        reasoning as _handle_confirmation_reply()'s own restore.

        For say()'s turn_log entry: sets self._last_selected_gap to the gap being resolved here,
        self._pending_gap_queries to [] (no fresh gap-finder query happens on this path), and
        self._pending_triples_pushed to whatever _push_gap_triple() ends up pushing (empty on a
        decline or an unrelated reply)."""
        gap, kind, question = self._pending_intent_answer
        self._pending_intent_answer = None
        self._pending_gap_queries = []
        self._last_selected_gap = {
            "subject": gap["subject"], "predicate": gap["predicate"], "kind": kind,
            "peer_coverage": gap.get("peer_coverage"),
        }
        self._pending_triples_pushed = []
        try:
            verdict, value = self._classify_intent_answer_reply(question, human_turn["utterance"])
        except ChatTimeoutError:
            self._pending_intent_answer = (gap, kind, question)
            raise

        if verdict == "answer":
            _, self._pending_triples_pushed = self._push_gap_triple(
                gap, role_value=value, role_type=gap.get("fill_role_type"), human_turn=human_turn,
                role_name=gap["fill_role"],
            )
            ack_prompt = self.replier._processor.get_prompt_for_gap_filled_ack(gap, value)
            ack = _call_openai("acknowledging your answer", self.replier.reply, ack_prompt)
            return True, self._continue_after_ack(ack)

        if verdict == "decline":
            ack_prompt = self.replier._processor.get_prompt_for_gap_declined_ack(gap)
            ack = _call_openai(
                "acknowledging that you don't have this", self.replier.reply, ack_prompt
            )
            return True, self._continue_after_ack(ack)

        return False, None


# --------------------------------------------------------------------------- #
# KgIntentChatSession
# --------------------------------------------------------------------------- #

class KgIntentChatSession(KgChatSession):
    """A KgChatSession whose follow-up questions come from hand-authored intents (intents/*.json
    at the project root, see chat_from_kg/intent_gap_finder.py) instead of kg_gap_finder's
    peer-statistics-based gaps -- see kg_intent_chat.ipynb for the full per-turn flow.

    kg_gap_finder's own gaps only fire once a MAJORITY of an activity's peers already have the
    predicate in question, so the very first "take_food" activity ever pushed to the graph can
    never produce one -- intents state what's expected directly, independent of how many similar
    activities already exist, which is exactly why this exists as a separate session class
    rather than just a different gap_activity_types/gap_threshold on KgChatSession.

    Only _is_gap_eligible_type(), _fetch_gap_queue() and _print_log_entry() (for its
    differently-shaped turn_log "gap_queries" entries -- no peer_coverage/threshold, since
    there's no peer voting here) differ from KgChatSession; everything else -- annotating and
    pushing turns to the graph, agent-confirmation handling, timeouts, the rest of turn_log --
    is unchanged, since a filled-in intent gap is annotated back onto the graph exactly the same
    way an ordinary kg_gap_finder gap's answer is (the human's plain-text reply to the follow-up
    question runs through the normal incremental extractor next turn -- see the class docstring
    on cross-turn coreference); nothing here needs its own push path.

    Two mechanisms guard against the same intent question repeating even though it was already
    answered (or declined) -- the incremental SRL extractor doesn't reliably coreference a short
    follow-up reply ("30 minutes", "no") back onto the SAME activity/role the question was about,
    which otherwise lets the identical gap resurface, unrecognized, on a freshly-minted subject
    next turn:
      - KgChatSession._handle_intent_answer_reply() (armed via a gap's own "fill_role" -- see
        intent_gap_finder._make_gap()) attaches the human's very next reply directly to the
        PENDING gap's own subject/role instead of relying on that coreference at all -- the
        primary fix, for the common case.
      - _fetch_gap_queue()'s own MAX_INTENT_GAP_ATTEMPTS backstop caps how many times the SAME
        underlying requirement (independent of which subject surfaces it) is ever asked about
        this chat, regardless -- the safety net for whatever the first mechanism still misses.

    self.intents holds the loaded intent definitions (see intent_gap_finder.load_intents()) --
    an activity type NOT covered by any of them (intent_gap_finder.find_intent() returns None)
    is simply never gap-driven: its turns are still annotated/pushed to the graph like any
    other, but the agent's reply for it always falls back to the plain LLM agent_fn -- "if there
    is no intent match, use the LLM response". This includes the case where several intents
    share one activity_type (see this project's own symptom_intents.json, where several
    different symptoms all share the generic "symptom" ActivityType) but the activity's own
    label -- self._subject_labels, inherited from KgChatSession -- doesn't match any of the
    matching intents' "activity_labels": find_intent() deliberately treats that as no match too,
    rather than guessing one of them (see its own docstring).
    """

    def __init__(self, chat, human, kg_address, log_dir="kg_logs", date=None, agent_fn=None,
                 system_prompt=None, extractor=None, replier=None, intents=None, intents_dir=None,
                 clear_all=False, on_new_subject=None, question_answerer=None):
        """Same parameters as KgChatSession, minus gap_threshold/gap_activity_types (not
        meaningful here -- there's no peer voting, and eligibility is decided by intent match,
        not an allow-list), plus:

        intents      -- pre-loaded list of intent dicts (see intent_gap_finder.load_intents()).
                         Loaded from `intents_dir` (or the project's own intents/ folder, if
                         `intents_dir` is also omitted) when not given.
        intents_dir  -- directory of intent *.json files, passed to
                         intent_gap_finder.load_intents() when `intents` isn't given directly.
        """
        super().__init__(
            chat=chat, human=human, kg_address=kg_address, log_dir=log_dir, date=date,
            agent_fn=agent_fn, system_prompt=system_prompt, extractor=extractor, replier=replier,
            gap_threshold=0.0, gap_activity_types=None, clear_all=clear_all,
            on_new_subject=on_new_subject, question_answerer=question_answerer,
        )
        # There's no peer voting here (see the class docstring), so gap_threshold is never read
        # by anything this class actually does -- dropped, not just left at 0.0, so
        # kg_chat_gui.py's ChatWindow (which shows its "Gap sensitivity" slider purely based on
        # hasattr(session, "gap_threshold")) doesn't offer a control that would silently do
        # nothing for this session type.
        del self.gap_threshold
        deps = _load_kg_dependencies()
        self._intent_gap_finder = deps["intent_gap_finder"]
        self.intents = intents if intents is not None else self._intent_gap_finder.load_intents(intents_dir)
        # (id(intent), kind, gap["predicate"]) -> how many times that requirement has been
        # surfaced so far this chat -- see MAX_INTENT_GAP_ATTEMPTS/_fetch_gap_queue(). Keyed by
        # id(intent) (object identity, stable for this session's own self.intents list) rather
        # than activity_type alone, so several distinct intents that share one activity_type
        # (e.g. symptom_intents.json's several label-disambiguated symptom intents -- see
        # find_intent()) each get their own independent attempt count, instead of e.g. a
        # "headache" duration question and an unrelated "dizziness" duration question wrongly
        # sharing one counter just because both are typed "symptom".
        self._intent_requirement_attempts = {}

    def _is_gap_eligible_type(self, subject_uri) -> bool:
        """Overrides KgChatSession's allow-list check: eligible exactly when
        intent_gap_finder.find_intent() finds a matching intent for this subject's activity type
        (and, if several intents share that type, its own label -- e.g. several distinct
        symptoms all typed "symptom") -- see the class docstring's "no intent match -> LLM"
        fallback."""
        activity_type = self._subject_types.get(subject_uri)
        activity_label = self._subject_labels.get(subject_uri)
        return self._intent_gap_finder.find_intent(activity_type, self.intents, activity_label=activity_label) is not None

    def _fetch_gap_queue(self, subject_uri):
        """Overrides KgChatSession's kg_gap_finder-based version: runs
        intent_gap_finder.next_intent_gap() for the one intent matching this subject's activity
        type/label (guaranteed to exist -- say() only calls this for
        _is_gap_eligible_type()-eligible subjects) instead of a peer-statistics kg_gap_finder
        query. An intent's own checks already run in priority order and stop at the first unmet
        one (see next_intent_gap()'s docstring), so the "queue" this returns is always length 0
        or 1 -- _next_gap() drains it exactly the same way regardless.

        MAX_INTENT_GAP_ATTEMPTS backstop: a found gap's "requirement key" --
        (id(intent), kind, gap["predicate"]) -- identifies the SAME underlying requirement (e.g.
        "measurement's missing patient", "exercise's duration") independent of which subject_uri
        surfaced it. Unlike self._asked_gap_keys (scoped to one subject_uri, so a freshly-minted,
        unrelated subject always looks like a brand-new gap -- exactly the repeated-question
        failure mode this exists to catch), this counter persists across every subject that ever
        surfaces this requirement for the rest of the chat. Once it's been surfaced
        MAX_INTENT_GAP_ATTEMPTS times, the intent gives up on it silently (found stays empty,
        "gave_up": True is logged for visibility) -- the activity's own turns are still
        annotated/pushed to the graph as normal, only this one requirement stops being asked
        about."""
        activity_type = self._subject_types.get(subject_uri)
        activity_label = self._subject_labels.get(subject_uri)
        intent = self._intent_gap_finder.find_intent(activity_type, self.intents, activity_label=activity_label)
        found = []
        gave_up = False
        if intent is not None:
            graph = self._kg_gap_finder.load_graph_from_endpoint(self.kg_address)
            next_gap = self._intent_gap_finder.next_intent_gap(graph, subject_uri, intent, activity_type=activity_type)
            if next_gap is not None:
                gap, kind = next_gap
                requirement_key = (id(intent), kind, gap["predicate"])
                attempts = self._intent_requirement_attempts.get(requirement_key, 0)
                if attempts >= MAX_INTENT_GAP_ATTEMPTS:
                    gave_up = True
                else:
                    self._intent_requirement_attempts[requirement_key] = attempts + 1
                    found = [next_gap]
        found = [(g, kind) for g, kind in found if self._gap_key(g) not in self._asked_gap_keys]
        self._pending_gap_queries.append({
            "subject": subject_uri,
            "activity_type": activity_type,
            "intent_source": intent.get("_source_file") if intent else None,
            "after_dedup": len(found),
            "gave_up": gave_up,
        })
        return found

    @staticmethod
    def _print_log_entry(entry: dict) -> None:
        """Like KgChatSession._print_log_entry(), but for this class's own "gap_queries" entry
        shape (subject/activity_type/intent_source/after_dedup -- no peer-voting fields to show,
        since there's no peer comparison here)."""
        print(f"[turn {entry['turn']}] {entry['speaker']}: {entry['utterance']}")
        if entry["triples_pushed"]:
            print(f"    pushed {len(entry['triples_pushed'])} triple(s):")
            for t in entry["triples_pushed"]:
                print(f"      {t['subject']}  {t['predicate']}  =  {t['object']}")
        for q in entry["gap_queries"]:
            gave_up_note = " GAVE UP (max attempts reached)" if q.get("gave_up") else ""
            print(
                f"    intent gap query: subject={q['subject']} activity_type={q['activity_type']} "
                f"intent={q['intent_source']} after_dedup={q['after_dedup']}{gave_up_note}"
            )
        if entry["selected_gap"]:
            g = entry["selected_gap"]
            print(f"    selected gap: subject={g['subject']} predicate={g['predicate']} kind={g['kind']}")
