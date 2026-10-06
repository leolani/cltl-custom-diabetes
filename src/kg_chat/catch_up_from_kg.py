"""
catch_up_from_kg.py
=====================

The "what happened since we last spoke" phase for kg_catchup_intent_chat.ipynb, built on top of
src/cltl/gaps_from_kg/get_temporal_containers.py's brain-side temporal queries
(LongTermMemory._submit_query()) -- a different query layer over the same GraphDB repository than
kg_gap_finder.py's rdflib/SPARQL-endpoint queries, which is what the rest of chat_sessions.py's
per-turn intent-driven gap-finding uses.

This isn't just a one-shot opening line: it drives the WHOLE conversation, in a loop, until the
CATCH-UP PERIOD (see below) has as much knowledge behind it as this human's own history says is
typical -- not "ask once per topic and move on regardless." The loop:

  1. Ask about a catch-up-period topic (SaturationTracker.next_question()) -- one of
     DEFAULT_GAP_ACTIVITY_TYPES the human has real history with but not yet ENOUGH reported for
     this period (see "Saturation", below).
  2. Whatever activity/condition the human reports in reply is handled entirely by
     chat_sessions.KgIntentChatSession's own EXISTING per-turn flow, completely unchanged: SRL
     extraction -> push to the KG -> intent_gap_finder.next_intent_gap() keeps asking follow-up
     questions about THAT SAME activity (what/how much/when/where) for as long as its own matching
     intent still has unmet requirements.
  3. Once that activity's own follow-ups are exhausted (say() falls through to the default
     agent_fn reply -- see chat_sessions.KgChatSession.say()'s "default" reply_sources tag), go
     back to step 1 -- another topic still short of saturation, or the SAME one again if it still
     is -- unless every topic has reached saturation, in which case replies fall through to the
     wrapped agent_fn unchanged and the conversation continues normally.

**The catch-up period is capped at MAX_SATURATION_GAP_DAYS (14 days)**, even when the real gap
since find_last_conversation_date() is much longer -- a human who hasn't talked in two months
still only needs the last two weeks caught up on live, not the whole two months (find_catch_up_topics()
computes this as `effective_recent_date = max(recent_date, current_date - MAX_SATURATION_GAP_DAYS)`).

**Saturation** -- "enough knowledge for the [capped] catch-up period", this module's whole goal --
is defined per topic against a fixed WEEKLY baseline, not a baseline tied to however long the
catch-up period itself happens to be: `_windowed_average_rate()`, always called with a 7-day
(`CATCH_UP_WINDOW_DAYS`) window, tiles that week-long window backwards across the human's own
history (everything before the catch-up period) to get their typical weekly rate for that topic,
which `find_catch_up_topics()` then scales to however many days the (capped) catch-up period
actually covers. If they've historically reported "exercise" an average of 3 times per week, and
the catch-up period is 2 weeks, 6 exercise activities reported live this session is "enough" --
not "however many kg_gap_finder.py or intent_gap_finder.py happen to ask about", and not "exactly
1, regardless of how often this person usually reports it."

Module contents:

- connect_brain()/ensure_role_hierarchy() -- see their own docstrings: makes sure
  n2mu_sem_roles.py's rdfs:subPropertyOf mapping is uploaded, without which
  get_temporal_containers() finds no date/actor/place for any real activity at all.
- find_last_conversation_date() -- get_temporal_containers.get_last_conversation_date().
- find_catch_up_topics() -- one entry per DEFAULT_GAP_ACTIVITY_TYPES type with real history,
  carrying both a concrete example (latest_label/latest_date), its own weekly baseline
  (weekly_rate) and saturation target (expected_count) for the capped catch-up period, and
  however much of that period the KG's own "gap" bucket already covers before this conversation
  even starts (initial_reported_count).
- SaturationTracker -- holds those targets plus how many of each topic have actually been
  reported LIVE this session (self.reported, updated via record_new_activity()), decides what's
  still worth asking about (next_question()) and when the whole loop is done (is_saturated()).
  Also builds the very first turn (opening_question()), the one-time wrap-up
  (wrap_up_message()), and answers an explicit "what have we discussed?" (recap_message()) --
  see _is_recap_request().
- wrap_agent_fn_with_saturation_loop() -- the ONLY integration point with chat_sessions.py: wraps
  a plain agent_fn so a still-unsaturated topic's question is asked instead of the plain default
  reply, until SaturationTracker.is_saturated() (at which point wrap_up_message() runs once), and
  so an explicit recap request (_is_recap_request()) is answered directly, ahead of both, without
  touching any tracker state. Nothing in chat_sessions.py needs to change for THIS -- agent_fn is
  already a pluggable constructor parameter of every ChatSession -- but
  SaturationTracker.record_new_activity() needs to be told about every new activity as it's
  pushed, which DOES need one small, additive hook: chat_sessions.KgChatSession's own
  `on_new_subject` constructor parameter (see its docstring), called exactly once per genuinely
  NEW activity_id (never on a later turn that just adds another role to one already known).
- save_intent_log() -- writes one JSON file per chat under notebooks/intents_log/ (see its own
  docstring)
  summarizing the whole session after it ends: the gap itself, the catch-up topics identified
  before the chat began, how each of them actually fared live, and every intent_gap_finder.py
  intent the per-turn flow consulted along the way.
"""

import json
import math
import re
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from chat_sessions import DEFAULT_GAP_ACTIVITY_TYPES, DEFAULT_MODEL, _call_openai, _openai_client


def _find_repo_src_dir() -> Path:
    """The repo's own src/ directory (parent of the cltl/ namespace package), found by walking up
    from the current working directory -- added to sys.path so `import
    cltl.gaps_from_kg.get_temporal_containers` resolves via Python's namespace-package merging:
    cltl/ itself has no __init__.py here (it's shared with the separately pip-installed
    cltl.brain/cltl.commons/... packages), and PEP 420 merges every "cltl" directory found on
    sys.path into that one namespace, so adding src/ is enough -- no need to import gaps_from_kg's
    own directory directly the way chat_sessions._load_kg_dependencies() does for
    chat_from_kg/events_from_chat's bare-name-importing flat modules."""
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "src"
        if (candidate / "cltl" / "gaps_from_kg").is_dir():
            return candidate
    raise FileNotFoundError(
        "Couldn't locate src/cltl/gaps_from_kg from the current working directory "
        f"({Path.cwd()}); run this notebook from within the kg-chat repo."
    )


_GAPS_DEPS = None


def _load_gaps_from_kg():
    """Import get_temporal_containers and LongTermMemory on first use only -- mirrors
    chat_sessions._load_kg_dependencies()'s own "don't pay for cltl.brain/rdflib unless this
    feature is actually used" reasoning."""
    global _GAPS_DEPS
    if _GAPS_DEPS is not None:
        return _GAPS_DEPS
    src_dir = _find_repo_src_dir()
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    import cltl.gaps_from_kg.get_temporal_containers as get_temporal_containers
    import cltl.gaps_from_kg.n2mu_sem_roles as n2mu_sem_roles
    from cltl.brain.long_term_memory import LongTermMemory
    _GAPS_DEPS = {
        "get_temporal_containers": get_temporal_containers,
        "n2mu_sem_roles": n2mu_sem_roles,
        "LongTermMemory": LongTermMemory,
    }
    return _GAPS_DEPS


def ensure_role_hierarchy(kg_address: str) -> bool:
    """Make sure `kg_address` has the n2mu_sem_roles.py role mappings uploaded -- see that
    module's own docstring for why get_temporal_containers()/get_last_conversation_date() find
    nothing at all without them (thought_util.get_sem_relation_query() only ever sees an
    activity's actor/place/time through those `rdfs:subPropertyOf` triples). A no-op if they're
    already there (n2mu_sem_roles.role_hierarchy_uploaded()). Returns True if an upload actually
    happened, False if the mapping was already present."""
    deps = _load_gaps_from_kg()
    roles = deps["n2mu_sem_roles"]
    if roles.role_hierarchy_uploaded(kg_address):
        return False
    roles.upload_role_hierarchy(kg_address)
    return True


def connect_brain(kg_address: str, log_dir: str = "kg_logs"):
    """One LongTermMemory brain connection to `kg_address` -- the exact same address
    chat_sessions.KgChatSession/KgIntentChatSession's own kg_address points
    populate_ekg_from_annotations() at (see events_from_chat/populate_ekg.py), just queried
    through cltl.brain's own SPARQL layer instead of kg_gap_finder.py's rdflib one. Never clears
    the graph (clear_all=False, unconditionally) -- this module only ever reads.

    Also calls ensure_role_hierarchy() -- every caller of connect_brain() goes on to run
    find_last_conversation_date()/find_catch_up_topics(), both built on
    thought_util.get_sem_relation_query(), so there's no point ever connecting without it."""
    deps = _load_gaps_from_kg()
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ensure_role_hierarchy(kg_address)
    return deps["LongTermMemory"](address=kg_address, log_dir=Path(log_dir), clear_all=False)


def find_last_conversation_date(human: str, brain, current_date: datetime,
                                 fallback_date: datetime) -> datetime:
    """The most recent date `human` is on record as having spoken in this KG at all, via
    get_temporal_containers.get_last_conversation_date() -- `fallback_date` is used as-is when the
    KG has no prior utterance from them (e.g. a brand new human, or a fresh/empty graph)."""
    deps = _load_gaps_from_kg()
    return deps["get_temporal_containers"].get_last_conversation_date(
        human, brain, current_date, fallback_date
    )


# The baseline unit find_catch_up_topics() measures history against: a WEEKLY average, always --
# see _windowed_average_rate()/find_catch_up_topics()'s own docstrings for why this is now fixed
# at 7 days regardless of how long the actual gap since the last conversation is.
CATCH_UP_WINDOW_DAYS = 7

# find_catch_up_topics() never tries to saturate more than this many days back from `current_date`
# -- if the real gap since recent_date is longer (a human who hasn't talked in months), only the
# most recent slice of it is what THIS session's saturation loop tries to fill in; everything
# further back than that still counts toward the historical weekly-average baseline (see
# find_catch_up_topics()), it just isn't itself a target to catch up on live.
MAX_SATURATION_GAP_DAYS = 14

# How many of the REAL running conversation's own messages (self._messages, as
# chat_sessions.ChatSession keeps it -- see wrap_agent_fn_with_saturation_loop(), which is what
# actually has access to it and passes it through) SaturationTracker.next_question()/
# wrap_up_message() include as context ahead of their own instruction. Without this, both used
# to build a totally ISOLATED [system, user] prompt from scratch every time -- meaning the LLM
# deciding whether to keep asking questions or wrap up and say goodbye had NO WAY to see anything
# the human actually just said. Concretely: a real session had the human type "What's next?"
# (an explicit request to keep going) immediately before wrap_up_message() ran, and the reply
# said goodbye anyway -- not because the LLM ignored the human, but because it was never shown
# that message at all. Any "system"-role entries in the tail are dropped (see _recent_context())
# since this module supplies its own.
RECENT_CONTEXT_MESSAGES = 6


def _windowed_average_rate(history_dates: List[datetime], window_days: int,
                            series_end: datetime) -> float:
    """The average number of `history_dates` per non-overlapping `window_days`-long window,
    tiling BACKWARDS from `series_end` through the earliest of `history_dates` -- "the average
    frequency of this topic in `window_days`-long periods before `series_end`". find_catch_up_topics()
    always calls this with `window_days=CATCH_UP_WINDOW_DAYS` (a WEEKLY average, per this module's
    own saturation goal -- see its docstring), independent of how long the actual catch-up period
    being saturated is (see MAX_SATURATION_GAP_DAYS) -- decoupling the baseline measurement unit
    from the target period's own length is what lets a capped-at-14-days catch-up period still be
    compared against a genuinely long, representative slice of history.

    A history spanning less than one full window still counts as exactly one (mostly-empty)
    window rather than being skipped or extrapolated -- an infrequent topic's genuinely low rate
    is real information (e.g. "this person mentions treatment once every couple of months"), not
    something to inflate by only counting "full" windows. Returns 0.0 for no history at all (the
    caller -- find_catch_up_topics() -- never calls this for a topic with none anyway).
    """
    if not history_dates:
        return 0.0
    window = timedelta(days=max(window_days, 1))
    earliest = min(history_dates)
    span = series_end - earliest
    num_windows = max(1, math.ceil(span / window)) if span > timedelta(0) else 1
    counts = []
    window_end = series_end
    for _ in range(num_windows):
        window_start = window_end - window
        counts.append(sum(1 for d in history_dates if window_start <= d < window_end))
        window_end = window_start
    return sum(counts) / len(counts)


def find_catch_up_topics(brain, current_date: datetime, recent_date: datetime,
                          activity_types=DEFAULT_GAP_ACTIVITY_TYPES) -> List[Dict]:
    """One entry per `activity_types` local name (default: chat_sessions.DEFAULT_GAP_ACTIVITY_TYPES
    -- the same set per-turn intent gap-finding already restricts itself to) the human has ever
    discussed before -- either get_temporal_containers.get_temporal_containers()'s "history"
    bucket (something with a resolvable date) or its "unknown" one (discussed, but never with a
    date `_parse_event_time()` could resolve -- see below) is non-empty -- sorted
    most-recently-discussed first. A type in NEITHER bucket is left out entirely -- there's
    nothing to calibrate a saturation target against, let alone catch up on.

    The period this session actually tries to saturate is capped at MAX_SATURATION_GAP_DAYS (14):
    `effective_recent_date = max(recent_date, current_date - MAX_SATURATION_GAP_DAYS)` -- the
    LATER of the true last-conversation date or 14 days ago, so a human who genuinely last talked
    3 days ago still only needs to cover those 3 days, while one who hasn't talked in 2 months
    only needs to cover the last 14 -- not the whole 2-month gap. `get_temporal_containers()` is
    called with THIS date as its own "recent_date", so its own "history"/"gap" bucketing lines up
    exactly: "history" becomes everything before the catch-up window (still this human's full
    real history, just minus whatever falls inside the last 14 days), and "gap" becomes exactly
    the catch-up window itself.

    Each entry: {"activity_type", "history_count", "latest_label", "latest_date", "weekly_rate",
    "expected_count", "initial_reported_count"}:

    - "weekly_rate" (see _windowed_average_rate(), always called with `window_days=
      CATCH_UP_WINDOW_DAYS`) -- how many of this topic's activities this human TYPICALLY reports
      per week, based on the average across every 7-day window found tiling backwards through
      their own history before the catch-up window -- "the weekly average of activities and
      conditions reported in the past".
    - "expected_count" -- "weekly_rate" scaled to however many days the (capped) catch-up window
      actually covers: `weekly_rate * effective_gap_days / CATCH_UP_WINDOW_DAYS`, rounded, at
      least 1 whenever there's any dated history at all so even an infrequent topic still gets
      asked about once. This is SaturationTracker's own per-topic target. A topic with only
      undated ("unknown") mentions has no weekly_rate to scale at all (`0.0`) -- expected_count
      falls back to a plain `1` for it instead, the same "ask about it once" default every topic
      used before saturation targets existed.
    - "initial_reported_count" -- however many of this topic's activities are ALREADY in the KG's
      own "gap" bucket (i.e. dated within the capped catch-up window) before this conversation
      even starts, e.g. from data pushed through some other channel. Seeded into
      SaturationTracker.reported so a topic that's already partly (or fully) covered needs that
      much LESS asked about live, instead of double-counting it.
    """
    deps = _load_gaps_from_kg()
    gtc = deps["get_temporal_containers"]
    effective_recent_date = max(recent_date, current_date - timedelta(days=MAX_SATURATION_GAP_DAYS))
    effective_gap_days = max((current_date.date() - effective_recent_date.date()).days, 1)
    topics = []
    for activity_type in activity_types:
        history, gap, future, unknown = gtc.get_temporal_containers(
            brain, current_date, effective_recent_date, activity_type="n2mu:" + activity_type
        )
        # get_temporal_containers() itself now falls back to an activity's own CONVERSATION date
        # (gaf:denotedIn -> sem:hasBeginTimeStamp) whenever none of its own time values resolve to
        # a real calendar date (a bare "for an hour"/"recently"-style phrase, not an actual date),
        # so a mention lands in `unknown` only in the rarer case where even that utterance-level
        # timestamp is missing. Still handled defensively here rather than assumed away: a topic
        # counts as worth catching up on whenever EITHER bucket is non-empty, so a genuinely
        # undated mention doesn't silently make this look like a topic with no history at all.
        if not history and not unknown:
            continue
        if history:
            latest = max(history, key=lambda a: a["time"])
            latest_label, latest_date = latest["label"], latest["time"]
            weekly_rate = _windowed_average_rate(
                [a["time"] for a in history], CATCH_UP_WINDOW_DAYS, effective_recent_date
            )
            expected_count = max(1, round(weekly_rate * effective_gap_days / CATCH_UP_WINDOW_DAYS))
        else:
            # No dated sample to compute a weekly rate from at all -- falls back to the same
            # "ask about it at least once" default every topic used before saturation targets
            # existed, using an undated mention's own label as the concrete memory prompt and
            # `effective_recent_date` itself as a stand-in "latest_date" (unknown precisely when,
            # but recent enough to be worth asking about) purely so this topic still sorts
            # sensibly alongside dated ones below.
            latest_label, latest_date = unknown[0]["label"], effective_recent_date
            weekly_rate = 0.0
            expected_count = 1
        topics.append({
            "activity_type": activity_type,
            "history_count": len(history),
            "latest_label": latest_label,
            "latest_date": latest_date,
            "weekly_rate": weekly_rate,
            "expected_count": expected_count,
            "initial_reported_count": len(gap),
        })
    topics.sort(key=lambda t: t["latest_date"], reverse=True)
    return topics


def _catch_up_system_prompt(human: str) -> str:
    """System prompt for every catch-up QUESTION this module asks -- same framing as
    chat_sessions.default_system_prompt(), restrained to short, non-advice-giving replies, since
    these are all questions, not coaching. See _wrap_up_system_prompt() for the one moment this
    module's own reply ISN'T necessarily a question (SaturationTracker.wrap_up_message())."""
    return (
        f"You are a lifestyle coach talking with {human}, a person with Type 2 diabetes, at the "
        "start of a new chat conversation about diet, exercise, sleep, stress and daily routines "
        "that affect their blood sugar management. Keep your reply short (1-3 sentences) and "
        "warm. Only ever ask a question -- do not give advice, recommendations, or suggestions."
    )


def _wrap_up_system_prompt(human: str) -> str:
    """System prompt for SaturationTracker.wrap_up_message() -- the one moment this module's own
    reply is allowed to be something other than a question: once every catch-up topic is
    saturated (or capped out), the LLM decides for itself whether to continue the conversation
    with one more short, natural question, or to wrap up warmly and say goodbye instead.

    Explicitly told to weigh the human's OWN last message heavily, not just the "everything's
    covered" framing in the user prompt (see wrap_up_message()) -- a real session once had the
    human type "What's next?" (an unambiguous request to keep going) and got a goodbye anyway,
    because a plain "decide for yourself" instruction let the LLM lean entirely on "we've covered
    everything" and never actually weigh what the human had just said against it."""
    return (
        f"You are a lifestyle coach talking with {human}, a person with Type 2 diabetes. Keep "
        "your reply short (1-2 sentences). The catch-up topics for this period are covered, but "
        "look at the human's OWN last message before deciding what to do: if it asks a question, "
        "asks what's next, or otherwise shows they want to keep talking, treat that as more "
        "important than \"everything's covered\" and continue the conversation naturally instead "
        "of ending it. Only wrap up warmly and say goodbye if the human's own tone suggests "
        "they're done, or they say goodbye themselves. Do not give advice, recommendations, or "
        "suggestions."
    )


def _recap_system_prompt(human: str) -> str:
    """System prompt for SaturationTracker.recap_message() -- like _wrap_up_system_prompt(), a
    moment this module's own reply is allowed to be a plain statement rather than a question: the
    human explicitly asked for a recap, so answering with one is exactly right, not "only ever
    ask a question" (_catch_up_system_prompt()'s own restriction, for every OTHER catch-up
    reply)."""
    return (
        f"You are a lifestyle coach talking with {human}, a person with Type 2 diabetes. They "
        "just asked you to recap what's been discussed. Give a short (2-3 sentence), warm, plain "
        "summary of what's actually been covered. Do not give advice, recommendations, or "
        "suggestions, and do not ask a new question unless they specifically asked you to."
    )


def _is_recap_request(utterance: str, agent_fn=None, model: str = DEFAULT_MODEL) -> bool:
    """True if `utterance` (the human's own last message) is asking for a recap of what's been
    discussed -- "tell me what we discussed previously", "what did we talk about?", "can you
    remind me what we've covered?", or any other paraphrase of that same request -- as opposed to
    reporting something new, answering a question, or anything else.

    A single, cheap LLM classification call, mirroring
    chat_sessions.KgChatSession._classify_confirmation_reply()'s own "small dedicated classifier"
    pattern -- matching a fixed set of literal phrasings would miss most real paraphrases, which
    is exactly the case this function exists to handle (see wrap_agent_fn_with_saturation_loop(),
    the only caller)."""
    if not utterance or not utterance.strip():
        return False
    messages = [
        {"role": "system", "content": (
            "You are a classifier. Decide whether the person's message is asking for a recap or "
            "summary of what's been discussed in this conversation (e.g. \"tell me what we "
            "discussed previously\", \"what did we talk about?\", \"can you remind me what we've "
            "covered?\", or any other paraphrase of that same request) -- as opposed to reporting "
            "something new, answering a question, or anything else. Respond with exactly one "
            "word: YES or NO."
        )},
        {"role": "user", "content": utterance},
    ]
    reply_fn = agent_fn or _default_reply_fn(model)
    raw = (_call_openai("checking whether you're asking for a recap", reply_fn, messages) or "")
    return raw.strip().upper().startswith("YES")


def _format_gap_description(current_date: datetime, recent_date: datetime) -> str:
    """A human-ish phrase for how long it's been since `recent_date` -- "yesterday", "3 days
    ago, on Tuesday", etc. -- for the LLM prompts below."""
    days = (current_date.date() - recent_date.date()).days
    if days <= 0:
        return "earlier today"
    if days == 1:
        return "yesterday"
    return f"{days} days ago, on {recent_date.strftime('%A, %B %d')}"


def _default_reply_fn(model: str):
    return lambda messages: _openai_client().chat.completions.create(
        model=model, messages=messages
    ).choices[0].message.content


def _recent_context(messages: Optional[List[Dict]]) -> List[Dict]:
    """The last RECENT_CONTEXT_MESSAGES entries of the REAL running conversation (see that
    constant's own docstring for why this exists at all), with any "system"-role entries dropped
    -- this module supplies its own system prompt ahead of these, and a chat's very first message
    is always ChatSession's own system prompt, which would otherwise land here too on a short
    conversation. Returns [] for `messages=None` (the default for every direct/test call that
    doesn't have a real conversation to draw from -- see next_question()/wrap_up_message())."""
    if not messages:
        return []
    return [m for m in messages[-RECENT_CONTEXT_MESSAGES:] if m.get("role") != "system"]


class SaturationTracker:
    """Drives the "keep asking about gap-period topics until we have enough" loop (see this
    module's own docstring). Holds each catch-up topic's saturation TARGET
    (find_catch_up_topics()'s own "expected_count", derived from how often this human
    historically reported this topic in gap-length periods before now) alongside how many of
    that topic's activities have actually been reported so far THIS session (self.reported,
    seeded from "initial_reported_count" and updated live via record_new_activity() -- see
    chat_sessions.KgChatSession's own `on_new_subject` hook, the only integration point this
    needs with chat_sessions.py beyond the plain agent_fn wrapping).

    next_question() asks about whichever still-unsaturated topic has the BIGGEST shortfall
    (expected_count - reported), tie-broken by most-recently-discussed -- phrasing a follow-up
    ("anything else...") differently once a topic's already been asked about before this session
    (self.asked). Gives up on a topic once it's been asked MAX_ASKS_PER_TOPIC times regardless of
    whether its target was ever reached -- the same backstop philosophy as
    chat_sessions.KgIntentChatSession's own MAX_INTENT_GAP_ATTEMPTS: a human who simply has
    nothing more to say about a topic shouldn't be asked about it forever. Since every topic is
    capped this way, the WHOLE loop is bounded too (at most
    len(topics) * MAX_ASKS_PER_TOPIC catch-up questions, even in the worst case).

    is_saturated() is True once every target topic is either at/above its own expected_count or
    has hit that per-topic ask cap -- "enough knowledge for the gap period" (this module's own
    stated goal), not "literally every topic's exact target hit no matter what." Once that first
    happens, wrap_up_message() (see wrap_agent_fn_with_saturation_loop()) hands off to the LLM's
    own judgement -- continue naturally if there's an obvious thread left, or wrap up and say
    goodbye -- exactly once (self.wrapped_up), rather than this module silently going quiet or
    the conversation just stopping with no closing turn at all.
    """

    MAX_ASKS_PER_TOPIC = 3

    def __init__(self, topics: List[Dict], human: str, model: str = DEFAULT_MODEL):
        self.targets: Dict[str, Dict] = {t["activity_type"]: t for t in topics}
        self.reported: Dict[str, int] = {
            t: target["initial_reported_count"] for t, target in self.targets.items()
        }
        self.asked: Dict[str, int] = {t: 0 for t in self.targets}
        self.human = human
        self.model = model
        # [{"activity_type", "attempt"}, ...] -- one entry per next_question() call, for
        # inspection after the chat (chat_sessions.py's own turn_log has no notion of this
        # module's questions -- they're plain "default" reply_sources turns from its own point of
        # view, indistinguishable from a generic LLM reply without this).
        self.asked_log: List[Dict] = []
        # Set by wrap_up_message() the first time it runs -- see its own docstring and
        # wrap_agent_fn_with_saturation_loop() for why this must only ever fire once.
        self.wrapped_up = False

    def record_new_activity(self, subject_uri: str, activity_type: str) -> None:
        """Call whenever a NEW activity/condition of `activity_type` is pushed to the KG during
        this live session -- matches chat_sessions.KgChatSession's own `on_new_subject` hook
        signature exactly (`callable(subject_uri, activity_type)`), so this can be passed
        straight through as `on_new_subject=tracker.record_new_activity`; `subject_uri` itself
        isn't used for anything here (this tracker only ever counts BY type). A no-op for any
        type this tracker isn't targeting (not one of find_catch_up_topics()'s own topics)."""
        if activity_type in self.reported:
            self.reported[activity_type] += 1

    def _remaining(self) -> List[str]:
        """Topics still worth asking about: short of their own target AND not yet at the
        per-topic ask cap."""
        return [
            t for t, target in self.targets.items()
            if self.reported[t] < target["expected_count"] and self.asked[t] < self.MAX_ASKS_PER_TOPIC
        ]

    def is_saturated(self) -> bool:
        """True once there's nothing left worth asking about -- see _remaining()."""
        return not self._remaining()

    def _mark_asked(self, activity_type: str) -> None:
        self.asked[activity_type] += 1
        self.asked_log.append({"activity_type": activity_type, "attempt": self.asked[activity_type]})

    def _select_topic(self) -> Optional[str]:
        """Whichever still-unsaturated topic (see _remaining()) has the biggest shortfall
        (expected_count - reported so far), tie-broken by most-recently-discussed -- or None once
        every topic is saturated or capped out. The one selection rule opening_question() and
        next_question() both use to pick which SINGLE topic to actually ask about next."""
        remaining = self._remaining()
        if not remaining:
            return None
        remaining.sort(key=lambda t: (
            -(self.targets[t]["expected_count"] - self.reported[t]),
            -self.targets[t]["latest_date"].timestamp(),
        ))
        return remaining[0]

    def opening_question(self, current_date: datetime, recent_date: datetime,
                          agent_fn=None) -> str:
        """The very first agent turn, in two parts, in this order -- see this module's own
        docstring: (1) how long it's been since the last conversation, followed by a SUMMARY of
        every topic (`self.targets`, this human's own full catch-up list, not just a couple of
        "lead" ones) talked about back then, and only THEN (2) a single, specific question about
        whichever ONE topic most needs catching up on (`_select_topic()` -- the exact same
        selection `next_question()` uses), rather than a bare, generic "what's new?". Marks that
        one topic as asked (`_mark_asked()`) so `next_question()` doesn't immediately ask about it
        again right after the opening turn already did.

        If there's nothing left to ask about at all (e.g. every topic is already covered by data
        pushed through some other channel before this chat even starts), the message still names
        every topic recapped from last time, just without the trailing specific question.
        """
        all_topics_phrase = ", ".join(t.replace("_", " ") for t in self.targets)
        activity_type = self._select_topic()
        specific_label = None
        if activity_type is not None:
            self._mark_asked(activity_type)
            specific_label = activity_type.replace("_", " ")
        user_prompt = (
            f"Our last conversation was {_format_gap_description(current_date, recent_date)}. "
            + (f"Back then we talked about: {all_topics_phrase}. " if all_topics_phrase else "")
            + "Write the opening message of today's chat, in this order: (1) greet them and "
            "mention how long it's been since you last talked, (2) in one short sentence, "
            "summarize the topics you talked about last time"
            + (f" ({all_topics_phrase})" if all_topics_phrase else "")
            + (
                f", (3) THEN, as a separate follow-up, ask a specific question inviting them to "
                f"share what's happened with their {specific_label} since then"
                if specific_label else ""
            )
            + ". Keep it natural and short (2-3 sentences)."
        )
        messages = [
            {"role": "system", "content": _catch_up_system_prompt(self.human)},
            {"role": "user", "content": user_prompt},
        ]
        reply_fn = agent_fn or _default_reply_fn(self.model)
        return _call_openai("generating the opening catch-up message", reply_fn, messages)

    def next_question(self, agent_fn=None, messages: Optional[List[Dict]] = None) -> Optional[str]:
        """Ask about whichever still-unsaturated topic (see _select_topic()) has the biggest
        shortfall (expected_count - reported so far), tie-broken by most-recently-discussed --
        or None once every topic is saturated or capped out. Phrases a FOLLOW-UP ("anything
        else...") when this topic's already been asked about before this session, instead of
        repeating the exact same question.

        `messages` -- the REAL running conversation (see RECENT_CONTEXT_MESSAGES) -- is included
        as context ahead of the instruction when given, so the question reacts to what the human
        actually just said instead of being generated in isolation from it."""
        activity_type = self._select_topic()
        if activity_type is None:
            return None
        target = self.targets[activity_type]
        is_followup = self.asked[activity_type] > 0
        self._mark_asked(activity_type)
        label = activity_type.replace("_", " ")
        if is_followup:
            user_prompt = (
                f"Earlier {self.human} mentioned {label} (most recently: "
                f"\"{target['latest_label']}\"), but that's still fewer than what's typical for "
                f"them over a period like this. Ask a short, natural follow-up inviting them to "
                f"share ANOTHER {label}-related thing from the same period, without repeating the "
                f"exact same question as before."
            )
        else:
            user_prompt = (
                f"Historically {self.human} has reported about {label} roughly "
                f"{target['expected_count']} time(s) over a period this length (most recently: "
                f"\"{target['latest_label']}\"). Ask them a short, natural question inviting them "
                f"to share what's happened with their {label} during this period."
            )
        prompt = (
            [{"role": "system", "content": _catch_up_system_prompt(self.human)}]
            + _recent_context(messages)
            + [{"role": "user", "content": user_prompt}]
        )
        reply_fn = agent_fn or _default_reply_fn(self.model)
        return _call_openai(f"asking a catch-up question about {label}", reply_fn, prompt)

    def wrap_up_message(self, agent_fn=None, messages: Optional[List[Dict]] = None) -> str:
        """The ONE message sent the moment every topic first becomes saturated (or capped out) --
        see wrap_agent_fn_with_saturation_loop(). Rather than this module going silent and the
        conversation just ending (the previous behaviour: is_saturated() becoming True meant every
        subsequent default reply skipped this module entirely, with nothing marking the moment),
        this hands off to the LLM's own judgement -- names what was actually covered this session,
        as concrete context -- and lets IT decide whether there's an obvious thread left to
        continue with, or whether to wrap up and say goodbye instead. Sets self.wrapped_up so this
        never fires a second time (see wrap_agent_fn_with_saturation_loop(), the only caller).

        `messages` (see RECENT_CONTEXT_MESSAGES) matters MOST here of anywhere in this module: a
        real session once had the human type "What's next?" -- an explicit request to keep going
        -- immediately before this ran, and it said goodbye anyway, because without the real
        conversation as context it had no way to know that message even existed. Passing it
        through is what lets the LLM actually notice a signal like that instead of deciding purely
        from the "covered: ..." summary below."""
        self.wrapped_up = True
        covered = [t.replace("_", " ") for t, target in self.targets.items() if self.reported[t] > 0]
        covered_phrase = ", ".join(covered) if covered else "nothing new for this period"
        user_prompt = (
            f"You've now caught up on everything worth covering for this period (covered: "
            f"{covered_phrase}). Decide for yourself: if there's an obvious natural thread left to "
            "follow up on, continue with a short question about it; otherwise wrap the "
            "conversation up warmly and say goodbye."
        )
        prompt = (
            [{"role": "system", "content": _wrap_up_system_prompt(self.human)}]
            + _recent_context(messages)
            + [{"role": "user", "content": user_prompt}]
        )
        reply_fn = agent_fn or _default_reply_fn(self.model)
        return _call_openai("wrapping up the catch-up conversation", reply_fn, prompt)

    def recap_message(self, messages: Optional[List[Dict]] = None, agent_fn=None) -> str:
        """Answer an explicit "tell me what we discussed [previously]" (or any paraphrase -- see
        _is_recap_request()) with a natural-language summary of the catch-up topics: which ones
        were identified as worth covering for this period (self.targets, set once at
        construction from find_catch_up_topics()'s own output) versus which of those have
        actually been covered so far THIS session (self.reported) and which are still open
        (short of target and not yet capped out -- the same test _remaining() uses).

        A pure read -- unlike next_question()/wrap_up_message(), this never changes self.asked/
        self.wrapped_up/anything else, since answering "what have we covered" isn't itself
        another catch-up question and shouldn't count as one, or consume one of a topic's own
        MAX_ASKS_PER_TOPIC attempts.
        """
        covered = [t.replace("_", " ") for t, target in self.targets.items() if self.reported[t] > 0]
        still_open = [
            t.replace("_", " ") for t, target in self.targets.items()
            if self.reported[t] < target["expected_count"] and self.asked[t] < self.MAX_ASKS_PER_TOPIC
        ]
        covered_phrase = ", ".join(covered) if covered else "nothing yet this time"
        open_phrase = ", ".join(still_open) if still_open else "nothing else"
        user_prompt = (
            f"The person just asked for a recap of what's been discussed. So far this "
            f"conversation you've covered: {covered_phrase}. Still worth asking about before "
            f"you're caught up: {open_phrase}. Give them a short, natural summary of what's been "
            f"covered so far, in plain language (no bullet points, no jargon like activity "
            f"types)."
        )
        prompt = (
            [{"role": "system", "content": _recap_system_prompt(self.human)}]
            + _recent_context(messages)
            + [{"role": "user", "content": user_prompt}]
        )
        reply_fn = agent_fn or _default_reply_fn(self.model)
        return _call_openai("summarizing what's been discussed so far", reply_fn, prompt)


def wrap_agent_fn_with_saturation_loop(agent_fn, tracker: SaturationTracker):
    """Wrap `agent_fn` (e.g. chat_sessions.openai_agent()) so that every call to it -- which
    chat_sessions.KgChatSession.say() only ever makes once it's found no per-turn intent gap left
    to ask about for whatever the human just said, see its own docstring's "default"
    reply_sources tag -- asks about the next still-unsaturated gap-period topic
    (tracker.next_question()) instead of the plain default reply, for as long as
    `tracker.is_saturated()` is False. The FIRST time every topic is saturated (or capped out),
    `tracker.wrap_up_message()` runs once instead of silently falling through -- see its own
    docstring for why. After that (`tracker.wrapped_up` is True), every call goes straight through
    to `agent_fn` unchanged, exactly as if this wrapper wasn't there -- so the conversation
    continues (or ends, if the human says goodbye back) as an ordinary chat from that point on.

    `messages` -- the REAL running conversation, which `say()` always passes to `agent_fn` -- is
    forwarded to both `next_question()` and `wrap_up_message()` (see RECENT_CONTEXT_MESSAGES) so
    neither is generated blind to whatever the human just actually said.

    Before any of that, EVERY call first checks whether the human's own last message is asking
    for a recap ("tell me what we discussed previously", or any paraphrase -- see
    `_is_recap_request()`); if so, `tracker.recap_message()` answers it directly, taking priority
    over both asking a fresh question and wrapping up, and without touching any tracker state
    (see that method's own docstring for why) -- asking to be reminded what's been covered so far
    isn't itself another catch-up question.
    """
    def _wrapped(messages):
        last_utterance = messages[-1]["content"] if messages else None
        if last_utterance and _is_recap_request(last_utterance):
            return tracker.recap_message(messages=messages)
        if not tracker.is_saturated():
            question = tracker.next_question(messages=messages)
            if question is not None:
                return question
        elif not tracker.wrapped_up:
            return tracker.wrap_up_message(messages=messages)
        return agent_fn(messages)
    return _wrapped


# --------------------------------------------------------------------------- #
# Post-chat summary: the "intent log"
# --------------------------------------------------------------------------- #

def _serialize_topic(topic: Dict) -> Dict:
    """One find_catch_up_topics()-shaped topic dict, made JSON-safe (`latest_date`'s datetime ->
    ISO 8601 string) -- used for both "topics_at_start" and (via SaturationTracker.targets)
    "topics_covered" in save_intent_log()."""
    serialized = dict(topic)
    if isinstance(serialized.get("latest_date"), datetime):
        serialized["latest_date"] = serialized["latest_date"].isoformat()
    return serialized


def _topics_covered_from_tracker(tracker: "SaturationTracker") -> List[Dict]:
    """One entry per SaturationTracker topic, summarizing how the LIVE chat actually went for it:
    how many of it were reported, how many times it was asked about, and whether it ended up
    "target_met" (expected_count actually reached) or just "capped_out" (MAX_ASKS_PER_TOPIC hit
    without ever reaching it) -- either one satisfies is_saturated() for that topic, but they mean
    different things (see SaturationTracker's own docstring)."""
    return [
        {
            "activity_type": activity_type,
            "expected_count": target["expected_count"],
            "reported_count": tracker.reported[activity_type],
            "asked_count": tracker.asked[activity_type],
            "target_met": tracker.reported[activity_type] >= target["expected_count"],
            "capped_out": tracker.asked[activity_type] >= tracker.MAX_ASKS_PER_TOPIC,
        }
        for activity_type, target in tracker.targets.items()
    ]


def _intents_covered_from_turn_log(turn_log: List[Dict]) -> List[Dict]:
    """Every intent_gap_finder.py intent actually consulted during the chat's own per-turn flow,
    summarized from chat_sessions.KgIntentChatSession.turn_log's own "gap_queries" entries (each
    shaped {"subject", "activity_type", "intent_source", "after_dedup", "gave_up"} -- see
    KgIntentChatSession._fetch_gap_queue()). One entry per DISTINCT "intent_source" file matched
    at least once, listing which activity type(s) it covered THIS chat, how many times it was
    queried, and whether MAX_INTENT_GAP_ATTEMPTS ever made it give up on one of its own
    requirements (see chat_sessions.py's own "Why the same question doesn't repeat"). Independent
    of the catch-up topics/SaturationTracker above -- an intent fires for ANY matching activity
    the human mentions, whether or not the saturation loop is what prompted it; a plain
    KgChatSession's turn_log (no "intent_source"/"gave_up" keys at all) yields an empty list.
    """
    covered: Dict[str, Dict] = {}
    for entry in turn_log:
        for query in entry.get("gap_queries") or []:
            source = query.get("intent_source")
            if not source:
                continue
            info = covered.setdefault(source, {
                "intent_source": source, "activity_types": set(), "queries": 0, "gave_up": False,
            })
            if query.get("activity_type"):
                info["activity_types"].add(query["activity_type"])
            info["queries"] += 1
            if query.get("gave_up"):
                info["gave_up"] = True
    return [
        {**info, "activity_types": sorted(info["activity_types"])}
        for info in sorted(covered.values(), key=lambda i: i["intent_source"])
    ]


def save_intent_log(kg_session, tracker: "SaturationTracker", catch_up_topics: List[Dict],
                     current_date: datetime, recent_date: datetime,
                     log_dir: str = "intents_log") -> Path:
    """Write one JSON file to `log_dir` (created if needed -- default "intents_log", resolved
    relative to the caller's own CWD, same convention as connect_brain()'s "kg_logs" default;
    typically notebooks/intents_log/, since a notebook's CWD is its own directory) summarizing
    this whole catch-up + intent-driven chat session, once it's over:

    - "gap" -- the TRUE gap since the last conversation (`recent_date`/`current_date`/`gap_days`)
      alongside the (possibly capped -- see MAX_SATURATION_GAP_DAYS) period this session actually
      tried to saturate (`effective_recent_date`/`effective_gap_days`).
    - "topics_at_start" -- find_catch_up_topics()'s own raw output, unchanged: every topic
      identified as worth catching up on BEFORE the chat began, with its saturation target
      (expected_count/weekly_rate/initial_reported_count -- see that function's own docstring).
    - "topics_covered" -- how each of those topics actually fared LIVE
      (_topics_covered_from_tracker()), `tracker.asked_log` (one entry per catch-up question
      actually asked, in order, from SaturationTracker's own bookkeeping), whether the whole
      tracker ended up saturated, and whether the chat ever reached wrap_up_message() (see
      SaturationTracker.wrapped_up).
    - "intents_covered" -- every intent_gap_finder.py intent actually consulted during the chat
      (_intents_covered_from_turn_log()) -- separate from "topics_covered" above, since an intent
      can fire for an activity the human brought up unprompted, not just one the saturation loop
      itself asked about.

    Named "chat<chat>_intents_<stamp>.json" under `log_dir`, timestamped the same way
    kg_chat_gui.save_session()'s own turns/stats/gaplog files are, so repeated runs never clobber
    each other. Returns the path written.
    """
    save_dir = Path(log_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    effective_recent_date = max(recent_date, current_date - timedelta(days=MAX_SATURATION_GAP_DAYS))
    log = {
        "chat": getattr(kg_session, "chat", None),
        "human": getattr(kg_session, "human", None),
        "gap": {
            "current_date": current_date.isoformat(),
            "last_conversation_date": recent_date.isoformat(),
            "gap_days": (current_date.date() - recent_date.date()).days,
            "effective_recent_date": effective_recent_date.isoformat(),
            "effective_gap_days": (current_date.date() - effective_recent_date.date()).days,
        },
        "topics_at_start": [_serialize_topic(t) for t in catch_up_topics],
        "topics_covered": {
            "per_topic": _topics_covered_from_tracker(tracker),
            "asked_log": tracker.asked_log,
            "saturated": tracker.is_saturated(),
            "wrapped_up": tracker.wrapped_up,
        },
        "intents_covered": _intents_covered_from_turn_log(getattr(kg_session, "turn_log", None) or []),
    }

    base = re.sub(r"[^A-Za-z0-9_.-]", "_", f"chat{log['chat']}")
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = save_dir / f"{base}_intents_{stamp}.json"
    with open(path, "w") as f:
        json.dump(log, f, indent=2)
    print(f"Wrote intent log to {path}")
    return path
