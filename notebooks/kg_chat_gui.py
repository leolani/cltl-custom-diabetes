"""
kg_chat_gui.py
===============
A single-window Tkinter chat UI for driving a `ChatSession` / `KgChatSession`
(chat_sessions.py), as an alternative to `ChatSession.run_interactive()`'s
terminal `input()` loop.

- The whole conversation (both the human's and the agent's turns) is shown scrolling in one
  window, instead of being interleaved into the notebook's cell output.
- The human types their line into an entry box built into the bottom of that *same* window --
  there's no separate popup/dialog for input, and no terminal `input()` prompt.
- Every agent turn is tagged **[KG]** or **[LLM]** so it's visible at a glance whether that
  reply came from a knowledge-graph gap question or the default LLM reply -- read from
  `session.reply_sources` (one "gap"/"default"/"timeout" entry per agent turn, appended by
  `KgChatSession.say()`; see chat_sessions.py). A plain `ChatSession` has no `reply_sources`
  attribute at all, so every one of its agent turns is just tagged **[LLM]** (its only kind of
  reply) rather than left unlabeled. A "timeout" turn is also shown as **[LLM]** (it isn't a KG
  gap question) -- its actual message text is what tells you an OpenAI call timed out and names
  which step; see chat_sessions.ChatTimeoutError.
- For a `KgChatSession` (anything with a `gap_threshold` attribute), a "Gap sensitivity" slider
  lets `gap_threshold` be adjusted live, mid-conversation, instead of only at construction time --
  see `_on_gap_threshold_change()`. Not shown for a plain `ChatSession`, which has no such thing.
- Quitting -- the **Quit** button, a quit word, or the window manager's close button -- writes the
  conversation and a statistics summary out as timestamped JSON files (`save_session()` /
  `session_statistics()`) -- plus, for a `KgChatSession`, its turn-by-turn gap log
  (`chat_sessions.KgChatSession.turn_log`) as a third file -- and prints their paths, before
  tearing the window down.
- When `session.kg_address` points at a GraphDB repository (`.../repositories/<id>`), the window
  splits into two panes: the chat on the left (everything above), and a **graph panel** on the
  right drawing the activity the conversation is currently about as an actual node-link diagram
  -- the activity as one centered node, every (predicate, object) triple pushed for it as a
  labeled edge fanning out to its own node, except `rdfs:label` triples (the label value is
  already what the center node's own text shows, so repeating it as a separate
  "label -> <that same text>" node/edge would just be clutter), `gaf:denotedIn`/`denotedBy`
  triples (`GAF_PROVENANCE_PREDICATES`: pure extraction provenance -- which utterance span a
  fact came from -- not semantic content, and typically several per subject), and
  `sem:eventProperty`/`eps:contextProperty` triples (`GENERIC_ANCESTOR_PREDICATES`: generic
  ancestor properties every real role predicate is inferred under on a repository with
  RDFS/OWL reasoning enabled, which would otherwise duplicate every single edge already drawn
  under its own real predicate) -- none of these three is ever drawn as its own node -- rendered
  directly on a `tk.Canvas`, entirely in-window, no browser involved (see the graph-panel section
  below and `_render_graph()`). Every
  node is filled by its RDF namespace (`_namespace_of()`/`_node_fill_color()`): red for `n2mu`,
  blue for `gaf`, green for `grasp` (this project's own ontology namespaces -- see
  `NAMESPACE_COLORS`), a plain literal (not a URI at all) gray, and any other namespace one of
  `OTHER_NAMESPACE_PALETTE`'s colors, assigned the first time it's seen and kept for the life of
  the window -- a small legend under the activity dropdown shows what's what. A dropdown lists
  every activity mentioned so far (most recent first) so you can switch which one's drawn; an
  **"Open in GraphDB ↗"** button is still there for the *real*, fully interactive D3 view (drag
  nodes, expand further, ...) in the system's default browser when that's wanted -- same URL
  shape as GraphDB's own UI:
  `<base>/graphs-visualizations?uri=<activity>&role=context&repositoryId=<repo>`. Not built at
  all for a plain `ChatSession`, or a `kg_address` that isn't a GraphDB repository URL
  (`graphdb_base_and_repository()` returns `(None, None)` and the panel is simply skipped).
- Two independent font-size sliders, both starting at `DEFAULT_FONT_SIZE` (16pt): **"Text size"**
  (always shown) scales the transcript/entry/buttons/status line together -- built on named
  `tkinter.font.Font` objects (`self.chat_font`/`chat_font_bold`/`chat_font_italic`) that every
  one of those widgets/tags references directly, so reconfiguring just their `size` live-resizes
  everything at once, already-inserted transcript text included, no re-insertion needed. The
  graph panel's own **"Font size"** slider (only when the panel exists) instead re-runs
  `_render_graph()` on its already-cached data at the new size -- canvas text items don't
  auto-resize the way Text-widget-tagged text does -- so this, like a resize, costs no network
  round-trip; node/center circle radii scale roughly *proportionally* with it (not by a small
  fixed add-on), so a node's label keeps roughly the same amount of room -- and wraps onto
  roughly the same number of lines -- at any size on the slider, instead of the circle staying
  about the same size while the text inside it keeps growing.

Usage (from a notebook, in place of `kg_session.run_interactive()`):
    from chat_sessions import KgChatSession
    from kg_chat_gui import run_gui

    kg_session = KgChatSession(chat=..., human=..., kg_address=..., ...)
    kg_turns = run_gui(kg_session)   # opens the window, blocks until it's closed

Talking to the graph/LLM happens on a background thread per turn (SPARQL pushes, gap lookups
and OpenAI calls can each take a few seconds -- occasionally longer, if a call hangs outright),
so the window stays responsive; only the resulting Tk widget updates are marshalled back onto
the main thread via `root.after()`, since Tkinter itself isn't thread-safe. The entry box is
never disabled while a reply is in flight, and a quit word (see `quit_words`, e.g. "bye") is
checked before anything else in `_on_send()` -- so the window always closes right away when you
say bye, even if the previous reply is still pending or never comes back. The "Quit" button next
to Send does the same thing (calls the same _on_close()) and, unlike Send, is never disabled
either -- a one-click way out that needs no typing. `_on_close()` calls `root.quit()` before
`root.destroy()`: destroy() alone tears down the widgets but doesn't reliably end a mainloop()
that IPython/Jupyter is driving or nesting, which left the window on screen and the notebook
cell hanging even after Quit was pressed.

Needs Tkinter (part of the Python standard library on most installs; on some Linux/pyenv
builds it needs a separate `python3-tk` / `tk-dev` package).
"""

import json
import math
import re
import threading
import tkinter as tk
import urllib.parse
import urllib.request
import webbrowser
from collections import Counter
from datetime import datetime
from pathlib import Path
from tkinter import font as tkfont, scrolledtext, ttk

DEFAULT_QUIT_WORDS = ("quit", "exit", "goodbye", "bye", "stop")

# Where save_session() writes when run_gui()/ChatWindow aren't given an explicit save_dir --
# notebooks/chat_logs (this file's own directory), resolved from this file's own location so
# it's the same directory no matter what a notebook's CWD happens to be. Pass save_dir=None to
# either one to turn saving off entirely.
DEFAULT_SAVE_DIR = Path(__file__).resolve().parent / "chat_logs"

# Font size (points) both the transcript/entry/buttons and the graph diagram start at -- each
# adjustable afterwards via its own slider (see the "Text size"/"Font size" controls in
# ChatWindow.__init__()/_build_graph_panel()), independently of the other.
DEFAULT_FONT_SIZE = 16
CHAT_FONT_RANGE = (10, 28)   # (min, max) for the conversation's "Text size" slider
GRAPH_FONT_RANGE = (8, 24)  # (min, max) for the graph panel's "Font size" slider


# --------------------------------------------------------------------------- #
# Graph panel: querying + linking to GraphDB, dependency-free (urllib only --
# no rdflib/SPARQLWrapper, so this stays usable without any of KgChatSession's
# heavier dependencies; see the module docstring).
# --------------------------------------------------------------------------- #

# Matches a GraphDB repository's SPARQL endpoint, e.g.
# "http://localhost:7200/repositories/event_sandbox" -> ("http://localhost:7200", "event_sandbox").
GRAPHDB_REPOSITORY_URL_PATTERN = re.compile(r"^(https?://[^/]+)/repositories/([^/?#]+)/?$")

LABEL_PREDICATE = "http://www.w3.org/2000/01/rdf-schema#label"
# gaf:denotedIn/denotedBy link a subject to the raw utterance-span(s) it was extracted from --
# cltl.brain's own provenance/bookkeeping, not semantic content (the same predicates
# kg_gap_finder.semantic_role_predicates() already excludes from gap-finding for the same
# reason). Excluded from the graph diagram too, alongside LABEL_PREDICATE -- see _render_graph().
GAF_NAMESPACE = "http://groundedannotationframework.org/gaf#"
GAF_PROVENANCE_PREDICATES = {GAF_NAMESPACE + "denotedIn", GAF_NAMESPACE + "denotedBy"}
# sem:eventProperty and eps:contextProperty are the SEM/episodic-awareness ontologies' own generic
# ANCESTOR properties: every real role predicate this project pushes (n2mu:agent, n2mu:location,
# ...) is rdfs:subPropertyOf sem:hasActor/hasPlace/hasTime etc. (see n2mu_sem_roles.py), which are
# themselves rdfs:subPropertyOf sem:eventProperty, and eps:contextProperty is declared
# owl:equivalentProperty to that same sem:eventProperty (see cltl.brain's own
# ontologies/integration.ttl). On a GraphDB repository with RDFS/OWL inference enabled, EVERY one
# of those real triples is therefore also materialized as an sem:eventProperty and an
# eps:contextProperty triple to the SAME object -- fetch_triples() has no way to tell an inferred
# triple from an asserted one, so without this exclusion the SAME (subject, object) pair draws as
# up to three near-identical edges/nodes (one real predicate, plus these two generic ones) instead
# of one. Excluded from the graph diagram alongside LABEL_PREDICATE/GAF_PROVENANCE_PREDICATES --
# see _render_graph().
GENERIC_ANCESTOR_PREDICATES = {
    "http://semanticweb.cs.vu.nl/2009/11/sem/eventProperty",
    "http://cltl.nl/episodicawareness/contextProperty",
}

# Groups of predicates that all express the SAME underlying role for a triple's OBJECT: the real
# n2mu: role predicate(s) this project's own extractor asserts, plus the one SEM-ontology
# superproperty n2mu_sem_roles.py's own ROLE_SUBPROPERTY_MAP declares each of them
# rdfs:subPropertyOf (duplicated here rather than imported -- see this module's own "plain
# urllib, no rdflib" design note above GRAPHDB_REPOSITORY_URL_PATTERN). Unlike
# GENERIC_ANCESTOR_PREDICATES's sem:eventProperty/eps:contextProperty (which carry no real
# information of their own and are always dropped outright), sem:hasActor/hasPlace/hasTime DO
# mean something -- so instead of excluding them, _merge_equivalent_edges() collapses each group
# down to just ONE edge per distinct object it shares with one of the group's own n2mu:
# predicates. Concretely: on a repository with RDFS/OWL inference enabled, a single
# `n2mu:agent -> "Jan"` triple is ALSO materialized as `sem:hasActor -> "Jan"` (agent_patient/
# participant/experiencer all the same way), and without merging, the diagram would draw two (or
# more) near-identical edges to "Jan" for what both really are the same one fact. Order within
# each tuple matters: the FIRST member actually present for a given object is what the merged
# edge is labeled with, so a real, specific n2mu: predicate is always shown in preference to its
# own generic sem: superproperty copy.
N2MU_NAMESPACE = "http://cltl.nl/leolani/n2mu/"
SEM_NAMESPACE = "http://semanticweb.cs.vu.nl/2009/11/sem/"
EDGE_EQUIVALENCE_GROUPS = (
    (N2MU_NAMESPACE + "agent", N2MU_NAMESPACE + "agent_patient", N2MU_NAMESPACE + "participant",
     N2MU_NAMESPACE + "experiencer", SEM_NAMESPACE + "hasActor"),
    (N2MU_NAMESPACE + "location", SEM_NAMESPACE + "hasPlace"),
    (N2MU_NAMESPACE + "time/dateTime", N2MU_NAMESPACE + "time/rangeTime",
     N2MU_NAMESPACE + "time/recurringTime", N2MU_NAMESPACE + "time/vagueTime",
     SEM_NAMESPACE + "hasTime"),
)

SPARQL_TIMEOUT = 8  # seconds -- fetch_triples() runs on a background thread, but shouldn't hang it forever.

# Node-link diagram colors (see ChatWindow._render_graph()) -- picked to read clearly on the
# canvas's plain white background; not tied to any dark/light theming (this is a native desktop
# window, not a themeable web page).
GRAPH_BG = "#ffffff"
GRAPH_NODE_OUTLINE = "#241f31"      # one uniform outline for every node, regardless of fill
GRAPH_NODE_TEXT = "#ffffff"         # white reads cleanly on every color in NAMESPACE_COLORS/
                                     # OTHER_NAMESPACE_PALETTE/LITERAL_NODE_FILL below -- see
                                     # ChatWindow._node_fill_color()
GRAPH_EDGE_COLOR = "#9aa5b1"
GRAPH_EDGE_LABEL_COLOR = "#3d3846"  # deliberately NOT green -- green is reserved for "grasp"
                                    # namespace nodes below, so an edge label can't be misread
                                    # as itself indicating a namespace
GRAPH_PLACEHOLDER_TEXT = "#9aa5b1"

# Node fill color by RDF namespace (see _namespace_of()/ChatWindow._node_fill_color()) -- the
# three namespaces this project's own ontology actually uses for instance data
# (cltl.brain.infrastructure.rdf_builder._define_namespaces()), explicitly colored as asked.
NAMESPACE_COLORS = {
    "n2mu": "#c01c28",   # red   -- http://cltl.nl/leolani/n2mu/...        (activities/roles)
    "gaf": "#1a5fb4",    # blue  -- http://groundedannotationframework.org/gaf#...  (mentions)
    "grasp": "#26a269",  # green -- http://groundedannotationframework.org/grasp...  (attribution)
}
# Any OTHER namespace (e.g. http://cltl.nl/leolani/world/, http://cltl.nl/leolani/friends/, a
# plain rdf:/rdfs:/owl: URI, ...) gets one of these instead, assigned the first time it's seen
# and cached per-window (ChatWindow._namespace_colors) so it stays the SAME color for the life
# of the window, not just within one render.
OTHER_NAMESPACE_PALETTE = [
    "#9c6f16", "#813d9c", "#c64600", "#0e7a6f", "#63452c", "#a2734c", "#3d3846", "#865e3c",
]
LITERAL_NODE_FILL = "#5e5c64"  # a plain literal object (no URI/namespace at all, e.g. "the park")


def _namespace_of(uri: str):
    """The RDF namespace `uri` belongs to, as a short key for _node_fill_color() --
    None for a plain literal (not a URI at all). Checked by substring against the three
    namespaces this project's own ontology uses for instance data
    (cltl.brain.infrastructure.rdf_builder._define_namespaces()) -- "grasp" also covers its
    grasp/factuality, grasp/sentiment, grasp/emotion, grasp/level sub-namespaces, all sharing
    that one substring. Anything else falls back to the URI's own prefix (everything up to and
    including its last '/' or '#') as a distinct "other" namespace key, so two different other
    namespaces still get two different (cached, stable) colors from OTHER_NAMESPACE_PALETTE."""
    if not uri or "://" not in uri:
        return None
    lower = uri.lower()
    if "/n2mu/" in lower or lower.rstrip("/").endswith("/n2mu"):
        return "n2mu"
    if "/gaf#" in lower or "/gaf/" in lower:
        return "gaf"
    if "/grasp" in lower:  # grasp#, grasp/factuality#, grasp/sentiment#, grasp/emotion#, ...
        return "grasp"
    return uri.rsplit("#", 1)[0] if "#" in uri else uri.rsplit("/", 1)[0]


def graphdb_base_and_repository(kg_address):
    """Split a repository SPARQL endpoint URL into (workbench_base_url, repository_id), e.g.
    "http://localhost:7200/repositories/event_sandbox" -> ("http://localhost:7200",
    "event_sandbox") -- the two pieces graphdb_visualization_url() needs to build a GraphDB
    Workbench link. Returns (None, None) if `kg_address` is falsy or doesn't look like a GraphDB
    repository endpoint (e.g. a different SPARQL 1.1 server); callers use that to skip building
    the graph panel/links instead of producing a broken URL."""
    if not kg_address:
        return None, None
    match = GRAPHDB_REPOSITORY_URL_PATTERN.match(kg_address.strip())
    if not match:
        return None, None
    return match.group(1), match.group(2)


def graphdb_visualization_url(kg_address, subject_uri, role="context"):
    """The GraphDB Workbench "Visual graph" URL centered on `subject_uri` -- the same shape
    GraphDB's own UI links to when you open a resource's graph view, e.g.
    "http://localhost:7200/graphs-visualizations?uri=<subject_uri>&role=context&repositoryId=<repo>".
    Returns None if `kg_address` doesn't look like a GraphDB repository endpoint (see
    graphdb_base_and_repository())."""
    base, repository_id = graphdb_base_and_repository(kg_address)
    if not base or not subject_uri:
        return None
    query = urllib.parse.urlencode({"uri": subject_uri, "role": role, "repositoryId": repository_id})
    return f"{base}/graphs-visualizations?{query}"


def fetch_triples(kg_address, subject_uri, timeout=SPARQL_TIMEOUT):
    """Every (predicate, object) triple with `subject_uri` as its subject -- one plain SPARQL
    SELECT over HTTP GET against `kg_address` (a GraphDB/RDF4J-style repository endpoint
    accepts SPARQL query results as JSON via a bare GET with an Accept header, no client library
    needed). Returns a sorted list of (predicate, object) tuples. Raises on a network or query
    error -- this talks to a live server that can be unreachable, slow, or return malformed
    results, so callers should catch broadly rather than assume it always succeeds."""
    query = f"SELECT ?p ?o WHERE {{ <{subject_uri}> ?p ?o }}"
    url = kg_address.rstrip("/") + "?" + urllib.parse.urlencode({"query": query})
    request = urllib.request.Request(url, headers={"Accept": "application/sparql-results+json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    rows = payload.get("results", {}).get("bindings", [])
    triples = [(row["p"]["value"], row["o"]["value"]) for row in rows]
    triples.sort()
    return triples


def _local_name(uri: str) -> str:
    """Human-ish tail of a URI (after the last '/' or '#'), for display when no rdfs:label is
    known yet. Mirrors prompts.response_processor.local_name() (deliberately duplicated, not
    imported -- kg_chat_gui.py otherwise has no dependency on src/cltl's sys.path setup; see the
    module docstring)."""
    if not uri:
        return uri
    return uri.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1] or uri


def _truncate(text: str, max_chars: int) -> str:
    """`text`, cut to at most `max_chars` characters with a trailing "…" if it was longer --
    used to keep a node-link diagram's labels from overrunning their node/canvas (see
    ChatWindow._render_graph()). The full value is always still visible via "Open in GraphDB ↗"
    or the plain triples this diagram is built from (fetch_triples())."""
    text = text if text is not None else ""
    return text if len(text) <= max_chars else text[: max_chars - 1].rstrip() + "…"


def _merge_equivalent_edges(triples):
    """Collapse every EDGE_EQUIVALENCE_GROUPS group down to one (predicate, object) edge per
    distinct object it shares with one of that group's own predicates -- see that constant's own
    docstring for why (an inferred sem:hasActor/hasPlace/hasTime triple duplicating a real
    n2mu:agent/location/time/* one to the same object). A triple whose predicate isn't in any
    group passes through unchanged. Preserves the ORIGINAL ordering: a merged edge is placed
    wherever its group+object pair was first seen among `triples`, so the diagram's edge order
    doesn't visibly reshuffle just because this ran.
    """
    group_by_predicate = {p: group for group in EDGE_EQUIVALENCE_GROUPS for p in group}
    clusters = {}  # (group, object) -> {"predicates": set(), "first_index": int}
    passthrough = []
    for index, (predicate, obj) in enumerate(triples):
        group = group_by_predicate.get(predicate)
        if group is None:
            passthrough.append((index, predicate, obj))
            continue
        cluster = clusters.setdefault((group, obj), {"predicates": set(), "first_index": index})
        cluster["predicates"].add(predicate)
    merged = [
        (cluster["first_index"], next(p for p in group if p in cluster["predicates"]), obj)
        for (group, obj), cluster in clusters.items()
    ]
    ordered = sorted(passthrough + merged, key=lambda item: item[0])
    return [(predicate, obj) for _, predicate, obj in ordered]


# --------------------------------------------------------------------------- #
# Saving the conversation + statistics on quit
# --------------------------------------------------------------------------- #

def session_statistics(session) -> dict:
    """Summarize a finished ChatSession/KgChatSession as a JSON-serializable dict.

    Every field beyond the turn counts is read defensively (getattr with a default), so this
    works for a plain `ChatSession` -- which has no reply_sources/annotations/kg_pushes/
    gap_threshold at all -- as well as a full `KgChatSession`; a key is simply absent when the
    session doesn't have the underlying attribute.

    Note this deliberately COUNTS `session.annotations` rather than including them: their
    "Output" holds raw SRLAnnotation Pydantic objects, which json.dump() can't serialize.
    """
    turns = list(getattr(session, "turns", None) or [])
    human = getattr(session, "human", None)
    human_turns = sum(1 for turn in turns if turn.get("speaker") == human)

    stats = {
        "chat": getattr(session, "chat", None),
        "human": human,
        "date": getattr(session, "date", None),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "turns": {"total": len(turns), "human": human_turns, "agent": len(turns) - human_turns},
    }

    reply_sources = getattr(session, "reply_sources", None)
    if reply_sources is not None:
        counts = Counter(reply_sources)
        stats["agent_replies"] = {
            "total": len(reply_sources),
            "from_kg_gap": counts.get("gap", 0),
            "from_kg_answer": counts.get("kg_answer", 0),
            "from_llm": counts.get("default", 0),
            # "timeout" -- chat_sessions.ChatTimeoutError -- an OpenAI call this turn depended
            # on didn't respond in time; the human was asked to just try again.
            "timed_out": counts.get("timeout", 0),
        }

    for attr in ("kg_address", "gap_threshold"):
        if hasattr(session, attr):
            stats[attr] = getattr(session, attr)

    annotations = getattr(session, "annotations", None)
    if annotations is not None:
        stats["annotated_turns"] = len(annotations)

    kg_pushes = getattr(session, "kg_pushes", None)
    if kg_pushes is not None:
        stats["kg_pushes"] = {
            "count": len(kg_pushes),
            "capsules": sum((push or {}).get("capsules", 0) for push in kg_pushes),
        }

    asked_gap_keys = getattr(session, "_asked_gap_keys", None)
    if asked_gap_keys is not None:
        stats["gaps_asked"] = {
            "count": len(asked_gap_keys),
            "gaps": [{"subject": subject, "predicate": predicate}
                     for subject, predicate in sorted(asked_gap_keys)],
        }

    if hasattr(session, "_pending_confirmation"):
        pending = session._pending_confirmation
        # Truthy only if the session ended mid-confirmation, i.e. the agent asked "was that
        # you...?" and the window was closed before the human answered.
        stats["unanswered_confirmation"] = None if pending is None else {
            "subject": pending[0].get("subject"),
            "predicate": pending[0].get("predicate"),
            "question": pending[2],
        }

    return stats


def save_session(session, save_dir=DEFAULT_SAVE_DIR, prefix=None):
    """Write `session`'s turns and its session_statistics() to two timestamped JSON files under
    `save_dir` (created if needed), and -- for a KgChatSession, which has one -- its
    `turn_log` (see chat_sessions.KgChatSession's own docstring: one entry per turn listing what
    it pushed to the knowledge graph, what gap queries it ran, and which gap it selected) to a
    third. Returns their paths as (turns_path, stats_path, turn_log_path) -- turn_log_path is
    None for a plain ChatSession, which has no turn_log attribute at all.

    Timestamped rather than fixed names so quitting a second session in the same directory never
    silently clobbers the first one's transcript.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base = prefix or f"chat{getattr(session, 'chat', '')}"
    base = re.sub(r"[^A-Za-z0-9_.-]", "_", str(base))
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")

    turns_path = save_dir / f"{base}_turns_{stamp}.json"
    stats_path = save_dir / f"{base}_stats_{stamp}.json"

    with open(turns_path, "w") as f:
        json.dump(list(getattr(session, "turns", None) or []), f, indent=2)
    with open(stats_path, "w") as f:
        json.dump(session_statistics(session), f, indent=2)

    turn_log = getattr(session, "turn_log", None)
    turn_log_path = None
    if turn_log is not None:
        turn_log_path = save_dir / f"{base}_gaplog_{stamp}.json"
        with open(turn_log_path, "w") as f:
            json.dump(turn_log, f, indent=2)

    return turns_path, stats_path, turn_log_path


class ChatWindow:
    """One Tk window wrapping a single ChatSession/KgChatSession. Build with the session, then
    call .run() (blocks in Tk's mainloop until the window is closed)."""

    def __init__(self, session, quit_words=DEFAULT_QUIT_WORDS, save_dir=DEFAULT_SAVE_DIR):
        self.session = session
        self.quit_words = quit_words
        self.save_dir = save_dir  # None disables the save-on-quit (see _save_on_quit())
        self._busy = False   # a session.say() call is in flight on the background thread
        self._closed = False  # the window has been destroyed -- ignore any late callbacks

        # Only a GraphDB-backed KgChatSession gets a graph panel -- see the module docstring's
        # "graph panel" bullet and graphdb_base_and_repository().
        self._graphdb_base, self._graphdb_repo = graphdb_base_and_repository(getattr(session, "kg_address", None))
        self._show_graph_panel = self._graphdb_repo is not None
        # Graph-panel state, used only when _show_graph_panel -- see _apply_graph_update() et al.
        self._known_subject_uris = []   # most-recent-first, unique
        self._subject_labels = {}       # uri -> display label (rdfs:label if found, else _local_name())
        self._selected_subject_uri = None
        self._opened_subject_uris = set()  # uris already auto-opened in the browser this session
        self._namespace_colors = {}     # "other" namespace -> its assigned color, see _node_fill_color()
        self._graph_font_size = DEFAULT_FONT_SIZE  # adjustable independently of chat_font -- see below

        self.root = tk.Tk()
        self.root.title(f"Leolani KG Chat — {session.human} ({session.date})")
        self.root.geometry("1040x600" if self._show_graph_panel else "640x560")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Named Font OBJECTS, not plain (family, size) tuples: every widget/tag below is
        # configured to reference these objects directly, so reconfiguring just THEIR size (see
        # _on_chat_font_size_change()) live-resizes everything that uses them -- the whole
        # transcript included, with no need to re-insert its already-written text.
        self.chat_font = tkfont.Font(family="Helvetica", size=DEFAULT_FONT_SIZE)
        self.chat_font_bold = tkfont.Font(family="Helvetica", size=DEFAULT_FONT_SIZE, weight="bold")
        self.chat_font_italic = tkfont.Font(family="Helvetica", size=DEFAULT_FONT_SIZE, slant="italic")

        if self._show_graph_panel:
            paned = ttk.PanedWindow(self.root, orient="horizontal")
            paned.pack(fill="both", expand=True)
            chat_frame = tk.Frame(paned)
            graph_frame = tk.Frame(paned)
            paned.add(chat_frame, weight=3)
            paned.add(graph_frame, weight=2)
        else:
            chat_frame = self.root

        self.transcript = scrolledtext.ScrolledText(
            chat_frame, wrap="word", state="disabled", font=self.chat_font, padx=6, pady=6
        )
        self.transcript.pack(fill="both", expand=True, padx=8, pady=(8, 0))
        self.transcript.tag_config("human", foreground="#1a5fb4", font=self.chat_font_bold)
        self.transcript.tag_config("agent_kg", foreground="#2ec27e", font=self.chat_font_bold)
        self.transcript.tag_config("agent_llm", foreground="#c64600", font=self.chat_font_bold)
        self.transcript.tag_config("body", foreground="#1a1a1a", font=self.chat_font)

        # Adjustable independently of the graph panel's own "Font size" slider below -- see
        # _on_chat_font_size_change(). Always shown (not just for a KgChatSession), unlike the
        # gap-sensitivity slider right after it.
        size_frame = tk.Frame(chat_frame)
        size_frame.pack(fill="x", padx=10, pady=(6, 0))
        tk.Label(size_frame, text="Text size:", font=("Helvetica", 10)).pack(side="left")
        self.chat_font_value_label = tk.Label(
            size_frame, text=str(DEFAULT_FONT_SIZE), font=("Helvetica", 10), width=3, anchor="w"
        )
        self.chat_font_scale = tk.Scale(
            size_frame, from_=CHAT_FONT_RANGE[0], to=CHAT_FONT_RANGE[1], resolution=1, orient="horizontal",
            showvalue=False, command=self._on_chat_font_size_change,
        )
        self.chat_font_scale.set(DEFAULT_FONT_SIZE)
        self.chat_font_scale.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.chat_font_value_label.pack(side="left")

        # Only KgChatSession has a gap_threshold to adjust -- plain ChatSession has no such
        # thing, so the control simply isn't built for it (see _on_gap_threshold_change()).
        self.threshold_scale = None
        if hasattr(session, "gap_threshold"):
            threshold_frame = tk.Frame(chat_frame)
            threshold_frame.pack(fill="x", padx=10, pady=(6, 0))
            tk.Label(threshold_frame, text="Gap sensitivity:", font=("Helvetica", 10)).pack(side="left")
            self.threshold_value_label = tk.Label(
                threshold_frame, text=f"{session.gap_threshold:.2f}", font=("Helvetica", 10), width=4, anchor="w"
            )
            self.threshold_scale = tk.Scale(
                threshold_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
                showvalue=False, command=self._on_gap_threshold_change,
            )
            self.threshold_scale.set(session.gap_threshold)
            self.threshold_scale.pack(side="left", fill="x", expand=True, padx=(6, 6))
            self.threshold_value_label.pack(side="left")

        self.status_label = tk.Label(
            chat_frame, text="", font=self.chat_font_italic, fg="#888888", anchor="w"
        )
        self.status_label.pack(fill="x", padx=10)

        entry_frame = tk.Frame(chat_frame)
        entry_frame.pack(fill="x", padx=8, pady=8)
        self.entry = tk.Entry(entry_frame, font=self.chat_font)
        self.entry.pack(side="left", fill="x", expand=True, ipady=4)
        self.entry.bind("<Return>", self._on_send)
        self.entry.focus_set()
        self.send_button = tk.Button(entry_frame, text="Send", font=self.chat_font, command=self._on_send)
        self.send_button.pack(side="left", padx=(6, 0))
        # Always enabled (unlike send_button, which is disabled while a reply is in flight) --
        # same as typing a quit word, just a click instead: an always-available way out even if
        # a reply never comes back. See _on_close()/quit_words.
        self.quit_button = tk.Button(entry_frame, text="Quit", font=self.chat_font, command=self._on_close)
        self.quit_button.pack(side="left", padx=(6, 0))

        if self._show_graph_panel:
            self._build_graph_panel(graph_frame)

        self._replay_existing_turns()

        # If `session` already had activity turns pushed before this window was built (e.g. a
        # notebook resuming an existing KgChatSession), prime the panel right away instead of
        # leaving it empty until the next turn -- one background fetch, same as a normal update.
        #
        # Starting that background thread is deferred via root.after(0, ...) rather than done
        # directly here: __init__() runs BEFORE run()'s root.mainloop() call, so a thread
        # started here can -- if fetch_triples() happens to return fast enough -- reach its own
        # self._schedule()'s root.after() call before mainloop() has actually started, which
        # Tkinter rejects ("main thread is not in main loop"); _schedule()'s blanket
        # `except RuntimeError: pass` then silently swallows that, so the panel would just never
        # get primed. root.after(0, ...) called from the MAIN thread, by contrast, is always
        # safe (queued regardless of whether mainloop() has started yet) and is guaranteed to
        # fire only once it has -- so starting the thread from inside that callback guarantees
        # mainloop() is already running by the time IT calls .after() later, from its own thread.
        if self._show_graph_panel:
            initial_uris = list(getattr(session, "last_subject_uris", None) or [])
            if initial_uris:
                self.root.after(0, lambda: threading.Thread(
                    target=self._initial_graph_fetch, args=(initial_uris,), daemon=True
                ).start())

    # ------------------------------------------------------------------- #
    # Rendering
    # ------------------------------------------------------------------- #

    def _replay_existing_turns(self):
        """Render any turns `session` already had before the window was built (e.g. from
        ChatSession.open_with()), tagged with the matching reply_sources entry where there is
        one. reply_sources holds one entry per agent turn produced by say() -- an open_with()
        turn (always turn index 0, if present) predates say() entirely and so has no entry."""
        reply_sources = list(getattr(self.session, "reply_sources", None) or [])
        agent_indices = [i for i, t in enumerate(self.session.turns) if t["speaker"] != self.session.human]
        if agent_indices and agent_indices[0] == 0:
            agent_indices = agent_indices[1:]  # the open_with() turn, if any -- no source for it
        source_by_index = dict(zip(agent_indices, reply_sources))
        for i, turn in enumerate(self.session.turns):
            self._append_turn(turn["speaker"], turn["utterance"], source_by_index.get(i))

    def _append_turn(self, speaker, utterance, source):
        """source is a chat_sessions.KgChatSession.reply_sources entry -- "gap" (KG-grounded),
        "kg_answer" (the human's question answered from the KG, see kg_question_answerer.py),
        "default" (LLM fallback), or None (a plain ChatSession has no reply_sources at all, or
        this is the human's own turn)."""
        self.transcript.configure(state="normal")
        if speaker == self.session.human:
            self.transcript.insert("end", f"{speaker}: ", "human")
        else:
            if source == "gap":
                tag, label = "agent_kg", "agent [KG]"
            elif source == "kg_answer":
                tag, label = "agent_kg", "agent [KG answer]"
            else:
                tag, label = "agent_llm", "agent [LLM]"
            self.transcript.insert("end", f"{label}: ", tag)
        self.transcript.insert("end", f"{utterance}\n\n", "body")
        self.transcript.configure(state="disabled")
        self.transcript.see("end")

    # ------------------------------------------------------------------- #
    # Sending a turn
    # ------------------------------------------------------------------- #

    def _on_send(self, event=None):
        utterance = self.entry.get().strip()
        if not utterance:
            return
        self.entry.delete(0, "end")

        # Checked first, and independent of _busy: a reply already in flight (SPARQL/OpenAI can
        # each take a while, occasionally hang outright) used to leave the whole window stuck --
        # the entry box got disabled until that call returned, so there was no way to even type
        # "bye" to get out. The entry box is never disabled (see below), so this always runs.
        if utterance.lower() in self.quit_words:
            self._on_close()
            return

        if self._busy:
            self.status_label.configure(text="still waiting on the previous reply… (you can still type \"bye\" to quit)")
            return

        self._append_turn(self.session.human, utterance, None)
        self._busy = True
        self.send_button.configure(state="disabled")
        self.status_label.configure(text="agent is thinking…")
        threading.Thread(target=self._get_reply, args=(utterance,), daemon=True).start()

    def _get_reply(self, utterance):
        """Runs off the main thread: session.say() does the SPARQL push, KG gap lookup and
        LLM call(s), any of which can take a few seconds. Only schedules its result back onto
        the main thread (via root.after) instead of touching Tk widgets directly here."""
        try:
            self.session.say(utterance)
        except Exception as exc:  # surfaced in the window instead of silently dying in the thread
            self._schedule(self._on_reply_error, exc)
            return
        # Also runs on THIS background thread (not the main one): it's a network round-trip to
        # GraphDB, same reasoning as session.say() itself. graph_update is None when there's no
        # graph panel at all (_refresh_graph_data() checks _show_graph_panel first).
        try:
            graph_update = self._refresh_graph_data()
        except Exception:
            graph_update = None  # a graph-panel hiccup must never break the chat turn itself
        self._schedule(self._on_reply_done, graph_update)

    def _schedule(self, callback, *args):
        """root.after(), but a no-op once the window has been closed -- e.g. the human typed
        "bye" while a reply was still in flight on the background thread; that reply can still
        land after self.root has been destroyed, and .after() on a destroyed root raises."""
        if self._closed:
            return
        try:
            self.root.after(0, callback, *args)
        except RuntimeError as exc:
            if "main loop" in str(exc):
                # Not the "window closed" case this except clause exists for -- a background
                # thread called this before root.mainloop() had actually started (Tkinter
                # rejects .after() from a non-main thread until it has). Should no longer
                # happen -- see the root.after(0, lambda: threading.Thread(...))) wrapper around
                # _initial_graph_fetch()'s thread start in __init__ -- but surfaced loudly
                # rather than silently dropped if it ever does, since it isn't the benign,
                # expected case below.
                print(f"[kg_chat_gui] dropped a scheduled update -- {exc}")
            # else: window closed in the (unlikely) gap between the check above and this call

    def _on_reply_done(self, graph_update=None):
        if self._closed:
            return
        agent_turn = self.session.turns[-1]
        reply_sources = getattr(self.session, "reply_sources", None)
        source = reply_sources[-1] if reply_sources else None
        self._append_turn(agent_turn["speaker"], agent_turn["utterance"], source)
        self._apply_graph_update(graph_update)
        self._ready_for_input()

    def _on_reply_error(self, exc):
        if self._closed:
            return
        self.status_label.configure(text=f"error getting a reply: {exc}")
        self._ready_for_input(clear_status=False)

    def _ready_for_input(self, clear_status=True):
        self._busy = False
        if clear_status:
            self.status_label.configure(text="")
        self.send_button.configure(state="normal")
        self.entry.focus_set()

    # ------------------------------------------------------------------- #
    # Live font-size control -- the conversation (see __init__'s "Text size" slider). The
    # graph's own, independent "Font size" slider is _on_graph_font_size_change(), below.
    # ------------------------------------------------------------------- #

    def _on_chat_font_size_change(self, value_str):
        """Scale `command` callback for the "Text size" slider. Reconfiguring chat_font/
        chat_font_bold/chat_font_italic's `size` is enough on its own -- every widget and every
        transcript tag was built referencing these Font OBJECTS (not plain (family, size)
        tuples), so Tk resizes all of them, already-inserted transcript text included, with no
        need to touch the Text widget's content directly."""
        size = int(float(value_str))
        self.chat_font.configure(size=size)
        self.chat_font_bold.configure(size=size)
        self.chat_font_italic.configure(size=size)
        self.chat_font_value_label.configure(text=str(size))

    # ------------------------------------------------------------------- #
    # Live gap_threshold control (KgChatSession only -- see __init__)
    # ------------------------------------------------------------------- #

    def _on_gap_threshold_change(self, value_str):
        """Scale `command` callback: fires on every drag tick with the slider's new value as a
        string. Updates session.gap_threshold directly -- kg_gap_finder.analyze() reads it fresh
        on every query (KgChatSession._fetch_gap_queue()), so this takes effect immediately for
        any *new* query. Also drops any already-cached per-instance gap queues
        (KgChatSession._gap_queues), which were fetched at the OLD threshold: without this, the
        new sensitivity would only kick in once whatever gaps happened to already be queued for
        an activity finished draining on their own, rather than right away."""
        value = float(value_str)
        self.session.gap_threshold = value
        self.threshold_value_label.configure(text=f"{value:.2f}")
        gap_queues = getattr(self.session, "_gap_queues", None)
        if gap_queues is not None:
            gap_queues.clear()

    # ------------------------------------------------------------------- #
    # Graph panel (GraphDB-backed KgChatSession only -- see __init__)
    #
    # Drawn as an actual node-link diagram, in-window, on a tk.Canvas (see _render_graph()) --
    # the activity as one centered node, every triple pushed for it as a labeled edge to its own
    # object node. This isn't GraphDB's own interactive D3 force-graph (Tkinter has no
    # embeddable browser/JS engine to reproduce that), but it IS a real, drawn graph, not text --
    # "Open in GraphDB ↗" is there for the real thing (drag nodes, expand further, ...) in a
    # browser tab whenever that's wanted instead.
    # ------------------------------------------------------------------- #

    def _build_graph_panel(self, graph_frame):
        """Build every widget in the right-hand graph panel. Only called when
        _show_graph_panel is True (see __init__)."""
        tk.Label(graph_frame, text="Knowledge Graph", font=("Helvetica", 12, "bold")).pack(
            anchor="w", padx=8, pady=(8, 4)
        )

        # Every activity mentioned so far, most-recent-first (see _known_subject_uris) --
        # picking one switches which activity's diagram/link the rest of the panel shows.
        self._activity_var = tk.StringVar()
        self.activity_combo = ttk.Combobox(graph_frame, textvariable=self._activity_var, state="readonly")
        self.activity_combo.pack(fill="x", padx=8)
        self.activity_combo.bind("<<ComboboxSelected>>", self._on_activity_selected)

        link_frame = tk.Frame(graph_frame)
        link_frame.pack(fill="x", padx=8, pady=(6, 0))
        self.open_graph_button = tk.Button(
            link_frame, text="Open in GraphDB ↗", command=self._open_selected_in_browser, state="disabled"
        )
        self.open_graph_button.pack(side="left")
        # Off by default -- opening a new browser tab for every freshly-mentioned activity,
        # unasked, would otherwise interrupt whatever else the browser is doing mid-conversation.
        self.auto_open_var = tk.BooleanVar(value=False)
        tk.Checkbutton(
            link_frame, text="auto-open new activities", variable=self.auto_open_var
        ).pack(side="left", padx=(10, 0))

        # Adjustable independently of the conversation's own "Text size" slider (see __init__) --
        # redraws the SAME cached data at the new size (_on_graph_font_size_change()), no re-fetch.
        graph_size_frame = tk.Frame(graph_frame)
        graph_size_frame.pack(fill="x", padx=8, pady=(6, 0))
        tk.Label(graph_size_frame, text="Font size:", font=("Helvetica", 10)).pack(side="left")
        self.graph_font_value_label = tk.Label(
            graph_size_frame, text=str(DEFAULT_FONT_SIZE), font=("Helvetica", 10), width=3, anchor="w"
        )
        self.graph_font_scale = tk.Scale(
            graph_size_frame, from_=GRAPH_FONT_RANGE[0], to=GRAPH_FONT_RANGE[1], resolution=1,
            orient="horizontal", showvalue=False, command=self._on_graph_font_size_change,
        )
        self.graph_font_scale.set(DEFAULT_FONT_SIZE)
        self.graph_font_scale.pack(side="left", fill="x", expand=True, padx=(6, 6))
        self.graph_font_value_label.pack(side="left")

        # What each node color means (see _node_fill_color()) -- a small swatch + label per
        # fixed namespace, plus one entry noting that any other namespace still gets its own
        # (just not individually listed here, since that set is open-ended and discovered live).
        legend_frame = tk.Frame(graph_frame)
        legend_frame.pack(fill="x", padx=8, pady=(4, 0))
        for name, color in (
            ("n2mu", NAMESPACE_COLORS["n2mu"]), ("gaf", NAMESPACE_COLORS["gaf"]),
            ("grasp", NAMESPACE_COLORS["grasp"]), ("other", OTHER_NAMESPACE_PALETTE[0]),
        ):
            tk.Label(legend_frame, text="  ", background=color, relief="solid", borderwidth=1).pack(
                side="left", padx=(0, 2)
            )
            tk.Label(legend_frame, text=name, font=("Helvetica", 8)).pack(side="left", padx=(0, 8))

        # State the diagram is drawn from -- kept around so _on_canvas_resize()/a font-size
        # change can redraw without a network round-trip when nothing but the size changed.
        self._graph_center_label = None
        self._graph_center_uri = None
        self._graph_triples = []

        self.graph_canvas = tk.Canvas(graph_frame, background=GRAPH_BG, highlightthickness=0)
        self.graph_canvas.pack(fill="both", expand=True, padx=8, pady=8)
        self.graph_canvas.bind("<Configure>", self._on_canvas_resize)
        self._render_graph(None, None, [])  # the "(nothing pushed yet)" placeholder

        self.graph_status_label = tk.Label(
            graph_frame, text="", font=("Helvetica", 9, "italic"), fg="#888888", anchor="w"
        )
        self.graph_status_label.pack(fill="x", padx=8, pady=(0, 8))

    def _refresh_graph_data(self):
        """Runs on the background thread that just ran session.say() (called from _get_reply(),
        right after it returns): reads session.last_subject_uris (see chat_sessions.KgChatSession)
        and fetches the triples for whichever activity should now be shown -- the newest one
        that appeared this turn if any, else whatever was already selected. Returns a payload
        dict for _apply_graph_update() to render on the main thread, or None if there's no graph
        panel to update at all."""
        if not self._show_graph_panel:
            return None
        new_uris = list(getattr(self.session, "last_subject_uris", None) or [])
        selected = new_uris[0] if new_uris else self._selected_subject_uri
        triples, error = [], None
        if selected:
            try:
                triples = fetch_triples(self.session.kg_address, selected)
            except Exception as exc:
                error = str(exc)
        return {"new_uris": new_uris, "selected_uri": selected, "triples": triples, "error": error}

    def _initial_graph_fetch(self, uris):
        """Like _refresh_graph_data(), but for priming the panel from __init__ when `session`
        already had activities pushed before this window was built -- runs on its own background
        thread (started in __init__), not the reply-handling one."""
        selected = uris[0]
        try:
            triples, error = fetch_triples(self.session.kg_address, selected), None
        except Exception as exc:
            triples, error = [], str(exc)
        payload = {"new_uris": uris, "selected_uri": selected, "triples": triples, "error": error}
        self._schedule(self._apply_graph_update, payload)

    def _apply_graph_update(self, payload):
        """Main-thread renderer for a _refresh_graph_data()/_initial_graph_fetch()/
        _on_activity_selected() payload: updates the known-activities list, the selected
        activity's label and triples, and (if enabled) auto-opens the browser for any activity
        in `payload["new_uris"]` never auto-opened before this session. `payload` is None when
        there's no graph panel (see _refresh_graph_data()) -- a no-op in that case."""
        if not payload:
            return
        for uri in payload["new_uris"]:
            if uri not in self._known_subject_uris:
                self._known_subject_uris.insert(0, uri)

        selected = payload["selected_uri"]
        if selected:
            if selected not in self._known_subject_uris:
                self._known_subject_uris.insert(0, selected)
            # Prefer the subject's own rdfs:label (present among its triples) over a raw URI.
            label = next((o for p, o in payload["triples"] if p == LABEL_PREDICATE), None)
            self._subject_labels[selected] = label or self._subject_labels.get(selected) or _local_name(selected)
            self._selected_subject_uri = selected
            self._render_graph(self._subject_labels[selected], selected, payload["triples"])

        self._refresh_activity_combo()
        self.graph_status_label.configure(
            text=f"couldn't refresh: {payload['error']}" if payload["error"] else ""
        )

        if self.auto_open_var.get():
            for uri in payload["new_uris"]:
                if uri not in self._opened_subject_uris:
                    self._opened_subject_uris.add(uri)
                    self._open_in_browser(uri)

    def _refresh_activity_combo(self):
        """Sync the activity dropdown's entries/selection to _known_subject_uris/_subject_labels,
        and enable/disable the Open-in-GraphDB button to match whether anything is selected."""
        self.activity_combo["values"] = [
            self._subject_labels.get(uri, _local_name(uri)) for uri in self._known_subject_uris
        ]
        if self._selected_subject_uri in self._known_subject_uris:
            self.activity_combo.current(self._known_subject_uris.index(self._selected_subject_uri))
        self.open_graph_button.configure(state="normal" if self._selected_subject_uri else "disabled")

    def _on_canvas_resize(self, event=None):
        """<Configure> handler: the graph pane was resized (dragging the PanedWindow's sash, or
        the whole window) -- redraw the SAME data at the new size. No network round-trip, so
        this is cheap enough to just redo on every resize event rather than debouncing it.
        Passes the event's own width/height straight through rather than letting _render_graph()
        query winfo_width()/winfo_height() itself -- see there for why that matters here."""
        self._render_graph(
            self._graph_center_label, self._graph_center_uri, self._graph_triples,
            width=event.width if event else None, height=event.height if event else None,
        )

    def _on_graph_font_size_change(self, value_str):
        """Scale `command` callback for the graph panel's own "Font size" slider -- independent
        of the conversation's "Text size" one (_on_chat_font_size_change()). Unlike that one,
        canvas text items don't live-resize themselves when a Font object is reconfigured the
        way Text-widget-tagged text does, so this just re-runs _render_graph() on the already-
        cached data (self._graph_center_label/_uri/_triples) -- no network round-trip, same as
        a resize."""
        self._graph_font_size = int(float(value_str))
        self.graph_font_value_label.configure(text=str(self._graph_font_size))
        self._render_graph(self._graph_center_label, self._graph_center_uri, self._graph_triples)

    def _node_fill_color(self, uri_or_literal):
        """The fill color for one graph node, by RDF namespace (_namespace_of()) -- the fixed
        red/blue/green NAMESPACE_COLORS for n2mu/gaf/grasp, LITERAL_NODE_FILL for a plain
        literal (no URI at all, e.g. "the park"), and a color from OTHER_NAMESPACE_PALETTE for
        any other namespace -- assigned the first time that namespace is seen and cached in
        self._namespace_colors, so it stays the SAME color for the rest of this window's life,
        not just this one render."""
        namespace = _namespace_of(uri_or_literal)
        if namespace is None:
            return LITERAL_NODE_FILL
        if namespace in NAMESPACE_COLORS:
            return NAMESPACE_COLORS[namespace]
        if namespace not in self._namespace_colors:
            self._namespace_colors[namespace] = OTHER_NAMESPACE_PALETTE[
                len(self._namespace_colors) % len(OTHER_NAMESPACE_PALETTE)
            ]
        return self._namespace_colors[namespace]

    def _render_graph(self, center_label, center_uri, triples, width=None, height=None):
        """Draw `triples` (a fetch_triples() result -- (predicate, object) tuples for the
        activity `center_uri`, labeled `center_label`) as a node-link diagram on
        self.graph_canvas: the activity as one node at the center, one labeled edge per triple
        fanning out to that triple's object as its own node, arranged evenly around a circle.
        This is a real, drawn graph -- not GraphDB's own interactive D3 force-graph (see this
        section's own comment above for why not), but genuinely a picture of the graph, not
        text. Every node is filled by its own RDF namespace's color (_node_fill_color()).
        Node/center circle radii (node_r/center_r, below) scale roughly proportionally with
        self._graph_font_size, not by a small fixed add-on -- so a label keeps roughly the same
        amount of room, and wraps onto roughly the same number of lines, at any font size rather
        than the circle staying about the same size while the text inside it keeps growing.

        Remembers (center_label, center_uri, triples) in self._graph_center_label/_uri/
        self._graph_triples so _on_canvas_resize()/_on_graph_font_size_change() can redraw the
        same data at a new canvas size or font size with no re-fetch.

        `width`/`height` are normally omitted (read from the canvas's own current size instead)
        -- callers only pass them from within a <Configure> handler (see _on_canvas_resize()),
        using that event's own width/height instead of re-querying winfo_width()/winfo_height().
        That distinction matters: querying them via canvas.update_idletasks() from INSIDE a
        <Configure> handler can synchronously re-enter that same handler (Tk processing the very
        geometry event that's still being handled), which used to double up every item this
        drew -- each nested call correctly cleared the canvas, but then BOTH the inner and the
        resumed outer call went on to redraw the same diagram on top of each other.
        """
        self._graph_center_label = center_label
        self._graph_center_uri = center_uri
        self._graph_triples = triples

        # rdfs:label triples aren't drawn as their own peripheral node/edge: the label value is
        # already what the CENTER node's own text shows (see _apply_graph_update()'s
        # LABEL_PREDICATE lookup for center_label), so drawing "label -> <that same text>" again
        # as a separate edge/node would just repeat, on screen, exactly what the center node
        # already says. A subject can also legitimately have SEVERAL rdfs:label triples (e.g.
        # one per turn that introduced a new phrase for the same activity -- see
        # prompts.response_processor._label_for_subject()'s own docstring), which would
        # otherwise draw as several near-duplicate peripheral nodes.
        #
        # gaf:denotedIn/denotedBy (GAF_PROVENANCE_PREDICATES) are excluded for a similar reason:
        # pure extraction provenance (which utterance span a fact came from), not semantic
        # content, and typically several per subject -- one per turn that ever mentioned it --
        # so drawing them would clutter the diagram with utterance-span nodes nobody asked about.
        #
        # sem:eventProperty/eps:contextProperty (GENERIC_ANCESTOR_PREDICATES) are excluded for yet
        # another reason: on a repository with inference enabled, these are the same object every
        # real role predicate already draws an edge for, just materialized again under a generic
        # ancestor property -- drawing them too would duplicate every single edge in the diagram.
        #
        # sem:hasActor/hasPlace/hasTime are a related but different case: unlike the two generic
        # ancestors above they DO carry real meaning, so rather than dropping them outright,
        # _merge_equivalent_edges() (see EDGE_EQUIVALENCE_GROUPS) collapses each one down to a
        # single edge with whichever real n2mu: role predicate (agent/agent_patient/participant/
        # experiencer, location, or one of the four time/* variants) shares the SAME object --
        # inference otherwise materializes both to that object, which would draw as two (or more)
        # near-identical edges for what both really are the same one fact.
        drawable_triples = _merge_equivalent_edges([
            (p, o) for p, o in triples
            if p != LABEL_PREDICATE
            and p not in GAF_PROVENANCE_PREDICATES
            and p not in GENERIC_ANCESTOR_PREDICATES
        ])

        # Node/text sizes all scale off the one "Font size" slider value. Node/center radii scale
        # roughly PROPORTIONALLY with it (not by a small fixed add-on) -- a fixed add-on barely
        # grows the circle at all across the slider's range, while the text drawn inside it keeps
        # getting bigger, so it wraps onto more and more lines the higher the slider goes. Scaling
        # the radius with font_size instead keeps roughly the same amount of text fitting per line
        # (and thus about the same number of wrapped lines) at any size on the slider.
        font_size = self._graph_font_size
        center_font = font_size
        node_font = max(6, font_size - 2)
        edge_font = max(6, font_size - 6)
        node_r = max(24, round(font_size * 1.8))
        center_r = max(32, round(font_size * 2.4))

        canvas = self.graph_canvas
        canvas.delete("all")
        if width is None or height is None:
            width, height = canvas.winfo_width(), canvas.winfo_height()
        if width <= 1 or height <= 1:
            # Not actually mapped/sized yet (e.g. called from _build_graph_panel() before the
            # window has done its first layout pass) -- fall back to a sane default so the
            # placeholder/first draw isn't degenerate; a real <Configure> event follows shortly
            # after and redraws at the true size anyway.
            width, height = 380, 380
        cx, cy = width / 2, height / 2

        if center_label is None:
            canvas.create_text(
                cx, cy, text="(nothing pushed to the knowledge graph yet)",
                fill=GRAPH_PLACEHOLDER_TEXT, font=("Helvetica", node_font, "italic"), width=width - 40,
            )
            return

        canvas.create_oval(
            cx - center_r, cy - center_r, cx + center_r, cy + center_r,
            fill=self._node_fill_color(center_uri), outline=GRAPH_NODE_OUTLINE, width=2,
        )
        canvas.create_text(
            cx, cy, text=_truncate(center_label, 40), fill=GRAPH_NODE_TEXT,
            font=("Helvetica", center_font, "bold"), width=center_r * 1.9, justify="center",
        )

        if not drawable_triples:
            canvas.create_text(
                cx, cy + center_r + 20, text="(no triples yet for this activity)",
                fill=GRAPH_PLACEHOLDER_TEXT, font=("Helvetica", edge_font, "italic"), width=width - 40,
            )
            return

        n = len(drawable_triples)
        # The layout radius (distance from the center node to each peripheral one) grows with
        # center_r/node_r too, not just the canvas size -- otherwise bigger nodes (a higher font
        # size) would start crowding/overlapping each other and clipping off the canvas edge
        # instead of just having more text room individually.
        radius = max(100, min(width, height) / 2 - center_r - node_r - 20)
        for i, (predicate, obj) in enumerate(drawable_triples):
            angle = (2 * math.pi * i / n) - (math.pi / 2)  # start straight up, go clockwise
            dx, dy = math.cos(angle), math.sin(angle)
            nx = cx + radius * dx
            ny = cy + radius * dy

            # Clipped to each node's own boundary (center_r/node_r out from its center), not the
            # node centers themselves -- the center node's circle is drawn BEFORE this line, so a
            # line running all the way to (cx, cy) would be painted on top of it, visibly cutting
            # into the node instead of stopping at its edge.
            edge_x0, edge_y0 = cx + center_r * dx, cy + center_r * dy
            edge_x1, edge_y1 = nx - node_r * dx, ny - node_r * dy
            canvas.create_line(edge_x0, edge_y0, edge_x1, edge_y1, fill=GRAPH_EDGE_COLOR, width=1.5)
            mx, my = (cx + nx) / 2, (cy + ny) / 2
            canvas.create_text(
                mx, my, text=_truncate(_local_name(predicate), 18), fill=GRAPH_EDGE_LABEL_COLOR,
                font=("Helvetica", edge_font, "italic"),
            )

            canvas.create_oval(
                nx - node_r, ny - node_r, nx + node_r, ny + node_r,
                fill=self._node_fill_color(obj), outline=GRAPH_NODE_OUTLINE, width=1.5,
            )
            canvas.create_text(
                nx, ny, text=_truncate(_local_name(obj), 22), fill=GRAPH_NODE_TEXT,
                font=("Helvetica", node_font), width=node_r * 2.0, justify="center",
            )

    def _on_activity_selected(self, event=None):
        """Combobox <<ComboboxSelected>> handler: the human picked a different activity from the
        dropdown. Fetching its triples is a network call, so it's dispatched to its own
        background thread rather than run inline here on the UI thread."""
        index = self.activity_combo.current()
        if index < 0 or index >= len(self._known_subject_uris):
            return
        uri = self._known_subject_uris[index]
        self._selected_subject_uri = uri
        self.open_graph_button.configure(state="normal")
        threading.Thread(target=self._fetch_and_show, args=(uri,), daemon=True).start()

    def _fetch_and_show(self, uri):
        """Background-thread half of _on_activity_selected(). new_uris is deliberately [] here
        -- switching the dropdown to an already-known activity must not re-trigger auto-open or
        reorder the "most recent" list, both of which are only for genuinely new activities."""
        try:
            triples, error = fetch_triples(self.session.kg_address, uri), None
        except Exception as exc:
            triples, error = [], str(exc)
        payload = {"new_uris": [], "selected_uri": uri, "triples": triples, "error": error}
        self._schedule(self._apply_graph_update, payload)

    def _open_selected_in_browser(self):
        if self._selected_subject_uri:
            self._open_in_browser(self._selected_subject_uri)

    def _open_in_browser(self, uri):
        """Open GraphDB's own interactive "Visual graph" view for `uri` in the system's default
        browser -- see graphdb_visualization_url(). new=0 asks the browser to reuse an existing
        window/tab where it can, so repeated opens (e.g. auto-open across several activities)
        don't necessarily pile up a fresh tab each time -- browser-dependent, not guaranteed."""
        url = graphdb_visualization_url(self.session.kg_address, uri)
        if url:
            webbrowser.open(url, new=0)

    # ------------------------------------------------------------------- #
    # Quitting
    # ------------------------------------------------------------------- #

    def _save_on_quit(self):
        """Write the conversation + statistics (+, for a KgChatSession, its turn-by-turn gap
        log) out via save_session(), reporting the paths on stdout (so they land in the notebook
        cell's output -- the window itself is about to go away). Never raises: a failed save
        must not stop the window from closing, which is the whole point of clicking Quit."""
        if self.save_dir is None:
            return
        try:
            turns_path, stats_path, turn_log_path = save_session(self.session, self.save_dir)
        except Exception as exc:
            print(f"[kg_chat_gui] could NOT save the conversation: {exc!r}")
            return
        print(f"[kg_chat_gui] conversation saved to {turns_path}")
        print(f"[kg_chat_gui] statistics saved to  {stats_path}")
        if turn_log_path is not None:
            print(f"[kg_chat_gui] gap log saved to     {turn_log_path}")

    def _on_close(self):
        # Idempotent: reachable from the Quit button, a quit word, the window manager's close
        # button, AND run()'s finally -- the save below must happen exactly once.
        if self._closed:
            return
        self._closed = True
        # Hide the window FIRST, before anything else (including the save, which is normally
        # fast but has no hard time bound -- e.g. a slow disk). withdraw() is synchronous and
        # near-instant, so this is what actually answers "the Quit button doesn't close the
        # window" even in a run where a later step below has a problem: the window visibly goes
        # away the moment Quit is clicked, rather than only after save/quit/destroy all succeed.
        try:
            self.root.withdraw()
            # withdraw() only queues the hide -- it isn't actually sent to the window server
            # until the next event-loop pump. Everything below runs synchronously, in the same
            # callback withdraw() was just called from (mainloop() doesn't get to process
            # anything else until this whole method returns), so without forcing that pump here,
            # the OS can go on showing the last-painted frame of the window for the entire
            # remaining time _on_close() takes (save_session() included) -- a known Tk/Aqua
            # quirk that reads exactly as "clicking Quit doesn't close the window". update()
            # forces it through right away, before that later work even starts.
            self.root.update()
        except Exception as exc:
            print(f"[kg_chat_gui] couldn't hide the window on quit: {exc!r}")
        self._save_on_quit()
        # destroy() alone tears down the widgets but does NOT necessarily make a running
        # mainloop() return -- under IPython/Jupyter (whose own event-loop integration may be
        # driving or nesting the loop) that left the loop spinning, so the window stayed on
        # screen and the notebook cell never finished. quit() is what actually ends mainloop();
        # call it first, then destroy(), and tolerate either already being gone -- but print
        # rather than silently swallow a real failure here, since withdraw() above only hides
        # the window; if quit()/destroy() themselves keep failing, the process/kernel never
        # actually frees it (visible as, e.g., the notebook cell never finishing).
        for step in (self.root.quit, self.root.destroy):
            try:
                step()
            except Exception as exc:
                print(f"[kg_chat_gui] {step.__name__}() failed on quit: {exc!r}")

    def run(self):
        try:
            self.root.mainloop()
        finally:
            # mainloop() can also return without _on_close() having run (an interrupted kernel,
            # an externally driven loop). Closing here too means the window is always torn down
            # and the transcript always saved, exactly once (see _on_close()'s guard).
            self._on_close()


def run_gui(session, quit_words=DEFAULT_QUIT_WORDS, save_dir=DEFAULT_SAVE_DIR) -> list:
    """Open a chat window for `session` (a ChatSession or KgChatSession) and block until it's
    closed (via the window's Quit button, its close button, or by typing one of `quit_words`).
    The human types directly into the window's own input box -- no input()/popup -- and every
    agent turn is shown tagged [KG] or [LLM] (see ChatWindow._append_turn()).

    On quit, the conversation and its statistics -- plus, for a `KgChatSession`, its
    turn-by-turn gap log -- are written as timestamped JSON files under `save_dir` (see
    save_session()/session_statistics()), and their paths printed; pass save_dir=None to skip
    that.

    Returns session.turns, same as ChatSession.run_interactive(), so it's a drop-in replacement
    in notebook cells."""
    ChatWindow(session, quit_words=quit_words, save_dir=save_dir).run()
    return session.turns
