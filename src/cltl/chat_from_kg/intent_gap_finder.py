#!/usr/bin/env python3
"""
intent_gap_finder.py
=====================

A domain-knowledge-driven sibling of kg_gap_finder.py: instead of deriving "what's expected"
from peer statistics (kg_gap_finder's B/D/E gap kinds -- "most instances of this class have
this predicate"), this module reads that expectation straight from a hand-authored intent
definition (see the intents/ folder at the project root, loaded by load_intents()), one per
activity type (data_type.ActivityType -- "take_food", "physical condition", ...).

Why not just use kg_gap_finder? Its B/D/E gaps only fire once a MAJORITY of an activity's own
peers (other instances of the same class already in the graph) have the predicate/pair in
question -- see find_predicate_gaps()'s docstring. The very FIRST "take_food" activity ever
pushed to the graph has no peers at all, so nothing can ever count as "expected" for it yet, and
kg_gap_finder reports no gap -- even though a diet-coaching intent clearly wants to know what was
eaten. Intents supply that expectation directly and unconditionally, independent of how many
(if any) similar activities already exist.

Each intent is a JSON object (see intents/*.json -- each file holds a JSON list of these, one
per activity type it covers) of the shape:

    {
      "activity_types": ["take_food"],
      "patient_type": ["food"],
      "activity_qualification": ["duration", "degree"],
      "activity_date": "date",
      "secondary_objectives": {
        "patient_qualification": "quantity",
        "activity_location": "place"
      }
    }

- "activity_types": which activity type(s) (local names as they appear in the graph's own
  rdf:type -- see find_intent_for_activity_type()) this intent applies to.
- "activity_labels" (optional): disambiguates between several intents that share one
  activity_type, by the activity's own label/phrase instead -- needed because
  data_type.ActivityType has only one generic "symptom" type covering many distinct real
  symptoms (a headache, blurry vision, ...), which only show up as an activity's LABEL, never
  its type. See find_intent() and this project's own symptom_intents.json, which declares four
  separate intents all typed "symptom", each naming the one symptom phrase (e.g. "headache")
  its own "activity_labels" covers.
- "patient_type": the activity's `patient` role must point at an object of one of these
  RoleType-style local names (e.g. "food", "drink", "medication") -- what was eaten/drunk/taken.
- "activity_qualification": one `qualification` value expected per named aspect, in order (e.g.
  "duration" then "degree") -- see _qualification_gap() for why aspects are distinguished by
  position, not by content.
- "activity_date": "date" -- the activity must have a `time` value of some kind.
- "secondary_objectives": only checked once every field above is satisfied ("[t]he data
  elements ... must first be met before moving on with the conversation") -- see
  next_intent_gap()'s docstring for exactly how each entry is interpreted.

All four are optional; an intent with none of them is trivially always satisfied. Checks run in
a fixed priority order and stop at (return) the first one not yet met -- see next_intent_gap().

Gap rows this module builds are deliberately kept in the same shape kg_gap_finder.py's
find_predicate_gaps()/find_predicate_object_gaps() rows use (class/predicate/subject/
subject_triples/object_triples/peer_examples[/object_type]), so
prompts.response_processor.PromptProcessor.get_prompt_for_kg_gap(gap, kind, human=...) turns
either kind of gap into a follow-up question the exact same way -- see
notebooks/chat_sessions.KgIntentChatSession, which swaps only the gap-finding step
(_fetch_gap_queue()) of the existing KgChatSession machinery for this module's
next_intent_gap(), and falls back to the ordinary LLM reply whenever find_intent() finds no
match at all (including the "several symptom-like intents share one type, but the activity's own
label doesn't match any of their activity_labels" case -- see find_intent()'s own docstring for
why that's treated as no match rather than an arbitrary guess).

Requires: rdflib (see kg_gap_finder.py, imported below for graph loading/querying).
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

try:
    import kg_gap_finder
except ImportError:
    # Standalone import from outside chat_from_kg/ (Python already puts a script's own
    # directory on sys.path when run directly, so this only matters when intent_gap_finder is
    # imported as a module from elsewhere) -- mirrors kg_gap_finder._import_data_type()'s own
    # sibling-directory fallback.
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import kg_gap_finder


# --------------------------------------------------------------------------- #
# Loading intents/*.json
# --------------------------------------------------------------------------- #

def _default_intents_dir() -> Path:
    """intents/ at the project root, found by walking up from the current working directory --
    mirrors notebooks/chat_sessions.py's own _find_src_dir() convention, so this works whether
    the caller's CWD is the repo root, notebooks/, or somewhere else under the repo."""
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / "intents"
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(
        f"Couldn't locate an intents/ directory from the current working directory "
        f"({Path.cwd()}); pass intents_dir= explicitly."
    )


def load_intents(intents_dir: Union[str, Path, None] = None) -> List[Dict]:
    """Load every *.json file under intents_dir (default: _default_intents_dir()) into one flat
    list of intent dicts. Each file may hold either a single intent object or a JSON list of
    them -- every intents/*.json file shipped with this project is a list (e.g.
    diet_intents.json covers "take_food" and "take_drink" as two separate entries). Each
    returned dict is tagged with its source filename as "_source_file", for diagnostics only --
    nothing in this module reads that field back.
    """
    directory = Path(intents_dir) if intents_dir else _default_intents_dir()
    intents = []
    for path in sorted(directory.glob("*.json")):
        with open(path) as f:
            data = json.load(f)
        entries = data if isinstance(data, list) else [data]
        for entry in entries:
            intents.append({**entry, "_source_file": path.name})
    return intents


def _normalize(name: str) -> str:
    """Fold case and "-"/" " vs "_" spelling differences so e.g. "physical condition" (a raw
    data_type.ActivityType value) matches an intent's "physical_condition" (how it actually
    appears in the graph's own rdf:type -- see notebooks/chat_sessions.DEFAULT_GAP_ACTIVITY_TYPES's
    own comment on this space-to-underscore conversion). Does NOT paper over a genuine spelling
    mismatch between an intent file and the real data_type.ActivityType value it means to cover
    -- fix those in the intent file itself if a real activity silently never matches any intent.
    """
    return re.sub(r"[\s_-]+", "_", (name or "").strip().lower())


def _label_matches(activity_label: str, declared_label: str) -> bool:
    """Loose "same or similar phrase" check between an activity's own label/phrase and one of an
    intent's declared "activity_labels" entries (see find_intent()) -- exact match after
    normalizing case and whitespace, one containing the other (e.g. "tingling" ~ "tingling in my
    feet"), or the two sharing a word longer than 3 characters. Mirrors
    kg_gap_finder._labels_similar()'s own heuristic (duplicated rather than imported -- see this
    module's "Reading a subject's own current triples/types" comment on why)."""
    a, b = (activity_label or "").strip().lower(), (declared_label or "").strip().lower()
    if not a or not b:
        return False
    if a == b or a in b or b in a:
        return True
    a_words = {w for w in re.findall(r"\w+", a) if len(w) > 3}
    b_words = {w for w in re.findall(r"\w+", b) if len(w) > 3}
    return bool(a_words & b_words)


def find_intent_for_activity_type(activity_type: Optional[str], intents: List[Dict]) -> Optional[Dict]:
    """The first intent in `intents` (see load_intents()) whose "activity_types" list matches
    `activity_type`, or None if none do. A plain type-only lookup -- use find_intent() instead
    when several intents may share the same activity_type and need disambiguating by the
    activity's own label (see there)."""
    if not activity_type:
        return None
    target = _normalize(activity_type)
    for intent in intents:
        if target in {_normalize(t) for t in intent.get("activity_types", [])}:
            return intent
    return None


def find_intent(activity_type: Optional[str], intents: List[Dict], activity_label: Optional[str] = None) -> Optional[Dict]:
    """Like find_intent_for_activity_type(), but also disambiguates between several intents that
    share the same "activity_types" entry using each candidate's own optional "activity_labels"
    list (see load_intents()) -- needed because data_type.ActivityType has only one generic
    "symptom" type covering many distinct real symptoms (a headache, blurry vision, ...), which
    only show up as an activity's own LABEL/phrase, never as its type (see
    notebooks/chat_sessions.KgChatSession's `_subject_labels`, populated from
    `extraction.activity.value`). This project's own symptom_intents.json is exactly that case:
    four separate intents all typed "symptom", each naming the one specific symptom phrase (e.g.
    "headache") its own "activity_labels" covers.

    Precedence among the intents whose "activity_types" matches `activity_type`:
      1. A candidate that declares "activity_labels" AND one of them matches `activity_label`
         (via _label_matches()) wins.
      2. Otherwise, the first candidate with NO "activity_labels" of its own (a plain,
         type-only intent, e.g. diet/condition/exercise/medication's) is used -- this is exactly
         find_intent_for_activity_type()'s own behaviour, so an intent that never uses
         "activity_labels" at all is unaffected by this function.
      3. If every candidate for this activity_type declares "activity_labels" and none of them
         matched `activity_label` (including when `activity_label` is None/empty) -- e.g. a
         symptom whose phrase isn't one of the ones covered -- returns None, the same "no intent
         match -> fall back to the LLM response" case as no candidates at all. This is
         deliberate: guessing one of the label-specific intents anyway would silently apply the
         wrong intent's requirements (e.g. asking about a "body_part" location that doesn't
         apply to whatever the actual symptom turns out to be).
    """
    if not activity_type:
        return None
    target = _normalize(activity_type)
    candidates = [i for i in intents if target in {_normalize(t) for t in i.get("activity_types", [])}]
    if activity_label:
        for intent in candidates:
            if any(_label_matches(activity_label, declared) for declared in (intent.get("activity_labels") or [])):
                return intent
    unlabeled = [i for i in candidates if not i.get("activity_labels")]
    return unlabeled[0] if unlabeled else None


# --------------------------------------------------------------------------- #
# Reading a subject's own current triples/types
# --------------------------------------------------------------------------- #
#
# Small, local queries rather than reaching into kg_gap_finder's underscore-prefixed helpers
# (_triples_of()/_types_of()) -- this module's checks are simple enough not to need their
# multi-call memoization caches, and kg_gap_finder.py's own module comments (e.g.
# _import_data_type()) already establish "duplicate a small helper rather than couple two flat
# modules together" as this project's own convention.

def _own_triples(graph, uri: str) -> List[Dict]:
    """Every (predicate, object) triple for `uri`, each tagged with whether the object is an
    IRI. Doubles as the "subject_triples" a kg_gap_finder.py-shaped gap row carries (stripped of
    the "object_is_iri" flag -- see _make_gap()), so
    prompts.response_processor.PromptProcessor._label_for_subject() can find a real rdfs:label
    for the question instead of falling back to the bare URI."""
    rows = kg_gap_finder.run_query(graph, f"SELECT ?p ?o (isIRI(?o) AS ?oIsIRI) WHERE {{ <{uri}> ?p ?o . }}")
    return [{"predicate": r["p"], "object": r["o"], "object_is_iri": r["oIsIRI"] == "true"} for r in rows]


def _object_types(graph, uri: str) -> List[str]:
    """rdf:type(s) of `uri` -- used both to check a role filler's object type against an
    intent's expected list, and (in _infer_activity_type()) to read a subject's own type."""
    rows = kg_gap_finder.run_query(graph, f"SELECT DISTINCT ?type WHERE {{ <{uri}> a ?type . FILTER(isIRI(?type)) }}")
    return [r["type"] for r in rows]


def _local_name(uri: str) -> str:
    if not uri or "://" not in uri:
        return uri
    return uri.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _infer_activity_type(graph, subject_uri: str) -> Optional[str]:
    """`subject_uri`'s own activity type, read straight off its rdf:type triples: the one local
    name that isn't the generic "activity" wrapper type every activity also carries (see
    events_to_capsules.get_triples_with_types_and_activity_id()'s own `activity_type` list) --
    or None if it has no more specific type than that (or no rdf:type at all). Only needed when
    a caller (e.g. find_gap()'s CLI) doesn't already know the activity's type some other way --
    notebooks/chat_sessions.KgChatSession tracks it directly in self._subject_types."""
    types = {_local_name(t) for t in _object_types(graph, subject_uri)}
    types.discard("activity")
    return next(iter(types), None)


def _activity_label_from_triples(subject_uri: str, own_triples: List[Dict]) -> str:
    """`subject_uri`'s own real rdfs:label (e.g. "big lunch", "cycling"), read straight off the
    `own_triples` next_intent_gap() already fetched -- same "skip the bare activity_id fallback
    label" filtering as _infer_activity_label()/prompts.response_processor.PromptProcessor.
    _label_for_subject(), just reusing triples the caller already has instead of a fresh query.
    Used to fill in a `question_template`'s "{activity}" placeholder (see
    _render_question_template()). Falls back to the subject's own local name if it has no real
    label at all."""
    subject_id = _local_name(subject_uri)
    labels = [
        t["object"] for t in own_triples
        if t["predicate"] == kg_gap_finder.LABEL_PREDICATE and t["object"] != subject_id
    ]
    return labels[0] if labels else subject_id


def _patient_label(graph, own_triples: List[Dict], label_cache: Dict[str, str]) -> Optional[str]:
    """The current `patient` role filler's own human-ish label (e.g. "pizza"), once it's already
    known -- used to fill in a secondary qualification question's "{patient}" placeholder (e.g.
    "how much {patient} did you have?"). Only meaningful once a patient_type check has already
    passed (see next_intent_gap()'s priority order: qualification is only ever asked about after
    patient_type is satisfied), so there's always a `patient` triple to read by the time this is
    called for real. None if `own_triples` has no `patient` triple at all."""
    patient_uri = kg_gap_finder.NAMESPACE + "patient"
    for t in own_triples:
        if t["predicate"] == patient_uri:
            return kg_gap_finder._label_for_value(graph, t["object"], t["object_is_iri"], label_cache)
    return None


def _render_question_template(template: Optional[str], activity_label: str, patient_label: Optional[str] = None,
                               patient_type: Optional[List[str]] = None) -> Optional[str]:
    """Fill in a hand-authored intents/*.json question template's "{activity}"/"{patient}"/
    "{patient_type}" placeholders (see this module's docstring and next_intent_gap()'s
    "question_source" handling), or None if `template` itself is falsy (no template given for
    this requirement -- the caller falls back to the plain triple-based question, exactly as if
    this feature didn't exist for it).

    - "{patient}" falls back to `activity_label` itself if `patient_label` wasn't resolved
      (should not normally happen -- see _patient_label()) so a template is never sent to the
      LLM with a raw, unfilled placeholder still in it.
    - "{patient_type}" is the intent's own declared "patient_type" list (e.g. ["body_function"]),
      human-ish-ified (underscores turned to spaces) and joined with " or " for several -- the
      EXPECTED category, for a question asked before the actual value is known yet (e.g.
      measurement_intents.json's "Did you measure {patient_type} lately?"), as opposed to
      "{patient}" (the actual value, once known). Falls back to `activity_label` if
      `patient_type` is empty/None.
    """
    if not template:
        return None
    rendered = template.replace("{activity}", activity_label)
    if "{patient}" in rendered:
        rendered = rendered.replace("{patient}", patient_label or activity_label)
    if "{patient_type}" in rendered:
        type_label = " or ".join(t.replace("_", " ") for t in (patient_type or [])) or activity_label
        rendered = rendered.replace("{patient_type}", type_label)
    return rendered


def _question_for_aspect(question_source, aspect: str) -> Optional[str]:
    """Resolve one aspect's (e.g. "duration", "degree", "quantity") own question template out of
    an intent's "qualification_question" field (see this module's docstring): either a single
    template string that applies no matter which aspect is being asked about (e.g. exercise's
    single-aspect "duration"), or a {aspect_name: template} mapping for intents whose
    "activity_qualification" lists more than one aspect, each needing its own distinct wording
    (e.g. condition's "duration"/"degree"). None if `question_source` is falsy, or a dict lookup
    for `aspect` misses (that particular aspect has no template of its own)."""
    if isinstance(question_source, dict):
        return question_source.get(aspect)
    return question_source


def _infer_activity_label(graph, subject_uri: str) -> Optional[str]:
    """`subject_uri`'s own rdfs:label, if it has a REAL one -- i.e. not just its bare
    activity_id used as a label fallback for a later, phrase-less reference to an activity
    introduced earlier (see events_to_capsules.get_triples_with_types_and_activity_id()'s own
    comment on this fallback, and prompts.response_processor.PromptProcessor._label_for_subject(),
    which does the same filtering for a kg_gap_finder-style gap row). Used by find_gap() to
    disambiguate between several intents sharing one activity_type (see find_intent()) when a
    caller doesn't already know the activity's label some other way -- e.g.
    notebooks/chat_sessions.KgChatSession, which tracks it directly in self._subject_labels.
    Returns None if `subject_uri` has no rdfs:label at all, or only that id fallback one."""
    subject_id = _local_name(subject_uri)
    rows = kg_gap_finder.run_query(
        graph, f"SELECT ?label WHERE {{ <{subject_uri}> <{kg_gap_finder.LABEL_PREDICATE}> ?label }}"
    )
    labels = [r["label"] for r in rows if r["label"] != subject_id]
    return labels[0] if labels else None


# --------------------------------------------------------------------------- #
# Building kg_gap_finder.py-shaped gap rows
# --------------------------------------------------------------------------- #

def _known_context(graph, own_triples: List[Dict]) -> Dict[str, str]:
    """Facts about this SAME activity that are already known -- currently just who did it and
    when -- so a gap row can carry them alongside whatever's actually missing (see
    prompts.response_processor._format_known_context()). Without this, the LLM turning a gap row
    into a question is given nothing but "<subject label>, <predicate>, <type>" (e.g. "big lunch,
    patient, food") and has no material to build a natural question like "What did you have for
    lunch yesterday?" from -- it can only paraphrase the bare triple, which is also how the raw
    predicate name (e.g. "patient") ends up leaking into the question when a weaker LLM backend
    doesn't fully honour get_instruct_for_subject_gap()'s "don't use the words agent, patient or
    experiencer" instruction on its own.

    - "agent": always the literal "you" whenever ANY agent-like triple (kg_gap_finder.
      AGENT_ROLE_PREDICATES) is present -- in this single-human chat the agent is always the
      person speaking (see kg_gap_finder.AGENT_ROLE_PREDICATES' own module comment), so there's
      nothing to resolve.
    - "time": the first TIME_PREDICATE_VARIANTS triple's own object, resolved to a human-ish
      label (e.g. "yesterday") via kg_gap_finder._label_for_value().

    Only these two are populated (for now) -- location/qualification aren't, since the
    patient/date checks that actually need this context (see next_intent_gap()) only ever want
    to weave in who and when, not e.g. an already-known location.
    """
    context: Dict[str, str] = {}
    label_cache: Dict[str, str] = {}
    time_predicates = {kg_gap_finder.NAMESPACE + "time/" + variant for variant in kg_gap_finder.TIME_PREDICATE_VARIANTS}
    for t in own_triples:
        if "agent" not in context and t["predicate"] in kg_gap_finder.AGENT_ROLE_PREDICATES:
            context["agent"] = "you"
        elif "time" not in context and t["predicate"] in time_predicates:
            context["time"] = kg_gap_finder._label_for_value(graph, t["object"], t["object_is_iri"], label_cache)
    return context


def _make_gap(cls_label: str, predicate: str, subject_uri: str, own_triples: List[Dict],
              object_type: Optional[str] = None, known_context: Optional[Dict[str, str]] = None,
              question_template: Optional[str] = None, fill_role: Optional[str] = None,
              fill_role_type: Optional[str] = None) -> Dict:
    """One kg_gap_finder.py-shaped gap row -- see its find_predicate_gaps()/
    find_predicate_object_gaps() docstrings for the shape this mirrors. `predicate` may be a
    real RDF predicate URI (e.g. kg_gap_finder.NAMESPACE + "patient") or a plain display string
    with no "://" in it (e.g. "quantity", "date") for a check with no single real predicate of
    its own to name -- prompts.response_processor.local_name() passes a value with no "://"
    straight through unchanged, so a plain string reads naturally in the resulting question
    ("pizza, quantity, something" -> "how much pizza did you have?") without inventing a fake
    URI. `peer_examples` is always [] here: unlike kg_gap_finder's peer-vote-based gaps, an
    intent's expectation carries no examples of what other instances actually had, since it
    isn't computed by comparing against any. `known_context` (see _known_context()) is carried
    through as-is, defaulting to {}. `question_template` (see _render_question_template()) is
    the intent author's own hand-written example question for this requirement, already filled
    in with {activity}/{patient} -- carried through as-is, None if the intent gave none, so
    prompts.response_processor.get_prompt_for_kg_gap() can have the LLM paraphrase THAT instead
    of inventing a question from the bare subject/predicate/type triple.

    `fill_role` (a plain ROLE_FIELDS_WITH_TYPE-style role name, e.g. "patient"/"location"/
    "qualification") is set ONLY when there's a single, unambiguous real RDF predicate this
    requirement can be filled in on directly from the human's own next reply -- see
    notebooks/chat_sessions.KgChatSession._reply_from_gaps()/_handle_intent_answer_reply(),
    which arms a "pending intent answer" for exactly this case: instead of relying on the
    general-purpose SRL extractor to coreference a short follow-up reply ("yoghurt with fresh
    fruit", "30 minutes") back to the SAME activity/role the question was actually about --
    which it doesn't reliably do, letting the identical gap resurface on a freshly-minted,
    unrelated subject next turn -- the reply is attached to `subject_uri`'s own `fill_role`
    directly. Left None (the default, from every caller that doesn't pass it) for a requirement
    with no single safe predicate to name this way -- see _predicate_presence_gap()'s own
    comment on why the multi-variant "date" check is deliberately one of these. `fill_role_type`
    is the single RoleType-ish value (e.g. "food") to tag the pushed object with when `fill_role`
    is a typed role like "patient"/"location" -- None for an untyped one like "qualification".
    """
    gap = {
        "class": cls_label,
        "predicate": predicate,
        "subject": subject_uri,
        "subject_triples": [{"predicate": t["predicate"], "object": t["object"]} for t in own_triples],
        "object_triples": [],
        "peer_examples": [],
        "known_context": known_context or {},
        "question_template": question_template,
        "fill_role": fill_role,
        "fill_role_type": fill_role_type,
    }
    if object_type is not None:
        gap["object_type"] = object_type
    return gap


def _predicate_object_type_gap(graph, subject_uri: str, cls_label: str, own_triples: List[Dict],
                                predicate_local: str, expected_types: List[str],
                                known_context: Optional[Dict[str, str]] = None,
                                question_template: Optional[str] = None) -> Optional[Dict]:
    """None if `subject_uri` already has a `predicate_local` (under kg_gap_finder.NAMESPACE)
    triple whose object's rdf:type includes one of `expected_types` (matched by local name,
    normalized) -- otherwise a "predicate_object_type" gap row for it. `predicate_local` itself
    (e.g. "patient", "location") is always a single, unambiguous real role name, and the first of
    `expected_types` (if any) a reasonable single type to tag a directly-pushed answer with -- so
    this always sets `fill_role`/`fill_role_type` (see _make_gap()) unconditionally."""
    predicate_uri = kg_gap_finder.NAMESPACE + predicate_local
    normalized_expected = {_normalize(t) for t in expected_types}
    for t in own_triples:
        if t["predicate"] != predicate_uri or not t["object_is_iri"]:
            continue
        object_type_names = {_normalize(_local_name(ot)) for ot in _object_types(graph, t["object"])}
        if normalized_expected & object_type_names:
            return None
    return _make_gap(cls_label, predicate_uri, subject_uri, own_triples, object_type=" or ".join(expected_types),
                      known_context=known_context, question_template=question_template,
                      fill_role=predicate_local, fill_role_type=next(iter(expected_types), None))


def _predicate_presence_gap(subject_uri: str, cls_label: str, own_triples: List[Dict],
                             predicate_uris: Union[str, List[str]], display_name: str,
                             known_context: Optional[Dict[str, str]] = None,
                             question_template: Optional[str] = None) -> Optional[Dict]:
    """None if `subject_uri` already has ANY triple whose predicate is in `predicate_uris` (a
    single URI, or several -- e.g. kg_gap_finder.TIME_PREDICATE_VARIANTS' four "time/..." URIs,
    any one of which counts as "has a date") -- otherwise a "predicate" gap row using
    `display_name` (not the real predicate URI) as the gap's own "predicate" field, so the
    resulting question reads naturally (see _make_gap()).

    `fill_role` (see _make_gap()) is only ever set when `predicate_uris` names exactly ONE real
    predicate -- e.g. a secondary_objectives entry's generic single-aspect check. The "date"
    check (`predicate_uris` = kg_gap_finder.TIME_PREDICATE_VARIANTS' four URIs) deliberately
    leaves it unset: a direct answer like "yesterday" or "for an hour" could resolve to any of
    dateTime/rangeTime/recurringTime/vagueTime depending on its own phrasing (see
    events_to_capsules.get_triples_with_types_and_activity_id()'s own time_resolved handling),
    which isn't something to guess blindly from a bare reply string -- so a "date" gap is never
    auto-filled this way, only ever answered through the ordinary SRL extractor, same as before
    this feature existed."""
    predicate_set = {predicate_uris} if isinstance(predicate_uris, str) else set(predicate_uris)
    if any(t["predicate"] in predicate_set for t in own_triples):
        return None
    fill_role = _local_name(next(iter(predicate_set))) if len(predicate_set) == 1 else None
    return _make_gap(cls_label, display_name, subject_uri, own_triples, known_context=known_context,
                      question_template=question_template, fill_role=fill_role)


def _qualification_gap(subject_uri: str, cls_label: str, own_triples: List[Dict],
                        aspects: List[str], known_context: Optional[Dict[str, str]] = None,
                        question_source=None, activity_label: str = "",
                        patient_label: Optional[str] = None,
                        patient_type: Optional[List[str]] = None) -> Optional[Dict]:
    """None if `subject_uri` already has at least len(aspects) `qualification` triples (under
    kg_gap_finder.NAMESPACE) -- otherwise a "predicate" gap row for aspects[<current count>],
    the next aspect (e.g. "duration", then "degree", then a secondary "quantity") in the order
    `aspects` declares them.

    There is no way to tell WHICH aspect an existing qualification value answers -- the graph
    model attaches every qualification as a plain, untyped value directly on the activity itself
    (see events_to_capsules.get_triples_with_types_and_activity_id()'s ROLE_FIELDS_WITH_TYPE;
    there is no per-aspect sub-predicate) -- so this simply assumes aspects get answered in the
    order they're asked, which holds as long as the human's replies get annotated back onto this
    same activity, exactly what every other kg-gap follow-up already relies on (see
    notebooks/chat_sessions.KgChatSession's own class docstring on cross-turn coreference).

    `question_source` (an intent's "qualification_question" field -- see _question_for_aspect())
    is resolved for the SPECIFIC aspect this call is about to return a gap for
    (aspects[<count>]), then rendered (see _render_question_template()) with `activity_label`/
    `patient_label`/`patient_type` into the gap's own `question_template`.

    Always sets `fill_role="qualification"` (see _make_gap()) -- whichever aspect is actually
    being asked about, the real underlying predicate to push a direct answer to is always the
    same untyped `n2mu:qualification` (there is no per-aspect predicate to begin with, per this
    docstring's own point above), so this is never ambiguous the way "date" is.
    """
    qualification_uri = kg_gap_finder.NAMESPACE + "qualification"
    count = sum(1 for t in own_triples if t["predicate"] == qualification_uri)
    if count >= len(aspects):
        return None
    aspect = aspects[count]
    template = _render_question_template(_question_for_aspect(question_source, aspect), activity_label,
                                          patient_label=patient_label, patient_type=patient_type)
    return _make_gap(cls_label, aspect, subject_uri, own_triples, known_context=known_context,
                      question_template=template, fill_role="qualification")


# --------------------------------------------------------------------------- #
# Putting it together: the next thing this intent still wants to know
# --------------------------------------------------------------------------- #

def next_intent_gap(graph, subject_uri: str, intent: Dict, activity_type: Optional[str] = None) -> Optional[Tuple[Dict, str]]:
    """The single next requirement `intent` (see load_intents()/find_intent_for_activity_type())
    says is still missing for `subject_uri`, as a (gap, kind) pair in the same shape
    kg_gap_finder.py's find_predicate_gaps()/find_predicate_object_gaps() rows use -- so
    prompts.response_processor.PromptProcessor.get_prompt_for_kg_gap(gap, kind, human=...) turns
    it straight into a follow-up question, exactly like a real kg_gap_finder gap. Returns None
    once every requirement `intent` declares is met (nothing left to ask -> caller falls back to
    whatever it would otherwise do, e.g. the default LLM reply).

    Checked in a fixed priority order, stopping at (and returning) the FIRST one not yet met --
    "the data elements for the intents must first be met before moving on with the
    conversation":

      1. "patient_type" (list) -- the activity must have a `patient` whose object is one of
         these RoleType-style local names (e.g. ["food"], ["drink"]) -- what was eaten/drunk/
         taken.
      2. "activity_qualification" (list) -- one `qualification` value per named aspect, in
         order (e.g. ["duration", "degree"]) -- see _qualification_gap().
      3. "activity_date" (== "date") -- any `time` value at all (kg_gap_finder.TIME_PREDICATE_VARIANTS).
      4. "secondary_objectives" (dict, only reached once 1-3 are all satisfied) -- each
         {key: value} entry, tried in the dict's own order. Only the part of `key` AFTER its
         last "_" (its "aspect") is actually used -- the current graph model attaches every role
         directly to the activity itself, never to a role-filler object of its own (see
         events_to_capsules.py), so there is no separate "the patient's own triples" to check;
         the "patient_"/"activity_" prefix on a key like "patient_qualification" is kept in the
         intent files purely for readability, not treated as a distinct target:
           - "*_qualification" -- one more qualification aspect, named by `value` (e.g.
             "patient_qualification": "quantity" asks how much, after any PRIMARY
             activity_qualification aspects are already filled).
           - "*_location" -- like patient_type, but for the `location` role, expected object
             type `value` (e.g. "place", "body_part").
           - "*_date" -- same check as top-level "activity_date" (used by intents that only want
             a date as a secondary follow-up, not a precondition -- see this project's own
             symptom_intents.json/condition_intents.json/excercise_intents.json).
           - any other aspect -- a generic presence check on that aspect's own predicate
             (kg_gap_finder.NAMESPACE + aspect), displayed as `value`.
         Any key ENDING in "_question" (e.g. "location_question", "date_question",
         "qualification_question") is metadata, not a requirement of its own -- see
         "Example questions" below -- and is skipped by this loop.

    Example questions: an intent may give each requirement above its own hand-authored example
    question in intents/*.json, so prompts.response_processor.get_prompt_for_kg_gap() can have
    the LLM paraphrase THAT (see get_instruct_for_templated_gap()) instead of inventing a
    question from the bare subject/predicate/type triple -- this is what actually fixes a weak
    LLM backend leaking a raw role name (e.g. "patient") into the question:
      - "patient_question" (sibling of "patient_type") -- "{activity}" filled in with the
        activity's own label (e.g. "lunch"); may also use "{patient_type}" (the intent's
        EXPECTED category, e.g. "body function" -- see _render_question_template()) for a
        question asked before the actual value is known yet, e.g. measurement_intents.json's
        "Did you measure {patient_type} lately?".
      - "qualification_question" (sibling of "activity_qualification") -- either one template
        string that applies to every aspect, or a {aspect_name: template} mapping keyed by the
        SPECIFIC aspect (e.g. "duration"/"degree") being asked about, when each needs its own
        wording (see _question_for_aspect()). "{activity}"/"{patient_type}" filled in as above.
      - "date_question" (sibling of "activity_date") -- "{activity}" filled in as above.
      - within "secondary_objectives", each requirement's own "{aspect}_question" sibling key
        (e.g. "location_question" beside "activity_location", "qualification_question" beside
        "*_qualification" -- this one may also use "{patient}", filled in with the ALREADY-KNOWN
        patient's own label, e.g. "how much {patient} did you have?" -> "how much pizza did you
        have?", since qualification is only ever asked once patient_type is already satisfied).
    Any requirement without a matching template falls back to the plain triple-based question,
    exactly as before this feature existed.

    `activity_type` (if given -- notebooks/chat_sessions.KgChatSession already tracks it in
    self._subject_types) is used only for the gap row's cosmetic "class" field; defaults to
    intent's own "activity_types" (joined with " | ") when not given.
    """
    cls_label = activity_type or " | ".join(intent.get("activity_types") or [])
    own_triples = _own_triples(graph, subject_uri)
    known_context = _known_context(graph, own_triples)
    activity_label = _activity_label_from_triples(subject_uri, own_triples)
    label_cache: Dict[str, str] = {}

    patient_type = intent.get("patient_type")
    if patient_type:
        template = _render_question_template(intent.get("patient_question"), activity_label, patient_type=patient_type)
        gap = _predicate_object_type_gap(graph, subject_uri, cls_label, own_triples, "patient", patient_type,
                                          known_context=known_context, question_template=template)
        if gap:
            return gap, "predicate_object_type"

    primary_aspects = list(intent.get("activity_qualification") or [])
    if primary_aspects:
        gap = _qualification_gap(subject_uri, cls_label, own_triples, primary_aspects, known_context=known_context,
                                  question_source=intent.get("qualification_question"), activity_label=activity_label,
                                  patient_type=patient_type)
        if gap:
            return gap, "predicate"

    time_predicates = [kg_gap_finder.NAMESPACE + "time/" + variant for variant in kg_gap_finder.TIME_PREDICATE_VARIANTS]
    if intent.get("activity_date"):
        template = _render_question_template(intent.get("date_question"), activity_label, patient_type=patient_type)
        gap = _predicate_presence_gap(subject_uri, cls_label, own_triples, time_predicates, "date",
                                       known_context=known_context, question_template=template)
        if gap:
            return gap, "predicate"

    secondary = intent.get("secondary_objectives") or {}
    for key, value in secondary.items():
        if key.endswith("_question"):
            continue
        aspect = key.rsplit("_", 1)[-1]
        if aspect == "qualification":
            patient_label = _patient_label(graph, own_triples, label_cache)
            gap = _qualification_gap(subject_uri, cls_label, own_triples, primary_aspects + [value],
                                      known_context=known_context, question_source=secondary.get("qualification_question"),
                                      activity_label=activity_label, patient_label=patient_label, patient_type=patient_type)
            kind = "predicate"
        elif aspect == "location":
            template = _render_question_template(secondary.get("location_question"), activity_label, patient_type=patient_type)
            gap = _predicate_object_type_gap(graph, subject_uri, cls_label, own_triples, "location", [value],
                                              known_context=known_context, question_template=template)
            kind = "predicate_object_type"
        elif aspect == "date":
            template = _render_question_template(secondary.get("date_question"), activity_label, patient_type=patient_type)
            gap = _predicate_presence_gap(subject_uri, cls_label, own_triples, time_predicates, "date",
                                           known_context=known_context, question_template=template)
            kind = "predicate"
        else:
            template = _render_question_template(secondary.get(f"{aspect}_question"), activity_label, patient_type=patient_type)
            gap = _predicate_presence_gap(subject_uri, cls_label, own_triples, kg_gap_finder.NAMESPACE + aspect, value,
                                           known_context=known_context, question_template=template)
            kind = "predicate"
        if gap:
            return gap, kind

    return None


# --------------------------------------------------------------------------- #
# Convenience entry point (e.g. for notebooks, no argparse needed) + CLI
# --------------------------------------------------------------------------- #

def find_gap(
    endpoint: str = None,
    file: str = None,
    fmt: str = None,
    graph=None,
    subject_uri: str = None,
    activity_type: str = None,
    activity_label: str = None,
    intents: List[Dict] = None,
    intents_dir: Union[str, Path, None] = None,
) -> Optional[Tuple[Dict, str]]:
    """Convenience wrapper for calling this module directly from a notebook or the CLI below,
    without wiring up a graph/intents list by hand first: loads a graph (endpoint/file, or
    reuses an already-loaded `graph`), loads intents (or reuses an already-loaded `intents`
    list), infers `subject_uri`'s activity type and label from its own rdf:type/rdfs:label when
    `activity_type`/`activity_label` aren't given, and returns next_intent_gap() for the matching
    intent (see find_intent()) -- or None if no intent matches at all, or nothing about it is
    missing.
    """
    if graph is None:
        if endpoint:
            graph = kg_gap_finder.load_graph_from_endpoint(endpoint)
        elif file:
            graph = kg_gap_finder.load_graph_from_file(file, fmt)
        else:
            raise ValueError("Provide graph=, endpoint=, or file=")
    if not subject_uri:
        raise ValueError("subject_uri is required")
    if intents is None:
        intents = load_intents(intents_dir)
    if activity_type is None:
        activity_type = _infer_activity_type(graph, subject_uri)
    if activity_label is None:
        activity_label = _infer_activity_label(graph, subject_uri)
    intent = find_intent(activity_type, intents, activity_label=activity_label)
    if intent is None:
        return None
    return next_intent_gap(graph, subject_uri, intent, activity_type=activity_type)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Find the next intent-driven knowledge gap for one activity subject URI."
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--endpoint", help="SPARQL endpoint URL, e.g. http://localhost:7200/repositories/sandbox")
    src.add_argument("--file", help="Path to a local RDF file")
    parser.add_argument("--format", default=None, help="rdflib format for --file, e.g. turtle, trig, xml, json-ld")
    parser.add_argument("--subject-uri", required=True, help="The activity's own subject URI")
    parser.add_argument("--activity-type", default=None,
                         help="Activity type local name, e.g. take_food; inferred from the subject's own rdf:type if omitted")
    parser.add_argument("--activity-label", default=None,
                         help="Activity label/phrase, e.g. headache -- used to disambiguate intents that share one "
                              "activity_type (see find_intent()); inferred from the subject's own rdfs:label if omitted")
    parser.add_argument("--intents-dir", default=None, help="Directory of intent *.json files (default: intents/ at the project root)")
    args = parser.parse_args()

    result = find_gap(
        endpoint=args.endpoint, file=args.file, fmt=args.format, subject_uri=args.subject_uri,
        activity_type=args.activity_type, activity_label=args.activity_label, intents_dir=args.intents_dir,
    )
    if result is None:
        print("No intent match, or nothing missing -- fall back to the LLM.")
        return
    gap, kind = result
    print(f"kind: {kind}")
    print(json.dumps(gap, indent=2))


if __name__ == "__main__":
    main()
