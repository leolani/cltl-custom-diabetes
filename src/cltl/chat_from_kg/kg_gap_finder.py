#!/usr/bin/env python3
"""
kg_gap_finder.py
================

Queries an RDF knowledge graph for "knowledge gaps", in the spirit of the
Leolani Brain (leolani/cltl-knowledgerepresentation), which generates
"curiosity" thoughts based on gaps in the graph.

Background
----------
cltl-knowledgerepresentation builds an episodic RDF graph (backed by GraphDB,
queried with SPARQL) and derives "thoughts" from it, one category of which is
knowledge-graph gaps: cases where an entity is missing information that
similar entities in the graph do have. This script re-implements that idea
generically, so it works against:

  1. A live SPARQL endpoint (e.g. a running GraphDB "sandbox" repository, the
     default used by the cltl-knowledgerepresentation examples), or
  2. A local RDF file (turtle/xml/n3/json-ld) loaded with rdflib.

Gaps are reported at the TYPE level for the *pattern* (which class/predicate
combination is affected, and how many peers share the expectation), while
still naming the exact affected entity. Each finding describes a pattern
("instances of Person are missing hasName in 4/10 cases") and, for gap kinds
B, D and E, one row per affected subject naming exactly which entity has the
gap, rather than aggregating into a single group with a handful of examples.
This makes the report useful both for spotting systematic modeling gaps in
the ontology/data, and for drilling straight down to the specific entity to
fix.

Gap kinds B, D and E only consider predicates from data_type.SemanticRole
(events_from_chat/data_type.py) by default -- agent, patient, location, time,
... -- not the brain's own provenance/bookkeeping predicates (rdfs:label,
gaf:denotedIn/denotedBy, rdf:type, ...), since nobody can usefully be asked a
follow-up question about those. Pass predicate_filter=None (or --all-predicates
on the CLI) to fall back to considering every predicate.

It reports five kinds of gaps:

  A. Untyped entities        - resources used as subjects that have no
                                rdf:type, grouped by the *signature* of
                                predicates they do have (a proxy for "what
                                type they probably are").
  B. Predicate gaps          - for each rdf:type class, predicates that a
                                clear majority of instances have, but some
                                instances lack; one row per (class,
                                predicate, subject) with a peer-coverage
                                count (the "I know X about others like you,
                                but not about you" gap).
  C. Dangling references     - objects that are referenced (as the object of
                                a triple) but that have no outgoing triples of
                                their own in the graph, grouped by the
                                (referencing type, predicate) that points at
                                them, i.e. "Person hasFriend points at 12
                                references we know nothing further about".
  D. Predicate-object gaps,  - for each rdf:type class, (predicate, object-
     TYPE level                TYPE) facts that a majority of instances
                                share; one row per (class, predicate,
                                object-type, subject) with a peer-coverage
                                count (finer grained than B, coarser than E:
                                "most people like you have a hasPet of type
                                Dog, but a few of you specifically don't" -
                                any Dog satisfies the expectation, regardless
                                of which one).
  E. Predicate-object gaps,  - the same idea as D, but objects are compared
     INSTANCE level            by exact identity instead of by type: "most
                                people like you have hasPet=Fido, but a few
                                of you specifically don't" - a different dog
                                does NOT satisfy the expectation. Tends to
                                fragment into many individual-specific gaps
                                when a predicate points at many distinct
                                instances of the same type.

B, D and E rows also carry `peer_examples`: the actual values peers *do* have for the
missing predicate (or predicate-object-type pair), most frequent first, e.g. for a missing
`time`, peer_examples might be [{"value": "in the morning", "count": 3}, {"value": "7pm",
"count": 2}] -- so a follow-up question can suggest concrete, evidence-backed options instead of
asking blind. Capped per row by `peer_example_limit` (--peer-examples on the CLI).

B, D and E also treat PREDICATE_GROUPS as single facts: currently just the agent-like roles
(agent/agent_patient/participant/experiencer -- interchangeable per data_type.SemanticRole's own
docstring, never meant to be filled in side by side) collapse to one canonical "agent" predicate
everywhere counting/comparing happens (_canonicalize_predicate()), so an instance already filled
in via ANY one of them is never reported as missing another -- e.g. an instance with
`experiencer` set is not flagged as missing `agent`.

Usage
-----
    # Against a running GraphDB / any SPARQL endpoint:
    python kg_gap_finder.py --endpoint http://localhost:7200/repositories/sandbox

    # Against a local RDF file:
    python kg_gap_finder.py --file mybrain.trig --format trig

    # Only show gaps for a specific class (also covers its rdfs:subClassOf
    # descendants), and require >=70% majority:
    python kg_gap_finder.py --endpoint http://localhost:7200/repositories/sandbox \\
        --class-uri http://cltl.nl/leolani/n2mu/Person --threshold 0.7

    # Restrict predicate gaps (B, D and E) to one specific subject instead of
    # aggregating over every instance of its class:
    python kg_gap_finder.py --file mybrain.trig --subject-uri http://example.org/Piek

    # Show more example instances per gap group for A and C (default 5):
    python kg_gap_finder.py --file mybrain.trig --examples 10

    # Save the full report as JSON:
    python kg_gap_finder.py --file mybrain.trig --json report.json

    # Consider every predicate for B/D/E, not just data_type.SemanticRole's:
    python kg_gap_finder.py --file mybrain.trig --all-predicates

Requires: rdflib  (pip install rdflib)
For remote SPARQL endpoints rdflib's built-in SPARQLWrapper support is used
(pip install rdflib sparqlwrapper).
"""

import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from rdflib import Graph, URIRef
    from rdflib.plugins.stores import sparqlstore
except ImportError:
    print("This script needs rdflib. Install it with: pip install rdflib sparqlwrapper")
    sys.exit(1)

RDF_TYPE = "http://www.w3.org/1999/02/22-rdf-syntax-ns#type"
DEFAULT_EXAMPLE_LIMIT = 5

# Default cap on how many "peer_examples" (see _top_object_examples()) each B/D/E gap row carries
# -- the actual values peers have for the missing predicate, most-frequent first, so a follow-up
# question can be asked concretely ("most peers exercise in the morning or around 7pm") instead of
# blind ("when do you exercise?").
DEFAULT_PEER_EXAMPLE_LIMIT = 5


# --------------------------------------------------------------------------- #
# Semantic-role predicate filter
# --------------------------------------------------------------------------- #

# Namespace + role -> predicate URI mapping, mirroring how
# events_from_chat/events_to_capsules.py's get_triples_with_types_and_activity_id() turns an
# SRLAnnotation's roles into RDF triples: every SemanticRole except "time" becomes
# NAMESPACE + role.value directly; "time" is never asserted as a bare predicate of its own -- it
# always resolves into one of these four typed variants under NAMESPACE + "time/" instead.
NAMESPACE = "http://cltl.nl/leolani/n2mu/"
TIME_PREDICATE_VARIANTS = ("dateTime", "rangeTime", "recurringTime", "vagueTime")

# Sentinel default for the `predicate_filter` parameter of find_predicate_gaps() /
# find_predicate_object_gaps() / find_predicate_object_instances_gaps(): "use
# semantic_role_predicates()". Pass predicate_filter=None explicitly to disable filtering (the
# original behaviour -- every predicate is fair game, useful against a graph that isn't this
# project's own ontology).
DEFAULT_PREDICATE_FILTER = object()

_semantic_role_predicates_cache: Optional[Set[str]] = None


def _import_data_type():
    """Import events_from_chat/data_type.py. It's a flat, non-package module (like the rest of
    events_from_chat), so it's found either because it's already on sys.path/in sys.modules
    (e.g. notebooks/chat_sessions.py's _load_kg_dependencies() imports it before kg_gap_finder),
    or by falling back to kg_gap_finder.py's own sibling directory here -- so kg_gap_finder.py
    also works standalone (its own CLI, or imported directly without going through chat_sessions).
    """
    try:
        import data_type
        return data_type
    except ImportError:
        pass
    sibling = Path(__file__).resolve().parent.parent / "events_from_chat"
    if not (sibling / "data_type.py").is_file():
        raise ImportError(
            "Couldn't import data_type (needed for the semantic-role predicate filter) -- "
            f"expected to find it at {sibling / 'data_type.py'}. Add events_from_chat/ to "
            "sys.path yourself, or pass predicate_filter=None to disable the filter."
        )
    if str(sibling) not in sys.path:
        sys.path.insert(0, str(sibling))
    import data_type
    return data_type


def semantic_role_predicates() -> Set[str]:
    """The RDF predicate URIs that can hold a semantic role, per data_type.SemanticRole
    (events_from_chat/data_type.py) -- the default `predicate_filter` for find_predicate_gaps() /
    find_predicate_object_gaps() / find_predicate_object_instances_gaps(), so gaps are only
    reported for predicates a coach could actually ask a follow-up question about (agent,
    patient, location, time, ...), not the brain's own provenance/bookkeeping predicates
    (rdfs:label, gaf:denotedIn/denotedBy, rdf:type, ...).

    Read from data_type.py rather than duplicated here, so this can't drift out of sync with the
    SRL extraction prompt/Pydantic models -- see data_type.py's own module docstring. Cached
    after the first call.
    """
    global _semantic_role_predicates_cache
    if _semantic_role_predicates_cache is None:
        data_type = _import_data_type()
        predicates = set()
        for role in data_type.SemanticRole:
            if role == data_type.SemanticRole.time:
                predicates.update(NAMESPACE + "time/" + variant for variant in TIME_PREDICATE_VARIANTS)
            else:
                predicates.add(NAMESPACE + role.value)
        _semantic_role_predicates_cache = predicates
    return _semantic_role_predicates_cache


# --------------------------------------------------------------------------- #
# Graph loading
# --------------------------------------------------------------------------- #

def load_graph_from_endpoint(endpoint: str) -> Graph:
    """Connect read-only to a remote SPARQL endpoint (e.g. GraphDB)."""
    store = sparqlstore.SPARQLStore(endpoint)
    graph = Graph(store=store)
    return graph


def load_graph_from_file(path: str, fmt: str = None) -> Graph:
    graph = Graph()
    graph.parse(path, format=fmt)
    return graph


# Toggle for run_query()'s per-query logging (row count + elapsed time, printed to stdout as
# each query runs). On by default -- flip to False (e.g. `kg_gap_finder.LOG_QUERIES = False`) for
# quiet runs. _query_stats accumulates count/total_time since the last _reset_query_log() call
# (build_report() calls that at the start of every report, so its own summary line -- see there --
# reflects just that one report, not every query ever run in the process).
LOG_QUERIES = True
_query_stats = {"count": 0, "total_time": 0.0}


def _reset_query_log() -> None:
    _query_stats["count"] = 0
    _query_stats["total_time"] = 0.0


def run_query(graph: Graph, query: str) -> List[Dict]:
    """Run a SPARQL SELECT query and return rows as plain dicts of strings.

    Unbound variables (e.g. from an OPTIONAL clause) are kept as None rather
    than being stringified, so callers can distinguish "unbound" from the
    literal string "None".

    Logs the row count and elapsed wall-clock time (query + fetching all rows) to stdout for
    every call when LOG_QUERIES is True (the default) -- there can be many of these per gap
    lookup (see find_comparison_peers()'s module comment for why that used to be far more, and
    still is for a full, subject-less report), so this is the way to see where the time actually
    goes on a given graph.
    """
    start = time.perf_counter()
    results = graph.query(query)
    rows = []
    for row in results:
        rows.append({str(v): (str(row[v]) if row[v] is not None else None) for v in results.vars})
    elapsed = time.perf_counter() - start

    _query_stats["count"] += 1
    _query_stats["total_time"] += elapsed
    if LOG_QUERIES:
        one_line = " ".join(query.split())
        if len(one_line) > 100:
            one_line = one_line[:100] + "..."
        print(f"[kg_gap_finder] query #{_query_stats['count']}: {len(rows)} row(s) in {elapsed:.3f}s -- {one_line}")

    return rows


# --------------------------------------------------------------------------- #
# A. Untyped entities, grouped by predicate signature
# --------------------------------------------------------------------------- #

UNTYPED_QUERY = """
SELECT ?entity ?p WHERE {
    ?entity ?p ?o .
    FILTER NOT EXISTS { ?entity a ?type }
    FILTER(isIRI(?entity))
}
LIMIT %d
"""


def find_untyped_entities(
    graph: Graph, row_limit: int = 5000, example_limit: int = DEFAULT_EXAMPLE_LIMIT
) -> List[Dict]:
    """
    Find entities with no rdf:type, then group them by the *set* of
    predicates they're used with. Entities that share an identical predicate
    signature are almost certainly missing the same type declaration, so
    this surfaces "there's a whole class of untyped entity here" instead of
    a flat list of individual untyped resources.
    """
    rows = run_query(graph, UNTYPED_QUERY % row_limit)

    entity_predicates: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        entity_predicates[r["entity"]].add(r["p"])

    signature_groups: Dict[frozenset, List[str]] = defaultdict(list)
    for entity, preds in entity_predicates.items():
        signature_groups[frozenset(preds)].append(entity)

    results = []
    for sig, instances in signature_groups.items():
        instances_sorted = sorted(instances)
        results.append(
            {
                "predicate_signature": sorted(sig),
                "instance_count": len(instances_sorted),
                "example_entities": instances_sorted[:example_limit],
            }
        )
    results.sort(key=lambda g: -g["instance_count"])
    return results


# --------------------------------------------------------------------------- #
# B. Predicate gaps per class ("others like you have this, some of you don't")
# --------------------------------------------------------------------------- #

CLASSES_QUERY = """
SELECT DISTINCT ?type WHERE { ?s a ?type . FILTER(isIRI(?type)) }
"""

INSTANCES_QUERY = """
SELECT DISTINCT ?instance WHERE { ?instance a <%s> . FILTER(isIRI(?instance)) }
LIMIT %d
"""

# Like INSTANCES_QUERY, but also matches instances of (transitive) rdfs:subClassOf
# descendants of <%s>, not just exact-type instances. Used when a class_filter is
# given explicitly, so e.g. --class-uri .../Person also picks up .../Student,
# .../Employee, etc. The subClassOf* path includes the zero-length case, so
# exact-type instances of <%s> itself are still matched too.
SUBCLASS_INSTANCES_QUERY = """
SELECT DISTINCT ?instance WHERE {
    ?instance a ?actualType .
    ?actualType <http://www.w3.org/2000/01/rdf-schema#subClassOf>* <%s> .
    FILTER(isIRI(?instance))
}
LIMIT %d
"""

TYPES_FOR_SUBJECT_QUERY = """
SELECT DISTINCT ?type WHERE { <%s> a ?type . FILTER(isIRI(?type)) }
"""

PREDICATE_OBJECT_FOR_INSTANCE_QUERY = """
SELECT DISTINCT ?p ?o (isIRI(?o) AS ?oIsIRI) WHERE { <%s> ?p ?o . FILTER(!isBlank(?o)) }
"""

ALL_TRIPLES_FOR_SUBJECT_QUERY = """
SELECT ?p ?o (isIRI(?o) AS ?oIsIRI) WHERE { <%s> ?p ?o . }
"""


def _triples_of(graph: Graph, uri: str, cache: Dict[str, List[Dict[str, str]]]) -> List[Dict[str, str]]:
    """
    Every (predicate, object) triple for a URI - i.e. everything already
    known about it, rdf:type included - memoized per call to a find_*
    function. Used to attach full context to a gap row: alongside the one
    fact a subject is missing, show what IS on record for it.

    Each returned dict also carries an internal "_object_is_iri" flag (only
    an IRI can meaningfully be the *subject* of further triples, so this is
    what _object_triples_of() uses to decide which objects to follow one
    more hop) - strip it before exposing rows to callers outside this
    module, e.g. with {k: v for k, v in t.items() if not k.startswith("_")}.
    """
    if uri not in cache:
        cache[uri] = sorted(
            (
                {"predicate": r["p"], "object": r["o"], "_object_is_iri": r["oIsIRI"] == "true"}
                for r in run_query(graph, ALL_TRIPLES_FOR_SUBJECT_QUERY % uri)
            ),
            key=lambda t: (t["predicate"], t["object"]),
        )
    return cache[uri]


def _strip_internal(triples: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Drop the internal "_object_is_iri" bookkeeping field before exposing triples in a report row."""
    return [{"predicate": t["predicate"], "object": t["object"]} for t in triples]


def _types_of(graph: Graph, uri: str, cache: Dict[str, List[str]]) -> List[str]:
    """rdf:type(s) of a URI, memoized per call to a find_* function."""
    if uri not in cache:
        cache[uri] = [r["type"] for r in run_query(graph, TYPES_FOR_SUBJECT_QUERY % uri)]
    return cache[uri]


def _object_triples_of(
    graph: Graph,
    subject_triples: List[Dict[str, str]],
    triples_cache: Dict[str, List[Dict[str, str]]],
    types_cache: Dict[str, List[str]],
) -> List[Dict[str, str]]:
    """
    One more hop out: for every object in `subject_triples` that is itself
    an IRI AND an instance (has its own rdf:type - so a data entity, not a
    class/schema/predicate URI referenced only as a value), fetch its own
    outgoing triples - i.e. "what do we know about the things this subject
    points at". Each object is only expanded once even if several
    subject_triples share it. Literal objects, and untyped IRIs (e.g. class
    URIs used as an rdf:type value), are skipped.
    """
    result = []
    seen_objects = set()
    for t in subject_triples:
        o = t["object"]
        if not t["_object_is_iri"] or o in seen_objects:
            continue
        seen_objects.add(o)
        if not _types_of(graph, o, types_cache):
            continue  # not an instance - nothing to follow
        for ot in _triples_of(graph, o, triples_cache):
            result.append({"subject": o, "predicate": ot["predicate"], "object": ot["object"]})
    result.sort(key=lambda t: (t["subject"], t["predicate"], t["object"]))
    return result


# --------------------------------------------------------------------------- #
# Fast, targeted peer selection for a single subject
#
# find_predicate_gaps() et al., when given a subject_filter but no class_filter, used to still
# scan every instance of every rdf:type class in the graph to compute peer statistics, then keep
# only the one row for subject_filter -- correct, but far too slow on a graph of any real size,
# since it does that full scan on every single-subject gap lookup (e.g. once per KgChatSession
# turn). find_comparison_peers() replaces that with a small, targeted peer group instead.
# --------------------------------------------------------------------------- #

# Types so generic that nearly every resource in the brain has them (confirmed against a live
# graph: owl:Thing and gaf:Instance are asserted on essentially every subject) -- comparing
# against "peers" sharing only one of these would mean comparing against almost the whole graph,
# defeating the point of narrowing.
GENERIC_TYPES = {
    "http://www.w3.org/2002/07/owl#Thing",
    "http://groundedannotationframework.org/gaf#Instance",
    "http://groundedannotationframework.org/gaf#Assertion",
}

LABEL_PREDICATE = "http://www.w3.org/2000/01/rdf-schema#label"
# Mirrors events_from_chat/events_to_capsules.py's own AGENT_LIKE_ROLES (duplicated, not
# imported -- see semantic_role_predicates()'s _import_data_type()-style comment on why this
# module avoids cross-directory imports for small constants like this one). "agent" stays first
# deliberately: it's PREDICATE_GROUPS' canonical representative below, and the one
# prompts.response_processor.AGENT_PREDICATES dispatches an agent-confirmation question on.
AGENT_ROLE_PREDICATES = (
    NAMESPACE + "agent", NAMESPACE + "agent_patient", NAMESPACE + "participant", NAMESPACE + "experiencer",
)

# Groups of predicates that are interchangeable ALTERNATIVES for the same underlying fact,
# never meant to be filled in side by side for one instance -- data_type.SemanticRole's own
# docstring documents agent_patient/experiencer/participant as replacing agent (+ patient) when
# a single participant already covers that role. _canonicalize_predicate() collapses every
# predicate in a group to that group's first member everywhere find_predicate_gaps()/
# find_predicate_object_gaps()/find_predicate_object_instances_gaps() count or compare
# predicates, so a subject already filled in via ANY one alternative (e.g. "experiencer") is
# never reported as missing another (e.g. "agent") -- the group is treated as one fact, not
# several independent ones. Add more groups here (as additional tuples) if another comparable
# situation ever comes up; there is currently just the one.
PREDICATE_GROUPS: Tuple[Tuple[str, ...], ...] = (AGENT_ROLE_PREDICATES,)


def _canonicalize_predicate(predicate: str, groups: Tuple[Tuple[str, ...], ...] = PREDICATE_GROUPS) -> str:
    """`predicate`, or -- if it belongs to one of `groups` -- that group's first member, used as
    the whole group's single canonical stand-in. Apply this to every predicate read off a triple
    before counting/comparing it, so e.g. "agent", "agent_patient", "participant" and
    "experiencer" collapse into the one canonical "agent" key rather than being tracked (and
    gap-checked) as four independent predicates."""
    for group in groups:
        if predicate in group:
            return group[0]
    return predicate


# find_predicate_object_gaps() (D) / find_predicate_object_instances_gaps() (E) ask "does this
# instance's value for predicate P match the SPECIFIC value/type most peers share" -- a
# meaningful question for e.g. location or time, where peers doing the same KIND of activity
# plausibly cluster around a shared place or hour. It is NOT meaningful for an agent-like role:
# who performed/experienced one activity is inherently that instance's own business, with no
# sensible "peers mostly share this exact same agent" expectation to compare against -- so an
# instance whose agent-like value simply differs from whatever value happens to be most common
# among its peers would otherwise get a spurious D/E gap on the canonical "agent" predicate,
# EVEN THOUGH it already has an agent-like role filled (which is exactly what B alone should be
# checking -- see find_predicate_gaps()/PREDICATE_GROUPS above). Subtracted from D/E's default
# predicate_filter below; B is unaffected, since "is ANY agent-like predicate filled at all" is
# exactly the meaningful check for this group.
D_E_EXCLUDED_PREDICATES: Set[str] = set(AGENT_ROLE_PREDICATES)


def _labels_similar(a: str, b: str) -> bool:
    """Loose "same or similar activity label" check: exact match after normalizing case and
    surrounding whitespace, one label containing the other (e.g. "cycling" ~ "went cycling this
    morning"), or the two sharing a word longer than 3 characters (long enough to skip articles/
    prepositions like "the"/"for")."""
    a_norm, b_norm = (a or "").strip().lower(), (b or "").strip().lower()
    if not a_norm or not b_norm:
        return False
    if a_norm == b_norm or a_norm in b_norm or b_norm in a_norm:
        return True
    a_words = {w for w in re.findall(r"\w+", a_norm) if len(w) > 3}
    b_words = {w for w in re.findall(r"\w+", b_norm) if len(w) > 3}
    return bool(a_words & b_words)


def _label_for_value(graph: Graph, value: str, is_iri: bool, label_cache: Dict[str, str]) -> str:
    """Human-ish display string for one peer-example object value (see _top_object_examples()):
    the literal itself if it isn't an IRI (e.g. a free-text time value), else that IRI's own
    rdfs:label if it has one, else just the URI's local name (tail after the last '/' or '#') --
    a plain-string equivalent of prompts.response_processor.local_name(), kept local so this
    module stays usable standalone (see its module docstring) without importing chat_from_kg's
    prompt-building code. Memoized in `label_cache` across a whole find_*() call, since the same
    object IRI (e.g. a shared Location instance) can recur across many peers/predicates.
    """
    if not is_iri:
        return value
    if value not in label_cache:
        rows = run_query(graph, f"SELECT ?label WHERE {{ <{value}> <{LABEL_PREDICATE}> ?label }} LIMIT 1")
        label_cache[value] = rows[0]["label"] if rows else value.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1]
    return label_cache[value]


def _top_object_examples(
    graph: Graph,
    rows: List[Tuple[str, bool]],
    label_cache: Dict[str, str],
    limit: int = DEFAULT_PEER_EXAMPLE_LIMIT,
) -> List[Dict]:
    """Turn a bag of (object_value, object_is_iri) rows -- one per peer that actually has a value
    for some predicate -- into the most common `limit` values as
    [{"value": <display string>, "count": <how many peers had it>}, ...], most frequent first.
    This is what lets a gap ("this activity has no `time`") come with concrete peer evidence
    ("most peers said 'in the morning' (3x) or '7pm' (2x)") instead of asking blind.
    """
    is_iri_by_value = dict(rows)  # a given raw value is either always an IRI or never, in practice
    counts = Counter(value for value, _ in rows)
    return [
        {"value": _label_for_value(graph, value, is_iri_by_value[value], label_cache), "count": count}
        for value, count in counts.most_common(limit)
    ]


def _values_clause(types: List[str]) -> str:
    return " ".join(f"<{t}>" for t in types)


def _grouped_by_instance(rows: List[Dict], key: str) -> Dict[str, Set[str]]:
    grouped: Dict[str, Set[str]] = defaultdict(set)
    for r in rows:
        grouped[r["instance"]].add(r[key])
    return grouped


def _instances_with_any_type(graph: Graph, types: List[str], row_limit: int) -> List[str]:
    """Every instance that has at least one of `types` -- the candidate pool find_comparison_peers()
    then narrows in Python, in ONE query regardless of how many types are given (a SPARQL VALUES
    clause), instead of one INSTANCES_QUERY per type."""
    query = f"""
    SELECT DISTINCT ?instance WHERE {{
        VALUES ?t {{ {_values_clause(types)} }}
        ?instance a ?t .
        FILTER(isIRI(?instance))
    }}
    LIMIT {row_limit}
    """
    return [r["instance"] for r in run_query(graph, query)]


def _labels_of_instances(graph: Graph, types: List[str], row_limit: int) -> Dict[str, Set[str]]:
    """{instance: {label, ...}} for every instance of any of `types` that has an rdfs:label,
    in ONE query."""
    query = f"""
    SELECT ?instance ?label WHERE {{
        VALUES ?t {{ {_values_clause(types)} }}
        ?instance a ?t .
        ?instance <{LABEL_PREDICATE}> ?label .
    }}
    LIMIT {row_limit}
    """
    return _grouped_by_instance(run_query(graph, query), "label")


def _types_of_instances(graph: Graph, types: List[str], row_limit: int) -> Dict[str, Set[str]]:
    """{instance: {rdf:type, ...}} (every type, not just the ones in `types`) for every instance
    of any of `types`, in ONE query -- so "same type" can be checked without a per-instance
    TYPES_FOR_SUBJECT_QUERY round-trip."""
    query = f"""
    SELECT ?instance ?type WHERE {{
        VALUES ?t {{ {_values_clause(types)} }}
        ?instance a ?t .
        ?instance a ?type .
        FILTER(isIRI(?type))
    }}
    LIMIT {row_limit}
    """
    return _grouped_by_instance(run_query(graph, query), "type")


def _agent_values_of_instances(
    graph: Graph, types: List[str], row_limit: int, agent_predicates: Tuple[str, ...] = AGENT_ROLE_PREDICATES
) -> Dict[str, Set[str]]:
    """{instance: {value of an agent/agent_patient role, ...}} for every instance of any of
    `types`, in ONE query."""
    union = " UNION ".join(f"{{ ?instance <{p}> ?agentValue }}" for p in agent_predicates)
    query = f"""
    SELECT ?instance ?agentValue WHERE {{
        VALUES ?t {{ {_values_clause(types)} }}
        ?instance a ?t .
        {union}
    }}
    LIMIT {row_limit}
    """
    return _grouped_by_instance(run_query(graph, query), "agentValue")


def find_comparison_peers(
    graph: Graph,
    subject_uri: str,
    max_instances_per_class: int = 2000,
    types_cache: Optional[Dict[str, List[str]]] = None,
) -> Tuple[str, List[str]]:
    """
    Pick a small, targeted comparison peer group for `subject_uri`, instead of the full "every
    instance of every class in the graph" scan find_predicate_gaps() et al. otherwise do -- see
    this module's "Fast, targeted peer selection" comment above for why that doesn't scale.

    Tries three narrowing strategies in order, within the pool of instances that share ANY of
    the subject's own non-generic rdf:type(s) (GENERIC_TYPES excluded -- types nearly every
    resource has, e.g. owl:Thing, which would defeat the narrowing):

      1. Same or similar rdfs:label (_labels_similar()) -- almost certainly the same real-world
         kind of activity (e.g. two mentions of "cycling").
      2. If that leaves fewer than 2 peers: same type (shares a non-generic type with the
         subject) AND the same value for an agent-like role (AGENT_ROLE_PREDICATES: agent,
         agent_patient, participant, experiencer) -- e.g. the same person did it.
      3. If still fewer than 2: same type only (the original per-class comparison basis, but
         scoped to just the subject's own type(s), never the whole graph).

    :return: (types_used, peers) -- peers always includes subject_uri itself (this is a
        comparison GROUP, not "everyone but the subject"); types_used is a " | "-joined display
        string of the subject's own non-generic type(s) (used as the report's "class" field).
        ("(untyped)", [subject_uri]) if the subject has no non-generic type, or fewer than 2
        instances share any of its types even before narrowing by label/agent.
    """
    types_cache = types_cache if types_cache is not None else {}
    subject_types = [t for t in _types_of(graph, subject_uri, types_cache) if t not in GENERIC_TYPES]
    types_label = " | ".join(subject_types) or "(untyped)"
    if not subject_types:
        return types_label, [subject_uri]

    pool = _instances_with_any_type(graph, subject_types, max_instances_per_class)
    if subject_uri not in pool:
        pool.append(subject_uri)
    if len(pool) < 2:
        return types_label, [subject_uri]

    subject_type_set = set(subject_types)

    # Stage 1: same/similar label.
    labels = _labels_of_instances(graph, subject_types, max_instances_per_class)
    subject_labels = labels.get(subject_uri, set())
    if subject_labels:
        peers = [
            inst for inst in pool
            if inst == subject_uri
            or any(_labels_similar(sl, l) for sl in subject_labels for l in labels.get(inst, set()))
        ]
        if len(peers) >= 2:
            return types_label, peers

    # Stage 2: same type + same agent-like role value.
    types_of_pool = _types_of_instances(graph, subject_types, max_instances_per_class)
    agents = _agent_values_of_instances(graph, subject_types, max_instances_per_class)
    subject_agents = agents.get(subject_uri, set())
    if subject_agents:
        peers = [
            inst for inst in pool
            if inst == subject_uri
            or ((types_of_pool.get(inst, set()) & subject_type_set) and (agents.get(inst, set()) & subject_agents))
        ]
        if len(peers) >= 2:
            return types_label, peers

    # Stage 3: same type only.
    peers = [inst for inst in pool if inst == subject_uri or (types_of_pool.get(inst, set()) & subject_type_set)]
    return types_label, peers


def find_predicate_gaps(
    graph: Graph,
    threshold: float = 0.6,
    class_filter: str = None,
    subject_filter: str = None,
    max_instances_per_class: int = 2000,
    predicate_filter: Optional[Set[str]] = DEFAULT_PREDICATE_FILTER,
    peer_example_limit: int = DEFAULT_PEER_EXAMPLE_LIMIT,
) -> List[Dict]:
    """
    For every class, compute which predicates are used by >= `threshold`
    fraction of its instances ("expected" predicates), then, per (class,
    predicate), report every instance missing it as its own row - e.g.
    "Person / hasOccupation / subject=.../Alice (known for 9/12 peers)".
    Each row carries the group's `missing_count`/`total_instances`/
    `peer_coverage` for context, plus a `subject` field naming exactly which
    instance the row is about, so the gap is drillable without a separate
    examples list. Each row also carries `subject_triples`: every other
    (predicate, object) fact already known about that subject (rdf:type
    included), so the row is self-contained context for "what IS known
    about this one, and what's missing" rather than just the latter. And
    `object_triples`: one more hop out - every triple whose subject is an
    IRI object from `subject_triples` that is itself an instance (e.g. if
    the subject worksAt some Organization, `object_triples` includes that
    Organization's own facts; a class URI like the subject's own rdf:type
    value is not followed, since it isn't itself typed).

    Each row also carries `peer_examples`: the up-to-`peer_example_limit` most common actual
    values peers of the same class *do* have for the missing predicate (e.g. for a missing
    `time`, the most frequent times peers reported), most frequent first as
    [{"value": ..., "count": ...}, ...] -- see _top_object_examples(). Empty if no peer has a
    value for that predicate either (shouldn't happen, since the predicate only counts as
    "expected" -- and thus a gap -- once >= threshold of peers have it).

    If `class_filter` is given, it also covers (transitive) rdfs:subClassOf
    descendants of that class - e.g. class_filter=".../Person" pulls in
    instances of ".../Student" or ".../Employee" too, not just exact-type
    ".../Person" instances. Without class_filter (every declared class is
    analyzed), each class is still matched by exact rdf:type only.

    If `subject_filter` is given WITHOUT `class_filter`, peers are chosen by
    find_comparison_peers() instead of scanning every class in the graph -- see its docstring
    for the same/similar-label -> same-type-and-agent -> same-type narrowing it does. Give
    `class_filter` too if you specifically want the old "every instance of exactly this class"
    behaviour for a known subject.

    `predicate_filter` restricts which predicates are even considered, on top of rdf:type always
    being excluded -- by default, semantic_role_predicates() (data_type.SemanticRole), so a
    provenance/bookkeeping predicate like gaf:denotedIn is never reported as a "gap" (nobody can
    usefully be asked a follow-up question about it). Pass an explicit set to use a different
    filter, or None to consider every predicate (the original, unfiltered behaviour).
    """
    if predicate_filter is DEFAULT_PREDICATE_FILTER:
        predicate_filter = semantic_role_predicates()

    subject_triples_cache: Dict[str, List[Dict[str, str]]] = {}
    object_types_cache: Dict[str, List[str]] = {}
    types_cache: Dict[str, List[str]] = {}

    if subject_filter and not class_filter:
        types_label, instances = find_comparison_peers(graph, subject_filter, max_instances_per_class, types_cache)
        class_groups = [(types_label, instances)]
    else:
        if class_filter:
            classes = [class_filter]
        else:
            classes = sorted({r["type"] for r in run_query(graph, CLASSES_QUERY)})
        instances_query = SUBCLASS_INSTANCES_QUERY if class_filter else INSTANCES_QUERY
        class_groups = [
            (cls, [r["instance"] for r in run_query(graph, instances_query % (cls, max_instances_per_class))])
            for cls in classes
        ]

    label_cache: Dict[str, str] = {}

    gaps = []
    for cls, instances in class_groups:
        if len(instances) < 2:
            continue  # nothing to compare against

        predicate_counts = defaultdict(int)
        instance_predicates: Dict[str, Set[str]] = {}
        # Every (object_value, object_is_iri) a peer has for a predicate -- the raw material for
        # that predicate's peer_examples once we know which predicates ended up as gaps.
        predicate_object_rows: Dict[str, List[Tuple[str, bool]]] = defaultdict(list)

        for inst in instances:
            preds = set()
            for r in run_query(graph, PREDICATE_OBJECT_FOR_INSTANCE_QUERY % inst):
                if r["p"] == RDF_TYPE:
                    continue
                if predicate_filter is not None and r["p"] not in predicate_filter:
                    continue
                # Canonicalized (see PREDICATE_GROUPS): "agent"/"agent_patient"/"participant"/
                # "experiencer" all collapse to one "agent" key, so having ANY one of them means
                # none of the group is later reported missing for this instance.
                p = _canonicalize_predicate(r["p"])
                preds.add(p)
                predicate_object_rows[p].append((r["o"], r["oIsIRI"] == "true"))
            instance_predicates[inst] = preds
            for p in preds:
                predicate_counts[p] += 1

        n = len(instances)
        expected = {p for p, c in predicate_counts.items() if c / n >= threshold}
        if not expected:
            continue

        predicate_missing_instances: Dict[str, List[str]] = defaultdict(list)
        for inst, preds in instance_predicates.items():
            if subject_filter and inst != subject_filter:
                continue
            for p in expected - preds:
                predicate_missing_instances[p].append(inst)

        for p, missing_instances in predicate_missing_instances.items():
            peer_examples = _top_object_examples(graph, predicate_object_rows[p], label_cache, peer_example_limit)
            for inst in sorted(missing_instances):
                inst_triples = _triples_of(graph, inst, subject_triples_cache)
                gaps.append(
                    {
                        "class": cls,
                        "predicate": p,
                        "subject": inst,
                        "missing_count": len(missing_instances),
                        "total_instances": n,
                        "peer_coverage": f"{predicate_counts[p]}/{n}",
                        "subject_triples": _strip_internal(inst_triples),
                        "object_triples": _object_triples_of(
                            graph, inst_triples, subject_triples_cache, object_types_cache
                        ),
                        "peer_examples": peer_examples,
                    }
                )

    gaps.sort(key=lambda g: (-g["missing_count"], g["class"], g["predicate"], g["subject"]))
    return gaps


# --------------------------------------------------------------------------- #
# D. Predicate-object gaps, TYPE level ("peers have this fact about a peer of
#    this TYPE, some of you don't") - per class, per (predicate, object-type)
# --------------------------------------------------------------------------- #

def find_predicate_object_gaps(
    graph: Graph,
    threshold: float = 0.6,
    class_filter: str = None,
    subject_filter: str = None,
    max_instances_per_class: int = 2000,
    predicate_filter: Optional[Set[str]] = DEFAULT_PREDICATE_FILTER,
    peer_example_limit: int = DEFAULT_PEER_EXAMPLE_LIMIT,
) -> List[Dict]:
    """
    For every class (or just `class_filter` if given), find (predicate,
    object-TYPE) combinations that a majority of instances of that class
    share, then, per (class, predicate, object-type), report every instance
    lacking a fact with an object of that type as its own row - even if it
    has some other value for the same predicate. Each row carries the
    group's `missing_count`/`total_instances`/`peer_coverage` for context,
    plus a `subject` field naming exactly which instance the row is about,
    `subject_triples`: every other (predicate, object) fact already known
    about that subject (rdf:type included), and `object_triples`: one more
    hop out - every triple whose subject is an IRI object from
    `subject_triples` that is itself an instance (has its own rdf:type).

    Object IRIs are compared by their rdf:type(s) rather than by their exact
    identity, e.g. "most Lunches have a hasPatient of type Wine" instead of
    "most Lunches have hasPatient=wine1" - two different Wine individuals
    (wine1, wine2) count as satisfying the same expectation, so this doesn't
    fragment into one gap per distinct individual the way instance-level
    comparison does. An object IRI with no declared rdf:type of its own falls
    back to being compared by its exact identity (nothing more specific is
    known about it). Literal objects have no separate "type" from their
    value, so they're still compared by exact literal value, same as before.
    An object with multiple rdf:types contributes one (predicate, type) pair
    per type.

    This is a finer-grained sibling of find_predicate_gaps(): that function
    flags a missing predicate in general (e.g. "some of you have no
    hasHobby"), while this one flags a specific missing predicate-object-type
    pair (e.g. "most people like you have a hasHobby of type OutdoorActivity,
    but 2 of you don't have that").

    For the exact-object-identity version of this check, see
    find_predicate_object_instances_gaps().

    Pairs and their peer counts are computed once per class and then reused
    for every instance in that class, rather than re-querying peers per
    subject.

    If `class_filter` is given, it also covers (transitive) rdfs:subClassOf
    descendants of that class (same as find_predicate_gaps()) - e.g.
    class_filter=".../Person" pulls in instances of ".../Student" etc. too.

    If `subject_filter` is given WITHOUT `class_filter`, peers are chosen by
    find_comparison_peers() instead of scanning every class in the graph -- see its docstring
    for the same/similar-label -> same-type-and-agent -> same-type narrowing it does. Give
    `class_filter` too if you specifically want the old "every instance of exactly this class"
    behaviour for a known subject. In that case every returned row's "subject" will be that one
    subject. Blank-node objects are excluded since they have no stable identity to compare
    across instances.

    `predicate_filter` restricts which predicates are even considered, on top of rdf:type always
    being excluded -- by default, semantic_role_predicates() (data_type.SemanticRole), so a
    provenance/bookkeeping predicate like gaf:denotedIn is never reported as a "gap" (nobody can
    usefully be asked a follow-up question about it). Pass an explicit set to use a different
    filter, or None to consider every predicate (the original, unfiltered behaviour).

    Each row also carries `peer_examples`: the up-to-`peer_example_limit` most common actual
    object VALUES (not just the shared type) peers have contributed to this (predicate,
    object-type) pair, most frequent first -- see _top_object_examples(). E.g. for "most Lunches
    have a hasPatient of type Wine", peer_examples might be [{"value": "Merlot", "count": 3},
    {"value": "Chardonnay", "count": 2}], so the follow-up question can suggest concrete options
    instead of just naming the type.

    The default predicate_filter also excludes D_E_EXCLUDED_PREDICATES (the agent-like role
    group) on top of restricting to semantic_role_predicates() -- see its own comment for why:
    "does this instance's agent value match most peers' specific agent value" isn't a meaningful
    gap the way it is for e.g. location or time. find_predicate_gaps() (B) is unaffected and
    still reports a gap for this group when NONE of its predicates are filled at all.
    """
    if predicate_filter is DEFAULT_PREDICATE_FILTER:
        predicate_filter = semantic_role_predicates() - D_E_EXCLUDED_PREDICATES

    object_types_cache: Dict[str, List[str]] = {}
    subject_triples_cache: Dict[str, List[Dict[str, str]]] = {}
    types_cache: Dict[str, List[str]] = {}
    label_cache: Dict[str, str] = {}

    if subject_filter and not class_filter:
        types_label, instances = find_comparison_peers(graph, subject_filter, max_instances_per_class, types_cache)
        class_groups = [(types_label, instances)]
    else:
        if class_filter:
            classes = [class_filter]
        else:
            classes = sorted({r["type"] for r in run_query(graph, CLASSES_QUERY)})
        instances_query = SUBCLASS_INSTANCES_QUERY if class_filter else INSTANCES_QUERY
        class_groups = [
            (cls, [r["instance"] for r in run_query(graph, instances_query % (cls, max_instances_per_class))])
            for cls in classes
        ]

    gaps = []
    for cls, instances in class_groups:
        if len(instances) < 2:
            continue  # nothing to compare against

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        instance_pairs: Dict[str, Set[Tuple[str, str]]] = {}
        # Every (object_value, object_is_iri) a peer actually has that counts towards a
        # (predicate, object-type) pair -- the raw material for that pair's peer_examples. A
        # single (p, o) triple can feed more than one pair when o has multiple rdf:types.
        pair_object_rows: Dict[Tuple[str, str], List[Tuple[str, bool]]] = defaultdict(list)

        for inst in instances:
            pairs = set()
            for r in run_query(graph, PREDICATE_OBJECT_FOR_INSTANCE_QUERY % inst):
                if r["p"] == RDF_TYPE:
                    continue
                if predicate_filter is not None and r["p"] not in predicate_filter:
                    continue
                # Canonicalized (see PREDICATE_GROUPS): a (predicate, object-type) pair on any
                # agent-like predicate (agent/agent_patient/participant/experiencer) collapses
                # onto one canonical "agent" key, same reasoning as find_predicate_gaps().
                p = _canonicalize_predicate(r["p"])
                o_is_iri = r["oIsIRI"] == "true"
                if o_is_iri:
                    obj_types = _types_of(graph, r["o"], object_types_cache)
                    if obj_types:
                        for t in obj_types:
                            pairs.add((p, t))
                            pair_object_rows[(p, t)].append((r["o"], o_is_iri))
                        continue
                # literal, or an untyped object IRI we know nothing more about
                pairs.add((p, r["o"]))
                pair_object_rows[(p, r["o"])].append((r["o"], o_is_iri))
            instance_pairs[inst] = pairs
            for pair in pairs:
                pair_counts[pair] += 1

        n = len(instances)
        expected = {pair for pair, c in pair_counts.items() if c / n >= threshold}
        if not expected:
            continue

        pair_missing_instances: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for inst, pairs in instance_pairs.items():
            if subject_filter and inst != subject_filter:
                continue
            for pair in expected - pairs:
                pair_missing_instances[pair].append(inst)

        for (p, o), missing_instances in pair_missing_instances.items():
            peer_examples = _top_object_examples(graph, pair_object_rows[(p, o)], label_cache, peer_example_limit)
            for inst in sorted(missing_instances):
                inst_triples = _triples_of(graph, inst, subject_triples_cache)
                gaps.append(
                    {
                        "class": cls,
                        "predicate": p,
                        "object_type": o,
                        "subject": inst,
                        "missing_count": len(missing_instances),
                        "total_instances": n,
                        "peer_coverage": f"{pair_counts[(p, o)]}/{n}",
                        "subject_triples": _strip_internal(inst_triples),
                        "object_triples": _object_triples_of(
                            graph, inst_triples, subject_triples_cache, object_types_cache
                        ),
                        "peer_examples": peer_examples,
                    }
                )

    gaps.sort(key=lambda g: (-g["missing_count"], g["class"], g["predicate"], g["object_type"], g["subject"]))
    return gaps


# --------------------------------------------------------------------------- #
# E. Predicate-object gaps, INSTANCE level ("peers have this exact fact, some
#    of you don't") - per class, per (predicate, object), for all instances
#    or one subject
# --------------------------------------------------------------------------- #

def find_predicate_object_instances_gaps(
    graph: Graph,
    threshold: float = 0.6,
    class_filter: str = None,
    subject_filter: str = None,
    max_instances_per_class: int = 2000,
    predicate_filter: Optional[Set[str]] = DEFAULT_PREDICATE_FILTER,
    peer_example_limit: int = DEFAULT_PEER_EXAMPLE_LIMIT,
) -> List[Dict]:
    """
    For every class (or just `class_filter` if given), find (predicate,
    object) combinations that a majority of instances of that class share,
    then, per (class, predicate, object), report every instance lacking that
    specific fact as its own row - even if it has some other value for the
    same predicate. Each row carries the group's
    `missing_count`/`total_instances`/`peer_coverage` for context, plus a
    `subject` field naming exactly which instance the row is about,
    `subject_triples`: every other (predicate, object) fact already known
    about that subject (rdf:type included), and `object_triples`: one more
    hop out - every triple whose subject is an IRI object from
    `subject_triples` that is itself an instance (has its own rdf:type).

    Object IRIs are compared by exact identity here (e.g. hasPatient=wine1 is
    a different fact from hasPatient=wine2), which means this tends to
    fragment into many small, individual-specific gaps whenever a predicate
    points at many distinct individuals of the same type. For the coarser,
    type-level version of this check (comparing "a Wine" rather than
    "wine1"), see find_predicate_object_gaps().

    Pairs and their peer counts are computed once per class and then reused
    for every instance in that class, rather than re-querying peers per
    subject.

    If `class_filter` is given, it also covers (transitive) rdfs:subClassOf
    descendants of that class (same as find_predicate_gaps()) - e.g.
    class_filter=".../Person" pulls in instances of ".../Student" etc. too.

    If `subject_filter` is given WITHOUT `class_filter`, peers are chosen by
    find_comparison_peers() instead of scanning every class in the graph -- see its docstring
    for the same/similar-label -> same-type-and-agent -> same-type narrowing it does. Give
    `class_filter` too if you specifically want the old "every instance of exactly this class"
    behaviour for a known subject. In that case every returned row's "subject" will be that one
    subject. Blank-node objects are excluded since they have no stable identity to compare
    across instances; literals and IRIs are both compared as-is.

    `predicate_filter` restricts which predicates are even considered, on top of rdf:type always
    being excluded -- by default, semantic_role_predicates() (data_type.SemanticRole), so a
    provenance/bookkeeping predicate like gaf:denotedIn is never reported as a "gap" (nobody can
    usefully be asked a follow-up question about it). Pass an explicit set to use a different
    filter, or None to consider every predicate (the original, unfiltered behaviour).

    Each row also carries `peer_examples`: here just the one expected object itself (this gap
    kind already names the exact expected value), as
    [{"value": <display string, resolved via rdfs:label if o is an IRI>, "count": peer_coverage
    numerator}] -- kept as the same peer_examples shape find_predicate_gaps() /
    find_predicate_object_gaps() use, so callers (e.g. prompts.response_processor) don't need to
    special-case this gap kind to show peer evidence.

    The default predicate_filter also excludes D_E_EXCLUDED_PREDICATES (the agent-like role
    group) on top of restricting to semantic_role_predicates() -- see its own comment for why:
    "does this instance's agent value match most peers' one exact agent value" is even less
    meaningful at the instance level (E) than at the type level (D). find_predicate_gaps() (B)
    is unaffected and still reports a gap for this group when NONE of its predicates are filled.
    """
    if predicate_filter is DEFAULT_PREDICATE_FILTER:
        predicate_filter = semantic_role_predicates() - D_E_EXCLUDED_PREDICATES

    subject_triples_cache: Dict[str, List[Dict[str, str]]] = {}
    object_types_cache: Dict[str, List[str]] = {}
    types_cache: Dict[str, List[str]] = {}
    label_cache: Dict[str, str] = {}

    if subject_filter and not class_filter:
        types_label, instances = find_comparison_peers(graph, subject_filter, max_instances_per_class, types_cache)
        class_groups = [(types_label, instances)]
    else:
        if class_filter:
            classes = [class_filter]
        else:
            classes = sorted({r["type"] for r in run_query(graph, CLASSES_QUERY)})
        instances_query = SUBCLASS_INSTANCES_QUERY if class_filter else INSTANCES_QUERY
        class_groups = [
            (cls, [r["instance"] for r in run_query(graph, instances_query % (cls, max_instances_per_class))])
            for cls in classes
        ]

    gaps = []
    for cls, instances in class_groups:
        if len(instances) < 2:
            continue  # nothing to compare against

        pair_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        instance_pairs: Dict[str, Set[Tuple[str, str]]] = {}
        object_is_iri: Dict[str, bool] = {}

        for inst in instances:
            pairs = set()
            for r in run_query(graph, PREDICATE_OBJECT_FOR_INSTANCE_QUERY % inst):
                if r["p"] == RDF_TYPE or (predicate_filter is not None and r["p"] not in predicate_filter):
                    continue
                # Canonicalized (see PREDICATE_GROUPS), same reasoning as find_predicate_gaps().
                pairs.add((_canonicalize_predicate(r["p"]), r["o"]))
                object_is_iri[r["o"]] = r["oIsIRI"] == "true"
            instance_pairs[inst] = pairs
            for pair in pairs:
                pair_counts[pair] += 1

        n = len(instances)
        expected = {pair for pair, c in pair_counts.items() if c / n >= threshold}
        if not expected:
            continue

        pair_missing_instances: Dict[Tuple[str, str], List[str]] = defaultdict(list)
        for inst, pairs in instance_pairs.items():
            if subject_filter and inst != subject_filter:
                continue
            for pair in expected - pairs:
                pair_missing_instances[pair].append(inst)

        for (p, o), missing_instances in pair_missing_instances.items():
            peer_examples = [
                {"value": _label_for_value(graph, o, object_is_iri.get(o, False), label_cache),
                 "count": pair_counts[(p, o)]}
            ] if peer_example_limit > 0 else []
            for inst in sorted(missing_instances):
                inst_triples = _triples_of(graph, inst, subject_triples_cache)
                gaps.append(
                    {
                        "class": cls,
                        "predicate": p,
                        "object": o,
                        "subject": inst,
                        "missing_count": len(missing_instances),
                        "total_instances": n,
                        "peer_coverage": f"{pair_counts[(p, o)]}/{n}",
                        "subject_triples": _strip_internal(inst_triples),
                        "object_triples": _object_triples_of(
                            graph, inst_triples, subject_triples_cache, object_types_cache
                        ),
                        "peer_examples": peer_examples,
                    }
                )

    gaps.sort(key=lambda g: (-g["missing_count"], g["class"], g["predicate"], g["object"], g["subject"]))
    return gaps


# --------------------------------------------------------------------------- #
# C. Dangling references (objects known only by name, nothing further known),
#    grouped by the (referencing type, predicate) that points at them
# --------------------------------------------------------------------------- #

DANGLING_QUERY = """
SELECT ?object ?p ?sType WHERE {
    ?s ?p ?object .
    OPTIONAL { ?s a ?sType }
    FILTER(isIRI(?object))
    FILTER NOT EXISTS { ?object ?p2 ?o2 . }
}
LIMIT %d
"""


def find_dangling_references(
    graph: Graph, row_limit: int = 5000, example_limit: int = DEFAULT_EXAMPLE_LIMIT
) -> List[Dict]:
    """
    Find objects that are referenced somewhere but have no outgoing triples
    of their own (we only know their name/id), grouped by the type of the
    referencing subject and the predicate used to reach them - e.g. "Person /
    hasFriend points at 12 references we know nothing further about". If the
    referencing subject itself has no declared type, it's grouped under
    "(untyped subject)".

    Note: since a dangling object has zero outgoing triples by definition,
    it cannot itself have an rdf:type - that's exactly the gap being
    reported. Grouping by the *referencing* type/predicate is what makes the
    finding pattern-level instead of a flat list of individual objects.
    """
    rows = run_query(graph, DANGLING_QUERY % row_limit)

    groups: Dict[Tuple[str, str], Set[str]] = defaultdict(set)
    for r in rows:
        s_type = r["sType"] if r["sType"] is not None else "(untyped subject)"
        groups[(s_type, r["p"])].add(r["object"])

    results = []
    for (s_type, p), objs in groups.items():
        objs_sorted = sorted(objs)
        results.append(
            {
                "referencing_type": s_type,
                "predicate": p,
                "count": len(objs_sorted),
                "example_objects": objs_sorted[:example_limit],
            }
        )
    results.sort(key=lambda g: -g["count"])
    return results


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def build_report(
    graph: Graph,
    threshold: float,
    class_filter: str,
    subject_filter: str = None,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
    predicate_filter: Optional[Set[str]] = DEFAULT_PREDICATE_FILTER,
    peer_example_limit: int = DEFAULT_PEER_EXAMPLE_LIMIT,
) -> Dict:
    """predicate_filter is passed through to the three predicate-based gap kinds (B, D, E) only
    -- see find_predicate_gaps() for what it does. Sections A and C (find_untyped_entities(),
    find_dangling_references()) look for a different kind of gap (missing rdf:type, and
    referenced-but-unknown entities) that predicate_filter doesn't apply to. peer_example_limit
    is likewise B/D/E-only -- see find_predicate_gaps()'s `peer_examples` for what it does.

    Resets run_query()'s query log (see LOG_QUERIES) at the start, and prints a one-line
    count/total-time summary at the end, so that summary -- and every per-query log line in
    between -- reflects just this one build_report() call."""
    _reset_query_log()
    report = {
        "untyped_entities": find_untyped_entities(graph, example_limit=example_limit),
        "predicate_gaps": find_predicate_gaps(
            graph, threshold=threshold, class_filter=class_filter, subject_filter=subject_filter,
            predicate_filter=predicate_filter, peer_example_limit=peer_example_limit,
        ),
        "dangling_references": find_dangling_references(graph, example_limit=example_limit),
        "predicate_object_gaps": find_predicate_object_gaps(
            graph,
            threshold=threshold,
            class_filter=class_filter,
            subject_filter=subject_filter,
            predicate_filter=predicate_filter,
            peer_example_limit=peer_example_limit,
        ),
        "predicate_object_instances_gaps": find_predicate_object_instances_gaps(
            graph,
            threshold=threshold,
            class_filter=class_filter,
            subject_filter=subject_filter,
            predicate_filter=predicate_filter,
            peer_example_limit=peer_example_limit,
        ),
    }
    if LOG_QUERIES:
        print(
            f"[kg_gap_finder] {_query_stats['count']} quer{'y' if _query_stats['count'] == 1 else 'ies'}, "
            f"{_query_stats['total_time']:.3f}s total"
        )
    return report


def _print_subject_triples(subject_triples: List[Dict[str, str]]) -> None:
    """Print a subject's other known triples, indented under its gap row."""
    if not subject_triples:
        print("        (no other triples known for this subject)")
        return
    for t in subject_triples:
        print(f"        {t['predicate']} = {t['object']}")


def _print_peer_examples(peer_examples: List[Dict]) -> None:
    """Print a gap row's peer_examples (see find_predicate_gaps()), most frequent first."""
    if not peer_examples:
        return
    rendered = ", ".join(
        f"{e['value']} ({e['count']}x)" if e["count"] > 1 else e["value"] for e in peer_examples
    )
    print(f"      peer examples: {rendered}")


def _print_object_triples(object_triples: List[Dict[str, str]]) -> None:
    """Print one-hop-out triples (facts about objects the subject points at), grouped by that object."""
    if not object_triples:
        return
    print("      object triples (facts about the subject's own instance-typed objects, one hop out):")
    last_subject = None
    for t in object_triples:
        if t["subject"] != last_subject:
            print(f"        {t['subject']}")
            last_subject = t["subject"]
        print(f"          {t['predicate']} = {t['object']}")


def print_report(report: Dict) -> None:
    print("\n=== A. Untyped entities, grouped by predicate signature ===")
    if not report["untyped_entities"]:
        print("  (none found)")
    for g in report["untyped_entities"][:50]:
        print(f"  - {g['instance_count']} untyped entities share predicates: {', '.join(g['predicate_signature']) or '(none)'}")
        for e in g["example_entities"]:
            print(f"      e.g. {e}")

    print("\n=== B. Predicate gaps (some instances missing what peers usually have) ===")
    if not report["predicate_gaps"]:
        print("  (none found)")
    for g in report["predicate_gaps"][:50]:
        print(f"  - {g['class']}  missing: {g['predicate']}  (known for {g['peer_coverage']} peers, {g['missing_count']}/{g['total_instances']} missing)")
        print(f"      subject: {g['subject']}")
        _print_subject_triples(g["subject_triples"])
        _print_peer_examples(g.get("peer_examples") or [])
        _print_object_triples(g["object_triples"])

    print("\n=== C. Dangling references (named but nothing else known) ===")
    if not report["dangling_references"]:
        print("  (none found)")
    for g in report["dangling_references"][:50]:
        print(f"  - {g['referencing_type']}  --{g['predicate']}-->  {g['count']} dangling references")
        for o in g["example_objects"]:
            print(f"      e.g. {o}")

    print("\n=== D. Predicate-object gaps, type level (some instances missing a fact whose object TYPE most peers share) ===")
    if not report["predicate_object_gaps"]:
        print("  (none found)")
    for g in report["predicate_object_gaps"][:50]:
        print(f"  - {g['class']}  missing: {g['predicate']} -> (type) {g['object_type']}  (known for {g['peer_coverage']} peers, {g['missing_count']}/{g['total_instances']} missing)")
        print(f"      subject: {g['subject']}")
        _print_subject_triples(g["subject_triples"])
        _print_peer_examples(g.get("peer_examples") or [])
        _print_object_triples(g["object_triples"])

    print("\n=== E. Predicate-object gaps, instance level (some instances missing the exact fact most peers share) ===")
    if not report["predicate_object_instances_gaps"]:
        print("  (none found)")
    for g in report["predicate_object_instances_gaps"][:50]:
        print(f"  - {g['class']}  missing: {g['predicate']} = {g['object']}  (known for {g['peer_coverage']} peers, {g['missing_count']}/{g['total_instances']} missing)")
        print(f"      subject: {g['subject']}")
        _print_subject_triples(g["subject_triples"])
        _print_object_triples(g["object_triples"])

    # B, D, E are now one row per affected subject, so their row count IS the
    # affected-instance count; distinct (class, predicate[, object]) combos
    # are counted separately below for the "grouped into" summary.
    predicate_gap_groups = {(g["class"], g["predicate"]) for g in report["predicate_gaps"]}
    predicate_object_gap_groups = {
        (g["class"], g["predicate"], g["object_type"]) for g in report["predicate_object_gaps"]
    }
    predicate_object_instance_gap_groups = {
        (g["class"], g["predicate"], g["object"]) for g in report["predicate_object_instances_gaps"]
    }

    total = (
        sum(g["instance_count"] for g in report["untyped_entities"])
        + len(report["predicate_gaps"])
        + sum(g["count"] for g in report["dangling_references"])
        + len(report["predicate_object_gaps"])
        + len(report["predicate_object_instances_gaps"])
    )
    print(f"\nTotal affected instances/entities across all gap types: {total}")
    print(
        f"Grouped into {len(report['untyped_entities'])} untyped-signature group(s), "
        f"{len(predicate_gap_groups)} predicate-gap group(s) affecting {len(report['predicate_gaps'])} subject(s), "
        f"{len(report['dangling_references'])} dangling-reference group(s), "
        f"{len(predicate_object_gap_groups)} predicate-object-type-gap group(s) affecting {len(report['predicate_object_gaps'])} subject(s), "
        f"{len(predicate_object_instance_gap_groups)} predicate-object-instance-gap group(s) affecting {len(report['predicate_object_instances_gaps'])} subject(s)."
    )


def _write_json_report(report: Dict, json_path: str) -> None:
    """
    Write the report as JSON to `json_path`, which may be relative (resolved
    against the current working directory, same as plain open()) or
    absolute. Any missing parent directories are created first, so e.g.
    --json out/report.json works even if "out/" doesn't exist yet.
    """
    json_path = os.path.abspath(json_path)
    parent = os.path.dirname(json_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)


# --------------------------------------------------------------------------- #
# Convenience entry point (e.g. for Colab / notebooks, no argparse needed)
# --------------------------------------------------------------------------- #

def analyze(
    file: str = None,
    endpoint: str = None,
    fmt: str = None,
    threshold: float = 0.6,
    class_uri: str = None,
    subject_uri: str = None,
    example_limit: int = DEFAULT_EXAMPLE_LIMIT,
    print_output: bool = True,
    json_path: str = None,
    predicate_filter: Optional[Set[str]] = DEFAULT_PREDICATE_FILTER,
    peer_example_limit: int = DEFAULT_PEER_EXAMPLE_LIMIT,
) -> Dict:
    """
    Convenience wrapper for calling the gap finder directly from a notebook
    (e.g. Colab) without going through the CLI/argparse.

    threshold is exposed here as a plain function parameter: the fraction of
    peers (0.0-1.0) that must share a predicate (or predicate-object pair)
    for it to count as "expected". Lower it (e.g. 0.5) to also catch 50/50
    splits like a 2-instance class where only one instance has a predicate.

    example_limit controls how many example entities are kept per group for
    drill-down in the untyped-entities (A) and dangling-references (C)
    sections (the reported counts always reflect the full group, regardless
    of this limit). The predicate-gap sections (B, D, E) instead report one
    row per affected subject, so example_limit does not apply to them.

    class_uri, if given, also covers (transitive) rdfs:subClassOf descendants
    of that class - e.g. class_uri=".../Person" also picks up instances of
    ".../Student" or ".../Employee".

    predicate_filter restricts sections B, D and E to a fixed set of predicates -- by default,
    semantic_role_predicates() (data_type.SemanticRole), so gaps are only reported for
    predicates a coach could ask a follow-up question about (agent, patient, location, time,
    ...), not the brain's own provenance/bookkeeping predicates (rdfs:label,
    gaf:denotedIn/denotedBy, ...). Pass None to consider every predicate instead (useful against
    a graph that isn't this project's own ontology).

    peer_example_limit caps how many "peer_examples" (see find_predicate_gaps()) each B/D/E gap
    row carries -- the actual values peers do have for the missing predicate, most-frequent
    first, so a follow-up question can be concrete ("most peers exercise in the morning") rather
    than blind. Pass 0 to disable.

    Example:
        report = analyze(file="mybrain.trig", fmt="trig", threshold=0.5)
    """
    if not file and not endpoint:
        raise ValueError("Provide either file= or endpoint=")

    if endpoint:
        graph = load_graph_from_endpoint(endpoint)
    else:
        graph = load_graph_from_file(file, fmt)

    report = build_report(
        graph,
        threshold=threshold,
        class_filter=class_uri,
        subject_filter=subject_uri,
        example_limit=example_limit,
        predicate_filter=predicate_filter,
        peer_example_limit=peer_example_limit,
    )

    if print_output:
        print_report(report)

    if json_path:
        _write_json_report(report, json_path)

    return report


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Query an RDF knowledge graph for knowledge gaps, grouped by type.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--endpoint", help="SPARQL endpoint URL, e.g. http://localhost:7200/repositories/sandbox")
    src.add_argument("--file", help="Path to a local RDF file")
    parser.add_argument("--format", default=None, help="rdflib format for --file, e.g. turtle, trig, xml, json-ld")
    parser.add_argument("--threshold", type=float, default=0.6, help="Fraction of peers that must share a predicate (or predicate-object pair) for it to count as 'expected' (default 0.6)")
    parser.add_argument("--class-uri", default=None, help="Restrict predicate-gap / predicate-object-gap analysis to a single rdf:type URI (also includes its rdfs:subClassOf descendants)")
    parser.add_argument("--subject-uri", default=None, help="Restrict predicate gaps (sections B, D and E) to one specific subject URI instead of all instances")
    parser.add_argument("--examples", type=int, default=DEFAULT_EXAMPLE_LIMIT, help=f"Number of example entities to keep per group for drill-down in sections A and C only (default {DEFAULT_EXAMPLE_LIMIT}); sections B, D, E always report one row per affected subject")
    parser.add_argument("--peer-examples", type=int, default=DEFAULT_PEER_EXAMPLE_LIMIT, help=f"Number of actual peer values to attach to each B/D/E gap row as 'peer_examples' (default {DEFAULT_PEER_EXAMPLE_LIMIT}; 0 disables)")
    parser.add_argument("--all-predicates", action="store_true", help="Consider every predicate for sections B, D and E, instead of only the semantic-role predicates from data_type.SemanticRole (agent, patient, location, time, ...) -- use this against a graph that isn't this project's own ontology")
    parser.add_argument("--json", default=None, help="Optional path to write the full report as JSON; may be relative (resolved against the current working directory) or absolute, any missing parent directories are created automatically")
    args = parser.parse_args()

    if args.endpoint:
        graph = load_graph_from_endpoint(args.endpoint)
        print(f"Connected to SPARQL endpoint: {args.endpoint}")
    else:
        graph = load_graph_from_file(args.file, args.format)
        print(f"Loaded {len(graph)} triples from {args.file}")

    report = build_report(
        graph,
        threshold=args.threshold,
        class_filter=args.class_uri,
        subject_filter=args.subject_uri,
        example_limit=args.examples,
        predicate_filter=None if args.all_predicates else DEFAULT_PREDICATE_FILTER,
        peer_example_limit=args.peer_examples,
    )
    print_report(report)

    if args.json:
        _write_json_report(report, args.json)
        print(f"\nFull report written to {os.path.abspath(args.json)}")


if __name__ == "__main__":
    main()
