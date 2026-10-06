"""
n2mu_sem_roles.py
===================

A small `rdfs:subPropertyOf` bridge between this project's own fine-grained SRL role predicates
(`http://cltl.nl/leolani/n2mu/agent`, `.../location`, `.../time/dateTime`, ... -- asserted by
`events_from_chat/events_to_capsules.py` on every activity/condition pushed to the graph) and the
coarser SEM ontology event roles (`http://semanticweb.cs.vu.nl/2009/11/sem/hasActor`/`hasPlace`/
`hasTime`) that `gaps_from_kg/thought_util.py`'s `get_sem_relation_query()` actually queries for.

Without this, `get_sem_relation_query()` (used by `get_temporal_containers.get_temporal_containers()`
to read an activity's own actor(s)/place/time) finds nothing for any real activity: activities are
never asserted with `sem:hasActor`/`sem:hasPlace`/`sem:hasTime` triples directly, only with the
`n2mu:` role predicates above. `get_sem_relation_query()` now looks for any predicate that is a
(reflexive-transitive) `rdfs:subPropertyOf` of the sem: role it wants -- see its own docstring --
so uploading the mapping below is what actually lets it see `n2mu:agent`/`n2mu:location`/
`n2mu:time/...` triples as the sem: role they stand in for, with no repository-level RDFS
reasoning required (the property path is evaluated directly against these mapping triples, so it
works whether or not the GraphDB repository has an inference ruleset enabled).

Only the roles `get_sem_relation_query()` actually reads are mapped:

- **actor-like roles -> `sem:hasActor`**: `agent`, `agent_patient`, `participant`, `experiencer` --
  the same four-role "who was involved" group `chat_from_kg/kg_gap_finder.py`'s own
  `AGENT_ROLE_PREDICATES` already treats as interchangeable alternatives for the same underlying
  fact (duplicated here, not imported -- `gaps_from_kg/` and `chat_from_kg/` are independent flat
  module directories, same convention `kg_gap_finder.py`'s own `_import_data_type()` comment
  documents for avoiding cross-directory imports of small constants). `patient` is deliberately
  NOT included -- it's the participant *affected by* the activity, not one *performing/
  experiencing* it, so it isn't an "actor" in SEM's sense.
- **`location` -> `sem:hasPlace`**.
- **`time/dateTime`, `time/rangeTime`, `time/recurringTime`, `time/vagueTime` -> `sem:hasTime`** --
  the four typed variants `events_to_capsules.py` resolves a `time` role into (see
  `chat_from_kg/kg_gap_finder.py`'s own `TIME_PREDICATE_VARIANTS`), never a bare `n2mu:time`.

`instrument`/`manner`/`qualification`/`result` are left unmapped -- `get_sem_relation_query()`
doesn't read them, and SEM has no obvious equivalent role for any of them.
"""

from rdflib import Graph
from rdflib.plugins.stores import sparqlstore

N2MU_NAMESPACE = "http://cltl.nl/leolani/n2mu/"
SEM_NAMESPACE = "http://semanticweb.cs.vu.nl/2009/11/sem/"
RDFS_SUBPROPERTY_OF = "http://www.w3.org/2000/01/rdf-schema#subPropertyOf"

ACTOR_LIKE_ROLES = ("agent", "agent_patient", "participant", "experiencer")
TIME_ROLE_VARIANTS = ("dateTime", "rangeTime", "recurringTime", "vagueTime")

# n2mu role local name -> sem role local name, for every mapping this module declares.
ROLE_SUBPROPERTY_MAP = {
    **{role: "hasActor" for role in ACTOR_LIKE_ROLES},
    "location": "hasPlace",
    **{f"time/{variant}": "hasTime" for variant in TIME_ROLE_VARIANTS},
}


def build_subproperty_triples():
    """One (n2mu role URI, sem role URI) pair per ROLE_SUBPROPERTY_MAP entry."""
    return [
        (N2MU_NAMESPACE + n2mu_role, SEM_NAMESPACE + sem_role)
        for n2mu_role, sem_role in ROLE_SUBPROPERTY_MAP.items()
    ]


def build_subproperty_update() -> str:
    """The SPARQL `INSERT DATA` update that asserts every ROLE_SUBPROPERTY_MAP mapping as an
    `rdfs:subPropertyOf` triple -- see upload_role_hierarchy(). Plain RDF triples in a set, so
    running this more than once is harmless (re-inserting an already-present triple is a no-op)."""
    lines = [
        f"<{n2mu_uri}> <{RDFS_SUBPROPERTY_OF}> <{sem_uri}> ."
        for n2mu_uri, sem_uri in build_subproperty_triples()
    ]
    return "INSERT DATA {\n  " + "\n  ".join(lines) + "\n}"


def _update_endpoint(kg_address: str) -> str:
    """The GraphDB SPARQL 1.1 Update endpoint for a repository's own query endpoint address
    (`.../repositories/<repo>` -> `.../repositories/<repo>/statements`) -- same repository
    `kg_gap_finder.load_graph_from_endpoint()`/`populate_ekg_from_annotations()` already point at,
    just its update (not query-only) path. Left unchanged if `kg_address` already ends in
    "/statements"."""
    base = kg_address.rstrip("/")
    return base if base.endswith("/statements") else base + "/statements"


def role_hierarchy_uploaded(kg_address: str) -> bool:
    """True if `kg_address` already has (at least) one of ROLE_SUBPROPERTY_MAP's triples --
    good enough to tell "has upload_role_hierarchy() already run against this repository" apart
    from "hasn't yet", without re-querying for every single mapping."""
    n2mu_uri, sem_uri = build_subproperty_triples()[0]
    store = sparqlstore.SPARQLStore(kg_address)
    graph = Graph(store=store)
    return bool(graph.query(f"ASK {{ <{n2mu_uri}> <{RDFS_SUBPROPERTY_OF}> <{sem_uri}> . }}"))


def upload_role_hierarchy(kg_address: str) -> None:
    """Upload every ROLE_SUBPROPERTY_MAP mapping (see the module docstring) to `kg_address` as
    `rdfs:subPropertyOf` triples, via a plain SPARQL 1.1 Update -- run this ONCE per repository
    (e.g. from a notebook's setup cell, guarded by `if not role_hierarchy_uploaded(KG_ADDRESS):`)
    before relying on `get_temporal_containers.get_temporal_containers()`/
    `get_last_conversation_date()` (or anything else built on
    `thought_util.get_sem_relation_query()`) to actually find real activities' actor/place/time.
    """
    # context_aware=False -- a plain `Graph(store=store)` gets a random BNode as its own
    # `identifier` (rdflib's default when none is given), and SPARQLUpdateStore.update() -- when
    # context_aware (its own default) -- wraps every update in a `GRAPH <that identifier> { ... }`
    # block to scope it to that "context". Since that identifier is a BNode, and SPARQL has no
    # syntax for a blank-node graph name, serializing it raises rdflib's own
    # "SPARQLStore does not support BNodes!" -- for every single update, not just ones that
    # happen to touch blank-node data. We don't want a named-graph-scoped update at all (just a
    # plain INSERT DATA against the repository's default graph), so context-awareness is
    # switched off entirely rather than worked around with an explicit identifier.
    store = sparqlstore.SPARQLUpdateStore(
        query_endpoint=kg_address, update_endpoint=_update_endpoint(kg_address),
        context_aware=False,
    )
    graph = Graph(store=store)
    graph.update(build_subproperty_update())
