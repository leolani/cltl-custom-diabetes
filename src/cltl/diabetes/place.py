"""Where the human is during a scenario, kept as the place of the scenario's
context in the knowledge graph.

The platform only knows the city (from the IP address), so the service writes
the place of a new scenario as "unknown". At the start of the conversation it
is compared with the last encounter with the human: the city of its context
and what the camera saw there (the scene types and objects of its visual
detections) against the city and the visuals of the current scenario.

- same city, visuals overlap at least `same_threshold`: the same place as last
  time; it is saved right away and only mentioned in the introduction.
- same city, overlap between `ask_threshold` and `same_threshold`, or no
  visuals to compare: the introduction asks whether they are there again.
- otherwise (overlap below `ask_threshold`, another city, no earlier place):
  the introduction asks where they are.

Places are named as uniquely as possible ("Jan's home in Amsterdam" rather
than "home"), as the place node is identified by its name.
"""
import logging
from dataclasses import dataclass
from typing import Callable, List, Optional, Set

from cltl.commons.casefolding import casefold_text

logger = logging.getLogger(__name__)

KNOWN = "known"
SAME = "same"
CONFIRM = "confirm"
ASK = "ask"

DEFAULT_SAME_THRESHOLD = 0.5
DEFAULT_ASK_THRESHOLD = 0.2

_PREFIXES = """PREFIX sem: <http://semanticweb.cs.vu.nl/2009/11/sem/>
    PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
    PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
    PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
    PREFIX prov: <http://www.w3.org/ns/prov#>
    PREFIX gaf: <http://groundedannotationframework.org/gaf#>
    PREFIX grasp: <http://groundedannotationframework.org/grasp#>
    PREFIX eps: <http://cltl.nl/episodicawareness/>
    PREFIX n2mu: <http://cltl.nl/leolani/n2mu/>
    PREFIX leolaniTalk: <http://cltl.nl/leolani/talk/>"""


@dataclass
class PlaceDecision:
    kind: str
    # The place of the last encounter, for SAME and CONFIRM.
    place_uri: Optional[str] = None
    name: Optional[str] = None
    overlap: Optional[float] = None


def classify(same_city: bool, overlap: Optional[float], same_threshold: float, ask_threshold: float) -> str:
    """SAME, CONFIRM or ASK for a current scenario whose place is not known yet,
    given the last encounter's place. `overlap` is None when there are no
    visuals to compare (e.g. no camera), which leaves only the city to go on."""
    if not same_city:
        return ASK
    if overlap is None:
        return CONFIRM
    if overlap >= same_threshold:
        return SAME
    if overlap >= ask_threshold:
        return CONFIRM
    return ASK


def overlap(visuals: Set[str], other: Set[str]) -> Optional[float]:
    """Jaccard overlap of two sets of visuals, None if either is empty."""
    if not visuals or not other:
        return None
    return len(visuals & other) / len(visuals | other)


class PlaceResolver:
    def __init__(self, ltm, human: str, llm: Callable[[List[dict]], str],
                 same_threshold: float = DEFAULT_SAME_THRESHOLD, ask_threshold: float = DEFAULT_ASK_THRESHOLD):
        self._ltm = ltm
        self._human = human
        self._llm = llm
        self._same_threshold = same_threshold
        self._ask_threshold = ask_threshold

    def decide(self, scenario_id: str) -> PlaceDecision:
        context = self.context_uri(scenario_id)
        if self._named_place(context):
            return PlaceDecision(KNOWN)

        previous = self._previous_encounter(context)
        if not previous or not previous["place"]:
            logger.info("No earlier place of %s to compare scenario %s with", self._human, scenario_id)
            return PlaceDecision(ASK)

        city = self.city(context)
        same_city = bool(city) and city == previous["city"]
        visual_overlap = overlap(self.visuals(context), self.visuals(previous["context"]))
        kind = classify(same_city, visual_overlap, self._same_threshold, self._ask_threshold)
        logger.info("Place of scenario %s: %s (last time %s in %s, now in %s, visual overlap %s)",
                    scenario_id, kind, previous["name"], previous["city"], city, visual_overlap)

        return PlaceDecision(kind, previous["place"], previous["name"], visual_overlap)

    def context_uri(self, scenario_id: str) -> str:
        return str(self._ltm._rdf_builder.create_resource_uri("LC", f"context{scenario_id}"))

    def city(self, context: str) -> Optional[str]:
        rows = self._select(f"""SELECT ?city WHERE {{ <{context}> sem:hasPlace ?place .
            ?place n2mu:in ?city . ?city rdf:type n2mu:city }} LIMIT 1""")
        return rows[0]["city"] if rows else None

    def visuals(self, context: str) -> Set[str]:
        """What the camera saw in `context`: the scene types (e.g. "scene:office")
        and the objects in them (e.g. "chair", "desk")."""
        rows = self._select(f"""SELECT DISTINCT ?scene ?object WHERE {{
            <{context}> sem:hasEvent ?visual . ?visual rdf:type grasp:Visual ; sem:hasSubEvent ?detection .
            ?mention prov:wasDerivedFrom ?detection ; gaf:denotes ?claim .
            GRAPH ?claim {{ ?scene sem:hasActor ?object }} }}""")
        visuals = set()
        for row in rows:
            visuals.add(_local_name(row["object"]))
            # The scene is <type>_<scenario id>, e.g. office_1af5d56f-...
            visuals.add("scene:" + _local_name(row["scene"]).split("_")[0])

        return visuals

    def name_place(self, utterance: str, city_name: Optional[str]) -> Optional[str]:
        """A name, as unique as possible, for where the human says they are."""
        return self._ask_llm(f"{self._human} was asked where they are right now. " + self._naming_rules(city_name),
                             utterance)

    def resolve_confirmation(self, utterance: str, name: str, city_name: Optional[str]) -> Optional[str]:
        """SAME if the human confirms being at `name`, a new name if they say
        they are somewhere else, None if it stays unclear."""
        return self._ask_llm(
            f"{self._human} was asked whether they are at {name} again. If they confirm, reply with "
            "exactly SAME. If they say they are somewhere else and say where, reply with a name for "
            "that place instead. " + self._naming_rules(city_name),
            utterance)

    def save_named_place(self, scenario_id: str, name: str) -> str:
        label = casefold_text(name, format="triple")
        place = str(self._ltm._rdf_builder.create_resource_uri("LC", label))
        self.set_place(scenario_id, place, label, name)

        return place

    def set_place(self, scenario_id: str, place: str, label: Optional[str] = None, name: Optional[str] = None):
        """Make `place` the place of the scenario's context, in place of the
        unknown one. A new place gets the city, region and country that one is
        in, and its id."""
        context = self.context_uri(scenario_id)
        new_place = (f"""<{place}> rdf:type n2mu:location, sem:Place ; rdfs:label {_literal(label)} ;
                         skos:prefLabel {_literal(name)} .""" if label else "")
        self._update(f"""INSERT {{ GRAPH leolaniTalk:Interactions {{
                <{context}> sem:hasPlace <{place}> . {new_place}
                <{place}> n2mu:in ?in . <{place}> n2mu:id ?id . }} }}
            WHERE {{ OPTIONAL {{ GRAPH leolaniTalk:Interactions {{ <{context}> sem:hasPlace ?old .
                FILTER(?old != <{place}>)
                OPTIONAL {{ ?old n2mu:in ?in . FILTER NOT EXISTS {{ <{place}> n2mu:in ?known_in }} }}
                OPTIONAL {{ ?old n2mu:id ?id . FILTER NOT EXISTS {{ <{place}> n2mu:id ?known_id }} }} }} }} }}""")
        # Then drop the link to the old place, and the old place if it was unknown.
        self._update(f"""DELETE {{ GRAPH leolaniTalk:Interactions {{ <{context}> sem:hasPlace ?old . ?unknown ?p ?o . }} }}
            WHERE {{ GRAPH leolaniTalk:Interactions {{ <{context}> sem:hasPlace ?old . FILTER(?old != <{place}>)
                OPTIONAL {{ ?old rdfs:label ?old_label . FILTER(LCASE(STR(?old_label)) = "unknown")
                           ?old ?p ?o . BIND(?old AS ?unknown) }} }} }}""")
        logger.info("Place of scenario %s is now %s", scenario_id, place)

    def city_name(self, city: Optional[str]) -> Optional[str]:
        return _local_name(city).replace("_", " ") if city else None

    def _named_place(self, context: str) -> bool:
        return self._ltm._submit_query(f"""{_PREFIXES}
            ASK {{ <{context}> sem:hasPlace ?place . ?place rdfs:label ?label .
                   FILTER(LCASE(STR(?label)) != "unknown") }}""", ask=True)

    def _previous_encounter(self, context: str) -> Optional[dict]:
        """The context of the last conversation with the human before this one
        (chat ids are timestamps), with its place, if it has a named one."""
        rows = self._select(f"""SELECT ?context (MAX(xsd:integer(STR(?chat_id))) AS ?last) WHERE {{
                GRAPH leolaniTalk:Interactions {{ ?context rdf:type eps:Context ; sem:hasEvent ?chat .
                    ?chat rdf:type grasp:Chat ; n2mu:id ?chat_id ; sem:hasSubEvent ?utterance .
                    ?utterance sem:hasActor ?actor . }}
                ?actor rdfs:label {_literal(self._human)} .
                FILTER(?context != <{context}>) }}
            GROUP BY ?context ORDER BY DESC(?last) LIMIT 1""")
        if not rows:
            return None

        previous = rows[0]["context"]
        places = self._select(f"""SELECT ?place ?label ?name WHERE {{ <{previous}> sem:hasPlace ?place .
                ?place rdfs:label ?label . FILTER(LCASE(STR(?label)) != "unknown")
                OPTIONAL {{ ?place skos:prefLabel ?name }} }} LIMIT 1""")
        place = places[0] if places else {}

        return {"context": previous, "place": place.get("place"), "city": self.city(previous),
                "name": place.get("name") or (place.get("label") or "").replace("-", " ")}

    def _naming_rules(self, city_name: Optional[str]) -> str:
        town = f"otherwise add \"in {city_name}\"" if city_name else "if they said it"
        return ("Name the place so that it is as unique as possible: use their name instead of "
                f"my/our (e.g. \"{self._human}'s home\", \"{self._human}'s daughter's house\") and add "
                f"the town they said, {town} (e.g. \"{self._human}'s home in Weesp\"). Reply with only "
                "the name. If it stays unclear where they are, reply with exactly UNKNOWN.")

    def _ask_llm(self, instruction: str, utterance: str) -> Optional[str]:
        answer = self._llm([{"role": "system", "content": instruction}, {"role": "user", "content": utterance}])
        answer = (answer or "").strip().strip(".\"'")

        return None if not answer or answer.upper() == "UNKNOWN" else answer

    def _select(self, query: str) -> List[dict]:
        rows = self._ltm._submit_query(f"{_PREFIXES}\n{query}")
        return [{key: value["value"] for key, value in row.items()} for row in rows]

    def _update(self, query: str):
        self._ltm._submit_query(f"{_PREFIXES}\n{query}", post=True)


def _local_name(uri: str) -> str:
    return uri.rstrip("/").rsplit("/", 1)[-1].rsplit("#", 1)[-1]


def _literal(text: Optional[str]) -> str:
    return '"' + (text or "").replace("\\", "\\\\").replace('"', '\\"') + '"'
