
import json
import zipfile

import requests
from cltl.brain.long_term_memory import LongTermMemory
from cltl.commons.discrete import UtteranceType
from cltl.brain.infrastructure.api import Perspective, Triple, RDFBase, Entity, Predicate
from tqdm import tqdm
from random import getrandbits
from pathlib import Path
import events_to_capsules
from dateutil import parser
from datetime import datetime
from enum import Enum
#from perspective.emotion_extraction import GoEmotionDetector

# Expects only SRL input and uses the context to identify the activities. A separate emotion detection is used to infer a perspective
def get_scenarios_from_srl_annotations_identifying_activities_in_context_and_inferring_perspectives(annotated_conversations):
    ## requires a sentiment/emotion detection module
    model_path = "AnasAlokla/multilingual_go_emotions"
    #  Languages: Arabic, English, French, Spanish, Dutch, Turkish
 #   emotion_detector = GoEmotionDetector(model=model_path)
    emotion_detector = None
    # Define contextual features
    place_id = getrandbits(8)
    location = requests.get("https://ipinfo.io").json()

    scenarios = []
    for conversation in annotated_conversations:
        if len(conversation) > 0:
            scenario_context = {"context_id": conversation[0]['chat'],
                                "date": parser.parse(conversation[0]['date']),
                                "place": "Piek's office",
                                "place_id": place_id,
                                "country": location['country'],
                                "region": location['region'],
                                "city": location['city']}
            capsules = []
            conversational_context= {}
            for turn in conversation:
                turn_capsule = events_to_capsules.get_capsule_with_event_details_from_turn_with_conversationa_context_similarity_match_and_time(conversational_context=conversational_context, turn_data=turn, emotion_detector=emotion_detector)
                if turn_capsule:
                    capsules.append(turn_capsule)
            if capsules:
                scenario = (scenario_context, capsules)
                scenarios.append(scenario)
        ###break
    return scenarios

# The activities are already identified by the LLM when processing the conversations
def get_scenarios_from_srl_annotations(annotated_conversations):
    # Define dummy contextual features
    place_id = getrandbits(8)
    location = requests.get("https://ipinfo.io").json()

    scenarios = []
    for conversation in annotated_conversations:
        if len(conversation) > 0:
            scenario_context = {"context_id": conversation[0]['chat'],
                                "date": parser.parse(conversation[0]['date']),
                                "place": "Piek's office",
                                "place_id": place_id,
                                "country": location['country'],
                                "region": location['region'],
                                "city": location['city']}
            capsules = []
            for turn in conversation:
                ### activities and conditions are identified by activity_id per conversation
                ### because there can be multiple activities in the same turn, we get a list of capsules
                turn_capsules = events_to_capsules.get_capsule_with_event_details_from_turn_with_activity_id(turn_data=turn)
                if turn_capsules:
                    capsules.extend(turn_capsules)
            if capsules:
                scenario = (scenario_context, capsules)
                scenarios.append(scenario)
       # break
    return scenarios


def _serialize_enum(val):
    """Serialize a Perspective field: handles None, single Enum, or list of Enum/scalar values."""
    if val is None:
        return None
    if isinstance(val, list):
        return [e.name if isinstance(e, Enum) else e for e in val]
    if isinstance(val, Enum):
        return val.name
    return val


class CapsuleEncoder(json.JSONEncoder):
    """Serializes capsule objects: Perspective/Triple/RDFBase → dict, Enum → name, datetime → ISO."""

    def _preprocess(self, obj):
        """Walk the object graph and convert all non-standard types before encoding.

        Enum subclasses that also inherit str/int are serialized by Python's JSON
        encoder as their raw value, bypassing default(). Preprocessing fixes that.
        """
        if isinstance(obj, Perspective):
            return {
                "certainty": _serialize_enum(obj.certainty),
                "polarity": _serialize_enum(obj.polarity),
                "sentiment": _serialize_enum(obj.sentiment),
                "emotion": _serialize_enum(obj.emotion),
                "level": _serialize_enum(obj.level),
            }
        if isinstance(obj, Triple):
            return {
                "subject": self._preprocess(obj.subject),
                "predicate": self._preprocess(obj.predicate),
                "complement": self._preprocess(obj.complement),
            }
        if isinstance(obj, Entity):
            return {"id": str(obj.id), "label": str(obj.label), "types": obj.types}
        if isinstance(obj, Predicate):
            return {"id": str(obj.id), "label": str(obj.label)}
        if isinstance(obj, RDFBase):
            return {"id": str(obj.id), "label": str(obj.label)}
        if isinstance(obj, Enum):
            return obj.name
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, dict):
            return {k: self._preprocess(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [self._preprocess(item) for item in obj]
        return obj

    def encode(self, obj):
        return super().encode(self._preprocess(obj))

    def iterencode(self, obj, _one_shot=False):
        return super().iterencode(self._preprocess(obj), _one_shot)

def populate_ekg_from_annotations(annotated_conversations, kg_address, log_dir, clear_all=False,
                                   capsules_out_path=None):
    """Push already-loaded annotation JSON (the annotation tool's export format) into a knowledge graph.

    :param annotated_conversations: array-of-conversations JSON, as produced by the annotation
        tool's "Export JSON" / "Push to Knowledge Graph" and consumed by ``get_scenarios_from_srl_annotations``.
    :param kg_address: SPARQL repository address of the target knowledge graph (e.g.
        ``http://localhost:7200/repositories/event_sandbox``).
    :param log_dir: directory to write step-wise graph logs to.
    :param clear_all: whether to start from an empty brain instead of appending to the existing graph.
    :param capsules_out_path: optional path to also dump the derived capsules as JSON, for debugging.
    :return: dict summarizing how many conversations/capsules were added.
    """
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    brain = LongTermMemory(address=kg_address,  # Location to save accumulated graph
                           log_dir=log_dir,  # Location to save step-wise graphs
                           clear_all=clear_all)

    ### A scenario has a context_capsule that identifies the scenario and a list of capsules extracted for a single conversation that need to be added to the brain.
    # The context capsule contains contextual information about the scenario (e.g. location, date) and
    # the capsules contain the information that needs to be added to the brain (e.g. triples, event details)
    scenarios = get_scenarios_from_srl_annotations(annotated_conversations)
    print('Total nr of scenarios', len(scenarios))

    if capsules_out_path:
        with open(capsules_out_path, "w") as f:
            json.dump(scenarios, f, indent=4, cls=CapsuleEncoder)

    total_capsules = 0
    # Loop through the scenarios
    for (context_capsule, conversation_capsules) in tqdm(scenarios):
        print('Conversation id', context_capsule['context_id'], 'Total number of capsules extracted for this conversation', len(conversation_capsules))        # Create context
        brain.capsule_context(context_capsule)
        # Add information to the brain
        for capsule in conversation_capsules:
            print('chat', capsule['chat'], 'out of ', len(scenarios), 'turn', capsule['turn'], 'out of', len(conversation_capsules), 'turns')
           # brain.capsule_statement(capsule, reason_types=True, return_thoughts=False, create_label=True)
            brain.capsule_event(capsule, reason_types=True, return_thoughts=False, create_label=True)
            total_capsules += 1

    return {"conversations": len(scenarios), "capsules": total_capsules}


def main():
    INPUT_ZIP = Path("../../data/event_srl.json.zip")

    scenario_filepath = Path('../../data/')
    graph_filepath = scenario_filepath / Path('graph/')

    ## Input is a JSON file that has the conversations, the meta data and the SRL results on a turn by turn basis
    print(f"Loading {INPUT_ZIP} …")
    with zipfile.ZipFile(INPUT_ZIP) as z:
        inner = [n for n in z.namelist() if n.endswith(".json")][0]
        with z.open(inner) as f:
            annotated_conversations = json.load(f)

    print('Total number of annotated conversations', len(annotated_conversations))
    print(annotated_conversations[0])

    populate_ekg_from_annotations(
        annotated_conversations,
        kg_address="http://localhost:7200/repositories/event_sandbox",
        log_dir=graph_filepath,
        clear_all=True,  # To start from an empty brain
        capsules_out_path=scenario_filepath / "capsules_with_event_details.json",
    )

if __name__ == '__main__':
    main()