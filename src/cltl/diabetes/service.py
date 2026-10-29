import ast
import json
import logging
from datetime import date, datetime
from typing import Dict, Optional, Set

from cltl.combot.event.emissor import TextSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.infra.event.util import extract_scenario_id
from cltl.combot.infra.resource import ResourceManager
from cltl.combot.infra.time_util import timestamp_now
from cltl.combot.infra.topic_worker import TopicWorker
from emissor.representation.scenario import TextSignal

from cltl.diabetes.api import Conversation, ConversationFactory

logger = logging.getLogger(__name__)

# Every reply takes several LLM and knowledge graph round trips. The
# TopicWorker default (buffer_size=1, OVERWRITE) would drop an utterance that
# arrives while the previous one is still being answered.
BUFFER_SIZE = 64

OPEN_ON_UTTERANCE = "utterance"
OPEN_ON_SCENARIO = "scenario"

SCENARIO_STARTED = "ScenarioStarted"
SCENARIO_STOPPED = "ScenarioStopped"

ERROR_REPLY = "Sorry, something went wrong on my side. Could you say that again?"


class DiabetesService:
    """Drives the conversation of one tenant with a knowledge graph driven
    lifestyle coach (see catchup.py).

    One :class:`Conversation` per scenario. It is created when the scenario
    starts (`open_on: scenario`) or at the first utterance in it
    (`open_on: utterance`), and opens with the gap in time since the last
    conversation and the topics discussed back then. Every further utterance
    is answered by that conversation. When the scenario stops, its logs are
    saved.

    `open_on: utterance` is the default because the platform's context opens
    the scenario when its container starts, typically before anyone has the
    chat UI open, and because it does not race the platform's own greeting.
    """

    @classmethod
    def from_config(cls, factory: ConversationFactory, ltm, event_bus: EventBus,
                    resource_manager: ResourceManager, config_manager: ConfigurationManager):
        config = config_manager.get_config("cltl.diabetes")

        input_topic = config.get("topic_input")
        output_topic = config.get("topic_output")
        # Where the GraphDB URL visualising the triples extracted from an
        # utterance is published; not published when empty.
        graph_topic = config.get("graph_output") if "graph_output" in config else None
        scenario_topic = config.get("topic_scenario")
        human = config.get("human")
        open_on = config.get("open_on") if "open_on" in config else OPEN_ON_UTTERANCE
        use_scenario_speaker = (config.get_boolean("use_scenario_speaker")
                                if "use_scenario_speaker" in config else False)

        tenant = None
        if "cltl.event.kombu" in config_manager:
            kombu_config = config_manager.get_config("cltl.event.kombu")
            tenant = kombu_config.get("tenant") if "tenant" in kombu_config else None
        implementation = (config_manager.get_config("cltl.event").get("implementation")
                          if "cltl.event" in config_manager else None)
        if implementation != "internal" and not tenant:
            logger.warning("[cltl.event.kombu] tenant is empty: this module answers EVERY tenant "
                           "on the exchange. Set CLTL_TENANT to scope it to one tenant.")

        return cls(input_topic, output_topic, scenario_topic, human, factory, ltm, event_bus,
                   resource_manager, open_on=open_on, use_scenario_speaker=use_scenario_speaker,
                   graph_topic=graph_topic or None)

    def __init__(self, input_topic: str, output_topic: str, scenario_topic: str, human: str,
                 factory: ConversationFactory, ltm, event_bus: EventBus, resource_manager: ResourceManager,
                 open_on: str = OPEN_ON_UTTERANCE, use_scenario_speaker: bool = False,
                 graph_topic: Optional[str] = None):
        if open_on not in (OPEN_ON_UTTERANCE, OPEN_ON_SCENARIO):
            raise ValueError(f"[cltl.diabetes] open_on must be '{OPEN_ON_UTTERANCE}' "
                             f"or '{OPEN_ON_SCENARIO}', got {open_on!r}")
        if not human:
            raise ValueError("[cltl.diabetes] human must name the person the agent talks to")

        self._input_topic = input_topic
        self._output_topic = output_topic
        self._scenario_topic = scenario_topic
        self._graph_topic = graph_topic
        self._human = human
        self._open_on = open_on
        self._use_scenario_speaker = use_scenario_speaker

        self._factory = factory
        # The one LongTermMemory (cltl.brain) for the knowledge graph, shared
        # by every conversation.
        self._ltm = ltm
        self._event_bus = event_bus
        self._resource_manager = resource_manager

        self._conversations: Dict[str, Conversation] = {}
        self._speakers: Dict[str, str] = {}
        self._contexts_added: Set[str] = set()
        self._topic_worker = None

    @property
    def app(self):
        return None

    def start(self, timeout=30):
        self._topic_worker = TopicWorker([self._input_topic, self._scenario_topic], self._event_bus,
                                         provides=[t for t in (self._output_topic, self._graph_topic) if t],
                                         resource_manager=self._resource_manager,
                                         processor=self._process, buffer_size=BUFFER_SIZE,
                                         name=self.__class__.__name__)
        self._topic_worker.start().wait()

    def stop(self):
        if self._topic_worker:
            self._topic_worker.stop()
            self._topic_worker.await_stop()
            self._topic_worker = None

        for scenario_id in list(self._conversations):
            self._close(scenario_id)

    def _process(self, event: Event):
        if event.metadata.topic == self._scenario_topic:
            self._process_scenario(event)
        elif event.metadata.topic == self._input_topic:
            self._process_utterance(event)
        else:
            raise ValueError("Unexpected topic " + event.metadata.topic)

    def _process_scenario(self, event: Event):
        scenario_id = extract_scenario_id(event, on_missing=None)
        if not scenario_id:
            logger.warning("Scenario event %s carries no scenario id, ignored", event.id)
            return

        event_type = getattr(event.payload, "type", None)
        if event_type == SCENARIO_STARTED:
            speaker = self._speaker_name(event.payload)
            self._add_context(scenario_id, event.payload)
            if speaker:
                self._speakers[scenario_id] = speaker
            if self._open_on == OPEN_ON_SCENARIO and scenario_id not in self._conversations:
                self._open(scenario_id, event)
        elif event_type == SCENARIO_STOPPED:
            self._close(scenario_id)
            self._speakers.pop(scenario_id, None)

    def _process_utterance(self, event: Event[TextSignalEvent]):
        scenario_id = extract_scenario_id(event)
        if scenario_id not in self._conversations:
            # The first utterance only opens the conversation; its content
            # is not answered separately (typically "hi" or the "yes" to the
            # platform's greeting).
            self._open(scenario_id, event)
            return

        text = event.payload.signal.text
        if not text or not text.strip():
            return

        try:
            response, graph_uri = self._conversations[scenario_id].respond(text)
        except Exception:
            logger.exception("Failed to respond to %r in scenario %s", text, scenario_id)
            response, graph_uri = ERROR_REPLY, None

        self._publish(response, scenario_id, event)
        if graph_uri and self._graph_topic:
            self._publish(graph_uri, scenario_id, event, topic=self._graph_topic)

    def _open(self, scenario_id: str, event: Event):
        human = self._speakers.get(scenario_id, self._human)
        try:
            conversation = self._factory.create(scenario_id, human, self._ltm)
            opening = conversation.open()
        except Exception:
            # Not stored, so the next utterance tries again, e.g. once the
            # knowledge graph is reachable.
            logger.exception("Failed to start a conversation with %s in scenario %s", human, scenario_id)
            self._publish(ERROR_REPLY, scenario_id, event)
            return

        self._conversations[scenario_id] = conversation
        logger.info("Started conversation with %s in scenario %s", human, scenario_id)
        self._publish(opening, scenario_id, event)

    def _close(self, scenario_id: str):
        conversation = self._conversations.pop(scenario_id, None)
        if not conversation:
            return

        try:
            conversation.close()
            logger.info("Closed conversation in scenario %s", scenario_id)
        except Exception:
            logger.exception("Failed to save the logs of the conversation in scenario %s", scenario_id)

    def _speaker_name(self, payload) -> Optional[str]:
        if not self._use_scenario_speaker:
            return None

        context = getattr(getattr(payload, "scenario", None), "context", None)
        speaker = getattr(context, "speaker", None)
        name = getattr(speaker, "name", None)

        return name.strip() if name and name.strip() else None

    def _publish(self, response: Optional[str], scenario_id: str, source: Event,
                 topic: Optional[str] = None):
        if not response:
            return

        signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, response)
        payload = TextSignalEvent.for_agent(signal)
        # source=event copies the tenant onto the reply; without it the reply
        # is routed to no tenant at all. for_scenario_payload also sets the
        # scenario id, which a scenario event does not carry in its metadata.
        self._event_bus.publish(topic or self._output_topic,
                                Event.for_scenario_payload(scenario_id, payload, source=source))

    def _add_context(self, scenario_id: str, payload):
        """Once per scenario: its context is the one the chat and the
        perceptions of the scenario share. A repeated ScenarioStarted for the
        same scenario is ignored; a failed attempt is retried at the next one."""
        if scenario_id in self._contexts_added:
            return

        try:
            capsule = self._context_capsule(scenario_id, payload)
            response = self._ltm.capsule_context(capsule)
            self._contexts_added.add(scenario_id)
            logger.info("Added context %s to the knowledge graph (response %s)",
                        scenario_id, response.get("response"))
        except Exception:
            # The conversation does not depend on the context, so carry on.
            logger.exception("Failed to add the context of scenario %s to the knowledge graph", scenario_id)

    @staticmethod
    def _context_capsule(scenario_id: str, payload) -> dict:
        """The capsule for LongTermMemory.capsule_context from the scenario's
        start time and the location in its context, if any."""
        scenario = getattr(payload, "scenario", None)
        start = getattr(scenario, "start", None)
        context = getattr(scenario, "context", None)
        location = _parse_location(_field(context, "location"))
        # A location that is not a record (e.g. "Piek's office") is the place itself.
        place = location if isinstance(location, str) else _field(location, "place")

        return {"context_id": scenario_id,
                "date": datetime.fromtimestamp(start / 1000).date() if start else date.today(),
                "place": place or "",
                "place_id": _field(context, "location_id"),
                "country": _field(location, "country") or "Unknown",
                "region": _field(location, "region") or "Unknown",
                "city": _field(location, "city") or "Unknown"}


def _parse_location(location):
    """The platform sends the location as a string holding a record (ipinfo.io's:
    ip, city, region, country, loc, ...), as JSON or as a Python dict."""
    if not isinstance(location, str):
        return location
    for parse in (json.loads, ast.literal_eval):
        try:
            parsed = parse(location)
        except (ValueError, SyntaxError):
            continue
        if isinstance(parsed, dict):
            return parsed

    return location


def _field(obj, name: str):
    """A field of a dict or an object; deserialized events can be either."""
    if obj is None or isinstance(obj, str):
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)
