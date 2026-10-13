import unittest
from queue import Empty, Queue
from typing import List, Optional

from cltl.combot.event.emissor import ScenarioStarted, ScenarioStopped, TextSignalEvent
from cltl.combot.infra.event import Event
from cltl.combot.infra.event.memory import SynchronousEventBus
from cltl.combot.infra.time_util import timestamp_now
from emissor.representation.scenario import Modality, Scenario, TextSignal

from cltl.diabetes.api import Conversation, ConversationFactory
from cltl.diabetes.service import DiabetesService, ERROR_REPLY

INPUT_TOPIC = "inputTopic"
OUTPUT_TOPIC = "outputTopic"
SCENARIO_TOPIC = "scenarioTopic"


class FakeConversation(Conversation):
    def __init__(self, human: str):
        self.human = human
        self.heard: List[str] = []
        self.closed = False

    def open(self) -> Optional[str]:
        return f"Hi {self.human}, it has been 3 days."

    def respond(self, utterance: str) -> Optional[str]:
        if utterance == "boom":
            raise RuntimeError("boom")
        self.heard.append(utterance)
        return f"About {utterance}?"

    def close(self) -> None:
        self.closed = True


class FakeLtm:
    def __init__(self):
        self.contexts: List[dict] = []

    def capsule_context(self, capsule: dict) -> dict:
        self.contexts.append(capsule)
        return {"response": 204, "context": capsule}


class FakeFactory(ConversationFactory):
    def __init__(self, fail_times: int = 0):
        self.created: List[FakeConversation] = []
        self.ltms = []
        self.fail_times = fail_times

    def create(self, scenario_id: str, human: str, ltm) -> Conversation:
        self.ltms.append(ltm)
        if self.fail_times:
            self.fail_times -= 1
            raise ConnectionError("knowledge graph down")
        conversation = FakeConversation(human)
        self.created.append(conversation)
        return conversation


def _utterance(text: str, scenario_id: str = "scenario-1") -> Event:
    signal = TextSignal.for_scenario(scenario_id, timestamp_now(), timestamp_now(), None, text)
    return Event.for_payload(TextSignalEvent.for_speaker(signal))


def _scenario(event_cls, scenario_id: str = "scenario-1") -> Event:
    scenario = Scenario.new_instance(scenario_id, timestamp_now(), None, None, [Modality.TEXT])
    return Event.for_payload(event_cls.create(scenario))


class DiabetesServiceTest(unittest.TestCase):
    def setUp(self):
        self.event_bus = SynchronousEventBus()
        self.replies = Queue()
        self.ltm = FakeLtm()
        self.event_bus.subscribe(OUTPUT_TOPIC, self.replies.put)
        self.service = None

    def tearDown(self):
        if self.service:
            self.service.stop()

    def _start(self, factory, **kwargs):
        self.service = DiabetesService(INPUT_TOPIC, OUTPUT_TOPIC, SCENARIO_TOPIC, "Jan", factory,
                                       self.ltm, self.event_bus, resource_manager=None, **kwargs)
        self.service.start()

    def _reply_text(self):
        return self.replies.get(timeout=1).payload.signal.text

    def test_first_utterance_opens_with_catch_up(self):
        factory = FakeFactory()
        self._start(factory)

        self.event_bus.publish(INPUT_TOPIC, _utterance("yes"))
        self.assertEqual("Hi Jan, it has been 3 days.", self._reply_text())
        self.assertEqual([], factory.created[0].heard)

        self.event_bus.publish(INPUT_TOPIC, _utterance("I went for a walk"))
        self.assertEqual("About I went for a walk?", self._reply_text())
        self.assertEqual(1, len(factory.created))

    def test_reply_is_for_the_agent_and_the_scenario(self):
        self._start(FakeFactory())

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi", "scenario-x"))
        reply = self.replies.get(timeout=1)

        self.assertEqual("TextSignalEvent", reply.payload.type)
        self.assertEqual("scenario-x", reply.payload.signal.time.container_id)
        self.assertEqual("scenario-x", reply.metadata.scenario_id)
        labels = [a.value for m in reply.payload.signal.mentions for a in m.annotations]
        self.assertIn("LEOLANI", labels)

    def test_scenario_start_opens_when_configured(self):
        self._start(FakeFactory(), open_on="scenario")

        self.event_bus.publish(SCENARIO_TOPIC, _scenario(ScenarioStarted))
        self.assertEqual("Hi Jan, it has been 3 days.", self._reply_text())

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi"))
        self.assertEqual("About hi?", self._reply_text())

    def test_scenario_start_does_not_open_by_default(self):
        self._start(FakeFactory())

        self.event_bus.publish(SCENARIO_TOPIC, _scenario(ScenarioStarted))
        self.assertRaises(Empty, lambda: self.replies.get(timeout=0.05))

    def test_context_capsule_parses_location_record(self):
        from types import SimpleNamespace
        record = ('{"ip": "130.37.159.198", "city": "Amsterdam", "region": "North Holland", '
                  '"country": "NL", "loc": "52.3740,4.8897"}')
        payload = SimpleNamespace(scenario=SimpleNamespace(
            start=None, context=SimpleNamespace(location=record, location_id="loc-1")))

        capsule = DiabetesService._context_capsule("scenario-1", payload)

        self.assertEqual("", capsule["place"])
        self.assertEqual("loc-1", capsule["place_id"])
        self.assertEqual(("Amsterdam", "North Holland", "NL"),
                         (capsule["city"], capsule["region"], capsule["country"]))

    def test_scenario_start_adds_context(self):
        factory = FakeFactory()
        self._start(factory)

        self.event_bus.publish(SCENARIO_TOPIC, _scenario(ScenarioStarted))
        self._wait_for(lambda: self.ltm.contexts)

        capsule = self.ltm.contexts[0]
        self.assertEqual("scenario-1", capsule["context_id"])
        self.assertEqual("Unknown", capsule["city"])

        # A repeated start of the same scenario adds nothing; a new scenario
        # gets its own context.
        self.event_bus.publish(SCENARIO_TOPIC, _scenario(ScenarioStarted))
        self.event_bus.publish(SCENARIO_TOPIC, _scenario(ScenarioStarted, "scenario-2"))
        # Events are processed in order: once this is answered, both
        # ScenarioStarted events have been handled too.
        self.event_bus.publish(INPUT_TOPIC, _utterance("hi", "scenario-2"))
        self._reply_text()
        self.assertEqual(["scenario-1", "scenario-2"], [c["context_id"] for c in self.ltm.contexts])
        # The conversation got the service's one LongTermMemory.
        self.assertIs(self.ltm, factory.ltms[0])

    def test_scenario_stop_closes_and_next_utterance_reopens(self):
        factory = FakeFactory()
        self._start(factory)

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi"))
        self._reply_text()
        self.event_bus.publish(SCENARIO_TOPIC, _scenario(ScenarioStopped))

        self._wait_for(lambda: factory.created[0].closed)

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi again"))
        self.assertEqual("Hi Jan, it has been 3 days.", self._reply_text())
        self.assertEqual(2, len(factory.created))

    def test_separate_conversation_per_scenario(self):
        factory = FakeFactory()
        self._start(factory)

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi", "scenario-1"))
        self._reply_text()
        self.event_bus.publish(INPUT_TOPIC, _utterance("hi", "scenario-2"))
        self._reply_text()

        self.assertEqual(2, len(factory.created))

    def test_failed_start_is_reported_and_retried(self):
        factory = FakeFactory(fail_times=1)
        self._start(factory)

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi"))
        self.assertEqual(ERROR_REPLY, self._reply_text())

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi"))
        self.assertEqual("Hi Jan, it has been 3 days.", self._reply_text())

    def test_failed_reply_is_reported_and_conversation_continues(self):
        self._start(FakeFactory())

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi"))
        self._reply_text()
        self.event_bus.publish(INPUT_TOPIC, _utterance("boom"))
        self.assertEqual(ERROR_REPLY, self._reply_text())
        self.event_bus.publish(INPUT_TOPIC, _utterance("walk"))
        self.assertEqual("About walk?", self._reply_text())

    def test_stop_closes_open_conversations(self):
        factory = FakeFactory()
        self._start(factory)

        self.event_bus.publish(INPUT_TOPIC, _utterance("hi"))
        self._reply_text()
        self.service.stop()
        self.service = None

        self.assertTrue(factory.created[0].closed)

    def test_rejects_unknown_open_on(self):
        with self.assertRaises(ValueError):
            DiabetesService(INPUT_TOPIC, OUTPUT_TOPIC, SCENARIO_TOPIC, "Jan", FakeFactory(),
                            self.ltm, self.event_bus, resource_manager=None, open_on="never")

    @staticmethod
    def _wait_for(condition, timeout=1.0):
        import time
        deadline = time.time() + timeout
        while not condition():
            if time.time() > deadline:
                raise AssertionError("condition not met in time")
            time.sleep(0.01)
