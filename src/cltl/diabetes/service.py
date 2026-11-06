import ast
import base64
import html
import io
import json
import logging
import re
import tempfile
import webbrowser
from collections import deque
from datetime import date, datetime
from pathlib import Path
from typing import Deque, Dict, Optional, Set

from _pytest._code import source
from cltl.combot.event.emissor import TextSignalEvent
from cltl.combot.infra.config import ConfigurationManager
from cltl.combot.infra.event import Event, EventBus
from cltl.combot.event.display import DisplayEvent
from cltl.combot.event.status import StatusEvent
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

# Echo filter (see DiabetesService._is_echo()): with speech, the microphone picks up the agent's own
# spoken reply, and answering that makes the agent talk to itself. An utterance of at least
# ECHO_MIN_WORDS words of which at least ECHO_OVERLAP are words of one of the last
# ECHO_REPLIES replies is taken for such an echo and ignored.
ECHO_REPLIES = 3
ECHO_MIN_WORDS = 3
ECHO_OVERLAP = 0.7

# An utterance asking to see the timeline ("show me my timeline", "can I see the timeline?") is
# not answered by the conversation: the patient's timeline (cltl.plots.patient_timeline) is
# rendered from the knowledge graph and opened in a new browser tab.
TIMELINE_REQUEST = re.compile(r"\b(show|display|open|see|view|plot|draw)\b.*\btime\s?line\b"
                              r"|\btime\s?line\b.*\b(please|show|display|open|see|view)\b",
                              re.IGNORECASE)
TIMELINE_DAYS = 60
TIMELINE_REPLY = "Here is your timeline, {human}. I opened it in a new browser tab."
TIMELINE_ERROR_REPLY = "Sorry, I could not make your timeline right now."
TIMELINE_DPI = 110

# The timeline page: the plot with a transparent box over every mark (positioned in % of the
# image, so they keep fitting when it is scaled), a tooltip on hover and the details on click.
# {title} and {subtitle} are HTML-escaped, {marks} is JSON, {png} base64.
TIMELINE_PAGE = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  body { margin: 16px; background: #fcfcfb; color: #0b0b0b; font-family: system-ui, sans-serif; }
  p.sub { color: #52514e; margin-top: -8px; }
  .wrap { display: flex; gap: 16px; align-items: flex-start; flex-wrap: wrap; }
  .plot { position: relative; flex: 1 1 640px; max-width: 1200px; }
  .plot img { width: 100%; display: block; }
  .mark { position: absolute; cursor: pointer; border-radius: 3px; }
  .mark:hover, .mark.on { outline: 2px solid #0b0b0b; background: rgba(11, 11, 11, .08); }
  #tip { position: fixed; pointer-events: none; display: none; white-space: pre-line; max-width: 320px;
         background: #0b0b0b; color: #fcfcfb; font-size: 13px; padding: 6px 8px; border-radius: 4px; }
  #details { flex: 0 1 340px; border: 1px solid #e1e0d9; border-radius: 6px; padding: 12px; background: #fff; }
  #details h3 { margin: 0 0 8px; font-size: 16px; }
  #details h4 { margin: 12px 0 4px; font-size: 13px; color: #52514e; }
  #details pre { white-space: pre-wrap; font: inherit; margin: 0; }
  #details .muted { color: #898781; font-size: 12px; word-break: break-all; }
</style></head>
<body>
<h2>{title}</h2>
<p class="sub">{subtitle}</p>
<div class="wrap">
  <div class="plot" id="plot"><img alt="{title}" src="data:image/png;base64,{png}"></div>
  <div id="details"><span class="muted">Click an event on the timeline to see its details.</span></div>
</div>
<div id="tip"></div>
<script>
const marks = {marks};
const plot = document.getElementById("plot"), tip = document.getElementById("tip"),
      details = document.getElementById("details");
function el(tag, text, cls) {
  const e = document.createElement(tag);
  if (text) e.textContent = text;
  if (cls) e.className = cls;
  return e;
}
function show(mark, box) {
  document.querySelectorAll(".mark.on").forEach(b => b.classList.remove("on"));
  box.classList.add("on");
  details.replaceChildren(el("h3", mark.title), el("pre", mark.summary),
                          el("h4", "Time roles"), el("pre", mark.times.join("\\n") || "none that is a date"),
                          el("h4", "Activity"), el("div", mark.id, "muted"));
  if (mark.graphdb) {
    const a = el("a", "Open in GraphDB \\u2197");
    a.href = mark.graphdb; a.target = "_blank"; a.rel = "noopener";
    const p = el("p"); p.appendChild(a); details.appendChild(p);
  }
}
marks.forEach(mark => {
  const box = el("div", "", "mark");
  [box.style.left, box.style.top, box.style.width, box.style.height] = mark.box.map(v => v + "%");
  box.addEventListener("mousemove", e => {
    tip.textContent = mark.summary; tip.style.display = "block";
    tip.style.left = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8) + "px";
    tip.style.top = (e.clientY + 14) + "px";
  });
  box.addEventListener("mouseleave", () => { tip.style.display = "none"; });
  box.addEventListener("click", () => show(mark, box));
  plot.appendChild(box);
});
</script>
</body></html>
"""


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
        status_topic = config.get("topic_status")
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

        return cls(input_topic, output_topic, scenario_topic, status_topic, human, factory, ltm, event_bus,
                   resource_manager, open_on=open_on, use_scenario_speaker=use_scenario_speaker,
                   graph_topic=graph_topic or None)

    def __init__(self, input_topic: str, output_topic: str, scenario_topic: str, status_topic, human: str,
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
        self._status_topic = status_topic
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
        # The agent's last ECHO_REPLIES replies per scenario, see _is_echo().
        self._recent_replies: Dict[str, Deque[str]] = {}
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
            self._event_bus.publish(self._status_topic, Event.for_scenario_payload(
                scenario_id, StatusEvent("Loading the memory from the KG", "memory", progress=0.5, key="agent.model")))

        elif event_type == SCENARIO_STOPPED:
            self._close(scenario_id)
            self._speakers.pop(scenario_id, None)
            self._event_bus.publish(self._status_topic, Event.for_scenario_payload(
                scenario_id, StatusEvent("This conversation is over", "agent", key="agent.model", resolved=True)))


    def _process_utterance(self, event: Event[TextSignalEvent]):
        scenario_id = extract_scenario_id(event)
        if scenario_id not in self._conversations:
            # The first utterance only opens the conversation; its content
            # is not answered separately (typically "hi" or the "yes" to the
            # platform's greeting).
            self._open(scenario_id, event)
            return

        text = event.payload.signal.text
        if text and self._is_timeline_request(text):
            self._publish(self._show_timeline(scenario_id), scenario_id, event)
            return
        if not text or not text.strip():
            return
        if self._is_echo(scenario_id, text):
            logger.info("Ignored %r in scenario %s: an echo of the agent's own reply", text, scenario_id)
            return

        try:
            ### Update status field
            statusEvent = Event.for_payload(StatusEvent("Started processing the utterance", "NLU", progress=0.5, key="agent.model"),
                                            source=event)
            self._event_bus.publish(self._status_topic, statusEvent)

            response, graph_uri = self._conversations[scenario_id].respond(text)

            logger.info("Response %r in scenario %s", response, scenario_id)
            if graph_uri:
                logger.info("Graph URI %r, for topic %s", graph_uri, self._graph_topic)

            ### Update status field
            statusEvent = Event.for_payload(StatusEvent("Done processing the utterance", "NLU", progress=0.5, key="agent.model"),
                                            source=event)
            self._event_bus.publish(self._status_topic, statusEvent)
        except Exception:
            logger.exception("Failed to respond to %r in scenario %s", text, scenario_id)
            response, graph_uri = ERROR_REPLY, None

        if response:
            self._publish(response, scenario_id, event)

        #### Feedback to UI
        if graph_uri and self._graph_topic:
            ### Update MONITORING field
            graphDisplayEvent = Event.for_payload(DisplayEvent(graph_uri,"Monitoring"), source=event)
            self._event_bus.publish(self._graph_topic, graphDisplayEvent)

            ### Update status field
            statusEvent = Event.for_payload(StatusEvent("KG response", "KG", progress=0.5, key="kg.model"),
                                            source=event)
            self._event_bus.publish(self._status_topic, statusEvent)

        elif response:
            logger.info("No graph_uri in response to %r in scenario %s", text, scenario_id)

            ### Update status field
            statusEvent = Event.for_payload(StatusEvent("LLM response", "LLM", progress=0.5, key="llm.model"),
                                            source=event)
            self._event_bus.publish(self._status_topic, statusEvent)

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
        self._recent_replies.pop(scenario_id, None)
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

    @staticmethod
    def _words(text: str):
        return re.findall(r"[a-z0-9']+", text.lower().replace("’", "'"))

    def _is_echo(self, scenario_id: str, text: str) -> bool:
        """Whether `text` mostly repeats one of the agent's last replies in this scenario: the
        microphone picking up the agent's own voice (a fragment of it, often). Short utterances
        ("No.", "Good") are never taken for an echo."""
        words = self._words(text)
        if len(words) < ECHO_MIN_WORDS:
            return False
        for reply in self._recent_replies.get(scenario_id, ()):
            reply_words = set(self._words(reply))
            if sum(word in reply_words for word in words) / len(words) >= ECHO_OVERLAP:
                return True
        return False

    @staticmethod
    def _is_timeline_request(text: str) -> bool:
        return bool(TIMELINE_REQUEST.search(text))

    def _show_timeline(self, scenario_id: str) -> str:
        """Plot the timeline of the human of this scenario from the knowledge graph, write it to
        an HTML page and open that in a new browser tab. Returns the reply to say."""
        conversation = self._conversations[scenario_id]
        # An AskNameConversation only knows its human once the name is given.
        human = getattr(conversation, "human", None) or self._speakers.get(scenario_id, self._human)
        try:
            path = self._write_timeline_page(conversation, human, scenario_id)
            webbrowser.open_new_tab(path.as_uri())
            logger.info("Opened the timeline of %s in scenario %s: %s", human, scenario_id, path)
        except Exception:
            logger.exception("Failed to show the timeline of %s in scenario %s", human, scenario_id)
            return TIMELINE_ERROR_REPLY

        return TIMELINE_REPLY.format(human=human)

    def _write_timeline_page(self, conversation, human: str, scenario_id: str) -> Path:
        # Imported here: matplotlib is only needed when a timeline is asked for, and the
        # service's worker thread must not use an interactive (GUI) backend.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from cltl.plots.patient_timeline import (describe_entry, fetch_timeline_entries, mark_regions,
                                                 plot_patient_timeline)

        entries = fetch_timeline_entries(self._ltm, human)
        fig = plot_patient_timeline(entries, human, current_date=datetime.now(), days=TIMELINE_DAYS,
                                    last_conversation_date=getattr(conversation, "last_conversation_date",
                                                                   None))
        try:
            # Drawn on an Agg canvas of its own, so the boxes measured after drawing
            # (mark_regions()) match the pixels of the PNG exactly.
            fig.set_dpi(TIMELINE_DPI)
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            regions = mark_regions(fig, canvas.get_renderer())
            image = io.BytesIO()
            canvas.print_png(image)
            width, height = canvas.get_width_height()
        finally:
            plt.close(fig)

        kg_address = getattr(conversation, "_kg_address", None)
        marks = [{"box": [round(100 * x0 / width, 3), round(100 * y0 / height, 3),
                          round(100 * (x1 - x0) / width, 3), round(100 * (y1 - y0) / height, 3)],
                  "title": f"{entry['type'].replace('_', ' ')}: {entry['label'] or entry['id'].rsplit('/', 1)[-1]}",
                  "summary": describe_entry(entry),
                  "times": [f"{role}: {', '.join(d.strftime('%Y-%m-%d') for d in dates)}"
                            for role, dates in entry.get("times", {}).items()],
                  "id": entry["id"],
                  "graphdb": _graphdb_url(kg_address, entry["id"])}
                 for entry, (x0, y0, x1, y1) in regions]

        page = (TIMELINE_PAGE
                .replace("{title}", html.escape(f"Timeline of {human}"))
                .replace("{subtitle}", html.escape(f"{datetime.now().strftime('%a %d %b %Y %H:%M')}, "
                                                   f"{len(entries)} entries in the knowledge graph. "
                                                   f"Point at an event for its details, click to keep them."))
                .replace("{png}", base64.b64encode(image.getvalue()).decode("ascii"))
                # </ cannot end the script element early: labels come from what the patient said.
                .replace("{marks}", json.dumps(marks).replace("</", "<\\/")))

        name = re.sub(r"[^A-Za-z0-9_.-]", "_", f"timeline_{human}_{scenario_id}")
        path = Path(tempfile.gettempdir()) / f"{name}.html"
        path.write_text(page, encoding="utf-8")
        return path

    def _publish(self, response: Optional[str], scenario_id: str, source: Event,
                 topic: Optional[str] = None):
        if not response:
            return
        if not topic or topic == self._output_topic:
            self._recent_replies.setdefault(scenario_id, deque(maxlen=ECHO_REPLIES)).append(response)

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


def _graphdb_url(kg_address: Optional[str], uri: str) -> Optional[str]:
    """GraphDB's own view of `uri`, or None without a knowledge graph address."""
    if not kg_address:
        return None
    try:
        from cltl.kg_chat.graphdb_links import graphdb_visualization_url
        return graphdb_visualization_url(kg_address, uri)
    except Exception:
        logger.exception("Failed to make the GraphDB link for %s", uri)
        return None


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
