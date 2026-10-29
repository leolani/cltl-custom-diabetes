import abc
from typing import NamedTuple, Optional


class Reply(NamedTuple):
    """The agent's reply to one utterance: its text (None to say nothing) and,
    when the utterance added triples to the knowledge graph, the GraphDB
    Workbench URL that visualises them. Unpacks as (text, graph_uri)."""
    text: Optional[str]
    graph_uri: Optional[str] = None


class Conversation(abc.ABC):
    """One knowledge-graph driven conversation with one human, i.e. one scenario.

    No EventBus and no configuration here: this is the boundary between the
    platform (service.py) and the conversation logic (catchup.py), so the
    service can be tested without a knowledge graph or an LLM behind it.
    """

    @abc.abstractmethod
    def open(self) -> Optional[str]:
        """The agent's opening turn: how long it has been since the last
        conversation, what was talked about back then, and a first question."""
        raise NotImplementedError()

    @abc.abstractmethod
    def respond(self, utterance: str) -> Reply:
        """The agent's reply to one human utterance, with the URL visualising
        the triples extracted from it, if any."""
        raise NotImplementedError()

    @abc.abstractmethod
    def close(self) -> None:
        """Called once when the scenario ends: persist whatever logs there are."""
        raise NotImplementedError()


class ConversationFactory(abc.ABC):
    @abc.abstractmethod
    def create(self, scenario_id: str, human: str, ltm) -> Conversation:
        """Start a new conversation with `human` for the scenario `scenario_id`,
        reading from and writing to the knowledge graph through `ltm`, the
        service's one LongTermMemory."""
        raise NotImplementedError()
