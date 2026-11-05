import logging
from typing import Optional

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton

from cltl.diabetes import place
from cltl.diabetes.api import ConversationFactory
from cltl.diabetes.catchup import CatchUpConversationFactory, connect_ltm
from cltl.diabetes.service import DiabetesService
from cltl.kg_chat import llm_backend

logger = logging.getLogger(__name__)

DEFAULT_MAX_ANSWER_WORDS = 35


def _kg_address(config) -> Optional[str]:
    """`kg_address` when it is set, otherwise <kg_server>/repositories/<kg_repository>.

    An unset environment variable reaches us as the literal "$VAR" (see
    config/custom.config.example), which counts as not set.
    """
    address = config.get("kg_address") if "kg_address" in config else None
    if address and not address.startswith("$"):
        return address

    server = config.get("kg_server") if "kg_server" in config else None
    repository = config.get("kg_repository") if "kg_repository" in config else None
    if not server or not repository or server.startswith("$") or repository.startswith("$"):
        return address

    return f"{server.rstrip('/')}/repositories/{repository.strip('/')}"


def _max_answer_words(config) -> Optional[int]:
    """None when answering questions from the knowledge graph is switched off."""
    if "answer_questions" in config and not config.get_boolean("answer_questions"):
        return None
    if "max_answer_words" in config and config.get("max_answer_words"):
        return config.get_int("max_answer_words")
    return DEFAULT_MAX_ANSWER_WORDS


def _float(config, key: str, default: float) -> float:
    return float(config.get(key)) if key in config and config.get(key) else default


def _str(config, key: str) -> Optional[str]:
    """`key`'s value, None when unset, empty or an unset "$VAR"."""
    value = config.get(key) if key in config else None
    return value.strip() if value and value.strip() and not value.startswith("$") else None


def _configure_llm(config):
    """OpenAI or a local OpenAI-compatible server (llama-server), see llm_backend."""
    llm_backend.configure(
        backend=_str(config, "llm_backend") or llm_backend.OPENAI,
        base_url=_str(config, "llm_base_url") if _str(config, "llm_backend") == llm_backend.LOCAL else None,
        model=_str(config, "model"),
        timeout=_float(config, "llm_timeout", 0) or None,
    )
    description = llm_backend.describe()
    if "NOT REACHABLE" in description:
        logger.warning("LLM: %s -- start llama-server first, or set llm_backend: openai", description)
    else:
        logger.info("LLM: %s", description)


class DiabetesContainer(InfraContainer):
    """Wires the knowledge graph driven conversation into a deployment.

    Accessors are prefixed with `diabetes_`: @singleton caches by the bare
    method name across the whole process, so `service` would collide with any
    other component mixed into the same application.
    """

    @property
    @singleton
    def diabetes_conversation_factory(self) -> ConversationFactory:
        config = self.config_manager.get_config("cltl.diabetes")
        kg_address = _kg_address(config)
        logger.info("Knowledge graph: %s", kg_address)
        _configure_llm(config)

        return CatchUpConversationFactory(
            kg_address=kg_address,
            intents_dir=config.get("intents_dir"),
            log_dir=config.get("log_dir"),
            fallback_gap_days=config.get_int("fallback_gap_days"),
            model=config.get("model") if "model" in config and config.get("model") else None,
            max_answer_words=_max_answer_words(config),
            place_same_threshold=_float(config, "place_same_threshold", place.DEFAULT_SAME_THRESHOLD),
            place_ask_threshold=_float(config, "place_ask_threshold", place.DEFAULT_ASK_THRESHOLD),
            ask_name=config.get_boolean("ask_name") if "ask_name" in config else False,
        )

    @property
    @singleton
    def diabetes_ltm(self):
        """The one LongTermMemory for the knowledge graph, connected when the
        container starts. `clear_brain` empties the repository first."""
        config = self.config_manager.get_config("cltl.diabetes")

        return connect_ltm(_kg_address(config), config.get("log_dir"),
                           clear_brain="clear_brain" in config and config.get_boolean("clear_brain"))

    @property
    @singleton
    def diabetes_service(self) -> DiabetesService:
        return DiabetesService.from_config(self.diabetes_conversation_factory, self.diabetes_ltm,
                                           self.event_bus, self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Diabetes coach")
        super().start()
        self.diabetes_service.start()

    def stop(self):
        logger.info("Stop Diabetes coach")
        self.diabetes_service.stop()
        super().stop()
