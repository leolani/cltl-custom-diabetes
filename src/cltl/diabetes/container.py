import logging
from typing import Optional

from cltl.combot.infra.container import InfraContainer
from cltl.combot.infra.di_container import singleton

from cltl.diabetes.api import ConversationFactory
from cltl.diabetes.catchup import CatchUpConversationFactory
from cltl.diabetes.service import DiabetesService

logger = logging.getLogger(__name__)

DEFAULT_MAX_ANSWER_WORDS = 80


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

        return CatchUpConversationFactory(
            kg_address=kg_address,
            intents_dir=config.get("intents_dir"),
            log_dir=config.get("log_dir"),
            fallback_gap_days=config.get_int("fallback_gap_days"),
            model=config.get("model") if "model" in config and config.get("model") else None,
            max_answer_words=_max_answer_words(config),
        )

    @property
    @singleton
    def diabetes_service(self) -> DiabetesService:
        return DiabetesService.from_config(self.diabetes_conversation_factory, self.event_bus,
                                           self.resource_manager, self.config_manager)

    def start(self):
        logger.info("Start Diabetes coach")
        super().start()
        self.diabetes_service.start()

    def stop(self):
        logger.info("Stop Diabetes coach")
        self.diabetes_service.stop()
        super().stop()
