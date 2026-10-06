import logging.config
import os
import signal

from cltl.combot.event.emissor import SIG, MEN
from cltl.combot.infra.config.k8config import K8LocalConfigurationContainer
from cltl.combot.infra.di_container import singleton
from cltl.combot.infra.event.api import Event, PAYLOAD
from cltl.combot.infra.event.memory import SynchronousEventBus
from cltl.diabetes.container import DiabetesContainer
from emissor.representation.util import marshal, unmarshal, register_type_var
from flask import Flask
from werkzeug.middleware.dispatcher import DispatcherMiddleware
from werkzeug.serving import run_simple

# Top-level module, deliberately: find_namespace_packages(include=['cltl.diabetes', ...])
# in setup.py does not match a bare `main`, so this file is never part of the
# installed distribution. Only the container image (CMD ["python", "src/main.py"])
# and a developer running it directly ever execute it — see docs/component.md.

logging.config.fileConfig(os.environ.get('CLTL_LOGGING_CONFIG', 'config/logging.config'),
                          disable_existing_loggers=False)
logger = logging.getLogger(__name__)

# Must happen before anything marshals an Event, or emissor cannot resolve the
# generic type variables and raises `TypeError: PAYLOAD is not a dataclass`.
register_type_var(PAYLOAD)
register_type_var(SIG)
register_type_var(MEN)


def serializer(obj):
    return marshal(obj, cls=Event)


def deserializer(obj):
    return unmarshal(obj, cls=Event)


class ApplicationContainer(DiabetesContainer):
    """This deployment: just the component. Scenario creation is not this
    module's job — it is the platform's own cltl-context, one per tenant,
    running in the deployment this module attaches to (see
    clients/context in cltl-apps-restructure_monitoring), so there is nothing
    to compose here beyond `DiabetesContainer`.
    """

    @property
    @singleton
    def event_bus_serializer(self):
        return serializer, deserializer

    @property
    @singleton
    def event_bus(self):
        config = self.config_manager.get_config("cltl.event")
        if config.get("implementation") == "internal":
            return SynchronousEventBus()
        return super().event_bus


def main():
    K8LocalConfigurationContainer.load_configuration()
    application = ApplicationContainer()

    # `docker compose down` and `docker stop` send SIGTERM, whose DEFAULT
    # disposition terminates the process outright — so `with application:`
    # would never reach its __exit__, TenantContainer.stop would never run, and
    # ScenarioStopped would never be published. The tenant's chat UI would be
    # left holding a scenario whose owner is gone. (cltl-context/src/main.py has
    # the identical latent bug.) Turning the signal into KeyboardInterrupt makes
    # it unwind through the `with` like a Ctrl-C does.
    def _interrupt(signum, frame):
        logger.info("Received signal %s; shutting down", signum)
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, _interrupt)

    with application:
        flask_app = Flask(__name__)

        @flask_app.route('/health')
        def health():
            return 'OK', 200

        try:
            run_simple('0.0.0.0', 8008, DispatcherMiddleware(flask_app),
                       threaded=True, use_reloader=False, use_debugger=False)
        except KeyboardInterrupt:
            # Caught here rather than relying on werkzeug: its
            # BaseWSGIServer.serve_forever does swallow KeyboardInterrupt, but
            # that is an implementation detail of a pinned version, and the cost
            # of not depending on it is one except clause.
            pass


if __name__ == '__main__':
    main()
