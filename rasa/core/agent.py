from asyncio import CancelledError
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Text,
    Tuple,
    Union,
)
import uuid

import aiohttp
from aiohttp import ClientError

import rasa
from rasa.engine.runner.interface import GraphRunner
from rasa.engine.storage.storage import ModelMetadata
import rasa.utils
from rasa.core import jobs, training
from rasa.core.channels.channel import OutputChannel, UserMessage
from rasa.core.constants import DEFAULT_REQUEST_TIMEOUT
from rasa.engine import loader
from rasa.engine.runner.dask import DaskGraphRunner
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.shared.core.domain import Domain
from rasa.core.exceptions import AgentNotReady
from rasa.shared.constants import (
    DEFAULT_SENDER_ID,
    DEFAULT_DOMAIN_PATH,
    DEFAULT_CORE_SUBDIRECTORY_NAME,
)
from rasa.shared.exceptions import InvalidParameterException
from rasa.core.lock_store import InMemoryLockStore, LockStore
from rasa.core.nlg import NaturalLanguageGenerator
from rasa.core.policies.policy import Policy, PolicyPrediction
from rasa.core.processor import MessageProcessor
from rasa.core.tracker_store import (
    FailSafeTrackerStore,
    InMemoryTrackerStore,
    TrackerStore,
)
from rasa.shared.core.trackers import DialogueStateTracker
import rasa.core.utils
from rasa.exceptions import ModelNotFound
from rasa.shared.importers.importer import TrainingDataImporter
from rasa.model import get_latest_model
from rasa.nlu.utils import is_url
import rasa.shared.utils.io
from rasa.shared.nlu.training_data.training_data import TrainingData
from rasa.utils.endpoints import EndpointConfig
import rasa.utils.io

from rasa.shared.core.generator import TrackerWithCachedStates
from rasa.core.tracker_store import TrackerStore
from rasa.core.utils import AvailableEndpoints
from rasa.core.brokers.broker import EventBroker
import rasa.utils.common

logger = logging.getLogger(__name__)


async def load_from_server(agent: "Agent", model_server: EndpointConfig) -> "Agent":
    """Load a persisted model from a server."""

    # We are going to pull the model once first, and then schedule a recurring
    # job. the benefit of this approach is that we can be sure that there
    # is a model after this function completes -> allows to do proper
    # "is alive" check on a startup server's `/status` endpoint. If the server
    # is started, we can be sure that it also already loaded (or tried to)
    # a model.
    await _update_model_from_server(model_server, agent)

    wait_time_between_pulls = model_server.kwargs.get("wait_time_between_pulls", 100)

    if wait_time_between_pulls:
        # continuously pull the model every `wait_time_between_pulls` seconds
        await schedule_model_pulling(model_server, int(wait_time_between_pulls), agent)

    return agent


def _load_and_set_updated_model(
    agent: "Agent", model_directory: Text, fingerprint: Text
) -> None:
    """Load the persisted model into memory and set the model on the agent.

    Args:
        agent: Instance of `Agent` to update with the new model.
        model_directory: Rasa model directory.
        fingerprint: Fingerprint of the supplied model at `model_directory`.
    """
    logger.debug(f"Found new model with fingerprint {fingerprint}. Loading...")
    agent.update_model(model_directory, fingerprint)

    logger.debug("Finished updating agent to new model.")


async def _update_model_from_server(
    model_server: EndpointConfig, agent: "Agent"
) -> None:
    """Load a zipped Rasa Core model from a URL and update the passed agent."""

    if not is_url(model_server.url):
        raise aiohttp.InvalidURL(model_server.url)

    model_directory = tempfile.mkdtemp()
    remove_dir = True

    try:
        new_fingerprint = await _pull_model_and_fingerprint(
            model_server, agent.fingerprint, model_directory
        )

        if new_fingerprint:
            _load_and_set_updated_model(agent, model_directory, new_fingerprint)
            remove_dir = False
        else:
            logger.debug(f"No new model found at URL {model_server.url}")
    except Exception:  # skipcq: PYL-W0703
        # TODO: Make this exception more specific, possibly print different log
        # for each one.
        logger.exception(
            "Failed to update model. The previous model will stay loaded instead."
        )
    finally:
        if remove_dir:
            shutil.rmtree(model_directory)


async def _pull_model_and_fingerprint(
    model_server: EndpointConfig, fingerprint: Optional[Text], model_directory: Text
) -> Optional[Text]:
    """Queries the model server.

    Args:
        model_server: Model server endpoint information.
        fingerprint: Current model fingerprint.
        model_directory: Directory where to download model to.

    Returns:
        Value of the response's <ETag> header which contains the model
        hash. Returns `None` if no new model is found.
    """
    headers = {"If-None-Match": fingerprint}

    logger.debug(f"Requesting model from server {model_server.url}...")

    async with model_server.session() as session:
        try:
            params = model_server.combine_parameters()
            async with session.request(
                "GET",
                model_server.url,
                timeout=DEFAULT_REQUEST_TIMEOUT,
                headers=headers,
                params=params,
            ) as resp:

                if resp.status in [204, 304]:
                    logger.debug(
                        "Model server returned {} status code, "
                        "indicating that no new model is available. "
                        "Current fingerprint: {}"
                        "".format(resp.status, fingerprint)
                    )
                    return None
                elif resp.status == 404:
                    logger.debug(
                        "Model server could not find a model at the requested "
                        "endpoint '{}'. It's possible that no model has been "
                        "trained, or that the requested tag hasn't been "
                        "assigned.".format(model_server.url)
                    )
                    return None
                elif resp.status != 200:
                    logger.debug(
                        "Tried to fetch model from server, but server response "
                        "status code is {}. We'll retry later..."
                        "".format(resp.status)
                    )
                    return None

                model_path = Path(model_directory) / resp.headers.get("filename")
                with open(model_path, "wb") as file:
                    file.write(await resp.read())

                logger.debug(
                    "Saved model to '{}'".format(os.path.abspath(model_path))
                )

                # return the new fingerprint
                return resp.headers.get("ETag")

        except aiohttp.ClientError as e:
            logger.debug(
                "Tried to fetch model from server, but "
                "couldn't reach server. We'll retry later... "
                "Error: {}.".format(e)
            )
            return None


async def _run_model_pulling_worker(
    model_server: EndpointConfig, agent: "Agent"
) -> None:
    # noinspection PyBroadException
    try:
        await _update_model_from_server(model_server, agent)
    except CancelledError:
        logger.warning("Stopping model pulling (cancelled).")
    except ClientError:
        logger.exception(
            "An exception was raised while fetching a model. Continuing anyways..."
        )


async def schedule_model_pulling(
    model_server: EndpointConfig, wait_time_between_pulls: int, agent: "Agent"
) -> None:
    (await jobs.scheduler()).add_job(
        _run_model_pulling_worker,
        "interval",
        seconds=wait_time_between_pulls,
        args=[model_server, agent],
        id="pull-model-from-server",
        replace_existing=True,
    )


def create_agent(model: Text, endpoints: Text = None) -> "Agent":
    """Create an agent instance based on a stored model.

    Args:
        model: file path to the stored model
        endpoints: file path to the used endpoint configuration
    """
    from rasa.core.tracker_store import TrackerStore
    from rasa.core.utils import AvailableEndpoints
    from rasa.core.brokers.broker import EventBroker
    import rasa.utils.common

    # TODO: JUZL: Can we move all this into Agent as we do it in multiple places?
    _endpoints = AvailableEndpoints.read_endpoints(endpoints)

    _broker = rasa.utils.common.run_in_loop(EventBroker.create(_endpoints.event_broker))
    _tracker_store = TrackerStore.create(_endpoints.tracker_store, event_broker=_broker)
    _lock_store = LockStore.create(_endpoints.lock_store)

    return Agent.load(
        model,
        generator=_endpoints.nlg,
        tracker_store=_tracker_store,
        lock_store=_lock_store,
        action_endpoint=_endpoints.action,
    )


async def load_agent(
    model_path: Optional[Text] = None,
    model_server: Optional[EndpointConfig] = None,
    remote_storage: Optional[Text] = None,
    endpoints: Optional[AvailableEndpoints] = None,
) -> Optional["Agent"]:
    """Loads agent from server, remote storage or disk.

    Args:
        model_path: Path to the model if it's on disk.
        model_server: Configuration for a potential server which serves the model.
        remote_storage: URL of remote storage for model.
        endpoints: Endpoint configuration.
    Returns:
        The instantiated `Agent` or `None`.
    """

    from rasa.core.tracker_store import TrackerStore
    from rasa.core.brokers.broker import EventBroker
    import rasa.utils.common

    tracker_store = None
    lock_store = None
    generator = None
    action_endpoint = None

    # TODO: JUZL: get model_server from endpoints.

    if endpoints:
        broker = rasa.utils.common.run_in_loop(
            EventBroker.create(endpoints.event_broker)
        )
        tracker_store = TrackerStore.create(
            endpoints.tracker_store, event_broker=broker
        )
        lock_store = LockStore.create(endpoints.lock_store)
        generator = endpoints.nlg
        action_endpoint = endpoints.action

    try:
        if model_server is not None:
            return await load_from_server(
                Agent(
                    generator=generator,
                    tracker_store=tracker_store,
                    lock_store=lock_store,
                    action_endpoint=action_endpoint,
                    model_server=model_server,
                    remote_storage=remote_storage,
                ),
                model_server,
            )

        elif remote_storage is not None:
            return Agent.load_from_remote_storage(
                remote_storage,
                model_path,
                generator=generator,
                tracker_store=tracker_store,
                lock_store=lock_store,
                action_endpoint=action_endpoint,
                model_server=model_server,
            )

        elif model_path is not None and os.path.exists(model_path):
            return Agent.load(
                model_path,
                generator=generator,
                tracker_store=tracker_store,
                lock_store=lock_store,
                action_endpoint=action_endpoint,
                model_server=model_server,
                remote_storage=remote_storage,
            )

        else:
            rasa.shared.utils.io.raise_warning(
                "No valid configuration given to load agent."
            )
            return None

    except Exception as e:
        logger.error(f"Could not load model due to {e}.")
        raise


class Agent:
    """The Agent class provides a convenient interface for the most important
    Rasa functionality.

    This includes training, handling messages, loading a dialogue model,
    getting the next action, and handling a channel."""

    def __init__(
        self,
        domain: Union[Text, Domain, None] = None,
        generator: Union[EndpointConfig, NaturalLanguageGenerator, None] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        fingerprint: Optional[Text] = None,
        model_directory: Optional[Text] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        path_to_model_archive: Optional[Text] = None,
        graph_runner: Optional[GraphRunner] = None,
    ):
        self.domain = domain
        self.nlg = NaturalLanguageGenerator.create(generator, self.domain)
        self.tracker_store = self._create_tracker_store(tracker_store, self.domain)
        self.lock_store = self._create_lock_store(lock_store)
        self.action_endpoint = action_endpoint
        self.graph_runner = graph_runner

        self._set_fingerprint(fingerprint)
        self.model_server = model_server
        self.remote_storage = remote_storage
        self.path_to_model_archive = path_to_model_archive

    def update_model(
        self,
        model_path: Union[Text, Path],
        fingerprint: Optional[Text],
    ) -> None:
        domain, graph_runner = self.load_graph_runner(model_path)
        self.domain = domain
        self.graph_runner = graph_runner

        self._set_fingerprint(fingerprint)

        # update domain on all instances
        self.tracker_store.domain = domain
        if hasattr(self.nlg, "responses"):
            self.nlg.responses = domain.responses if domain else {}

    @classmethod
    def load(
        cls,
        model_path: Union[Text, Path],
        generator: Union[EndpointConfig, NaturalLanguageGenerator] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        model_server: Optional[EndpointConfig] = None,
        remote_storage: Optional[Text] = None,
        path_to_model_archive: Optional[Text] = None,
        new_config: Optional[Dict] = None,
        finetuning_epoch_fraction: float = 1.0,
    ) -> "Agent":
        """Load a persisted model from the passed path."""

        # TODO: JUZL:
        # new_config=new_config,
        # finetuning_epoch_fraction=finetuning_epoch_fraction,
        # # ensures the domain hasn't changed between test and train
        # domain.compare_with_specification(core_model)

        domain, graph_runner = cls.load_graph_runner(model_path)

        agent = cls(
            domain=domain,
            generator=generator,
            tracker_store=tracker_store,
            lock_store=lock_store,
            action_endpoint=action_endpoint,
            model_directory=model_path,
            model_server=model_server,
            remote_storage=remote_storage,
            path_to_model_archive=path_to_model_archive,
            graph_runner=graph_runner,
        )

        agent.initialize_processor()
        return agent

    # TODO: move to processor?
    @classmethod
    def load_graph_runner(cls, model_path: Union[Text, Path]) -> Tuple[Domain, GraphRunner]:
        model_tar = rasa.model.get_latest_model(model_path)
        if not model_tar:
            raise ModelNotFound(f"No model found at path {model_path}.")
        tmp_model_path = tempfile.mkdtemp()
        metadata, graph_runner = loader.load_predict_graph_runner(
            Path(tmp_model_path), Path(model_tar), LocalModelStorage, DaskGraphRunner,
        )
        return metadata.domain, graph_runner

    def is_ready(self) -> bool:
        """Check if all necessary components are instantiated to use agent."""
        # TODO: JUZL: check more?
        return self.tracker_store is not None and self.processor is not None

    async def parse_message(
        self, message_data: Text, tracker: DialogueStateTracker = None
    ) -> Dict[Text, Any]:
        """Handles message text and intent payload input messages.

        The return value of this function is parsed_data.

        Args:
            message_data (Text): Contain the received message in text or\
            intent payload format.
            tracker (DialogueStateTracker): Contains the tracker to be\
            used by the interpreter.

        Returns:
            The parsed message.

            Example:

                {\
                    "text": '/greet{"name":"Rasa"}',\
                    "intent": {"name": "greet", "confidence": 1.0},\
                    "intent_ranking": [{"name": "greet", "confidence": 1.0}],\
                    "entities": [{"entity": "name", "start": 6,\
                                  "end": 21, "value": "Rasa"}],\
                }

        """
        if not self.is_ready():
            raise AgentNotReady(
                "Agent needs to be prepared before usage. You need to set an "
                "interpreter and a tracker store."
            )
        message = UserMessage(message_data)
        return await self.processor.parse_message(message, tracker)

    async def handle_message(
        self,
        message: UserMessage,
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message."""
        if not self.is_ready():
            logger.info("Ignoring message as there is no agent to handle it.")
            return None

        if message_preprocessor:
            self.processor.message_preprocessor = message_preprocessor

        async with self.lock_store.lock(message.sender_id):
            return await self.processor.handle_message(message)

    async def predict_next(
        self, sender_id: Text
    ) -> Optional[Dict[Text, Any]]:
        """Predict the next action."""
        return await self.processor.predict_next(sender_id)

    async def log_message(
        self,
        message: UserMessage,
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        **kwargs: Any,
    ) -> DialogueStateTracker:
        """Append a message to a dialogue - does not predict actions."""
        # TODO: JUZL: Should this just be for this message?
        if message_preprocessor:
            self.processor.message_preprocessor = message_preprocessor

        return await self.processor.log_message(message)

    async def execute_action(
        self,
        sender_id: Text,
        action: Text,
        output_channel: OutputChannel,
        policy: Optional[Text],
        confidence: Optional[float],
    ) -> Optional[DialogueStateTracker]:
        """Execute an action."""
        prediction = PolicyPrediction.for_action_name(
            self.domain, action, policy, confidence or 0.0
        )
        return await self.processor.execute_action(
            sender_id, action, output_channel, self.nlg, prediction
        )

    async def trigger_intent(
        self,
        intent_name: Text,
        entities: List[Dict[Text, Any]],
        output_channel: OutputChannel,
        tracker: DialogueStateTracker,
    ) -> None:
        """Trigger a user intent, e.g. triggered by an external event."""

        await self.processor.trigger_external_user_uttered(
            intent_name, entities, tracker, output_channel
        )

    async def handle_text(
        self,
        text_message: Union[Text, Dict[Text, Any]],
        message_preprocessor: Optional[Callable[[Text], Text]] = None,
        output_channel: Optional[OutputChannel] = None,
        sender_id: Optional[Text] = DEFAULT_SENDER_ID,
    ) -> Optional[List[Dict[Text, Any]]]:
        """Handle a single message.

        If a message preprocessor is passed, the message will be passed to that
        function first and the return value is then used as the
        input for the dialogue engine.

        The return value of this function depends on the ``output_channel``. If
        the output channel is not set, set to ``None``, or set
        to ``CollectingOutputChannel`` this function will return the messages
        the bot wants to respond.

        :Example:

            >>> from rasa.core.agent import Agent
            >>> from rasa.core.interpreter import RasaNLUInterpreter
            >>> agent = Agent.load("examples/moodbot/models")
            >>> await agent.handle_text("hello")
            [u'how can I help you?']

        """

        if isinstance(text_message, str):
            text_message = {"text": text_message}

        msg = UserMessage(text_message.get("text"), output_channel, sender_id)

        return await self.handle_message(msg, message_preprocessor)

    def _set_fingerprint(self, fingerprint: Optional[Text] = None) -> None:

        if fingerprint:
            self.fingerprint = fingerprint
        else:
            self.fingerprint = uuid.uuid4().hex

    async def visualize(
        self,
        resource_name: Text,
        output_file: Text,
        max_history: Optional[int] = None,
        nlu_training_data: Optional[TrainingData] = None,
        should_merge_nodes: bool = True,
        fontsize: int = 12,
    ) -> None:
        """Visualize the loaded training data from the resource."""

        # TODO: JUZL:
        from rasa.shared.core.training_data.visualization import visualize_stories
        from rasa.shared.core.training_data import loading

        # if the user doesn't provide a max history, we will use the
        # largest value from any policy
        max_history = max_history or self._max_history()

        story_steps = loading.load_data_from_resource(resource_name, self.domain)
        await visualize_stories(
            story_steps,
            self.domain,
            output_file,
            max_history,
            self.interpreter,
            nlu_training_data,
            should_merge_nodes,
            fontsize,
        )


    @staticmethod
    def _create_tracker_store(
        store: Optional[TrackerStore], domain: Domain
    ) -> TrackerStore:
        if store is not None:
            store.domain = domain
            tracker_store = store
        else:
            tracker_store = InMemoryTrackerStore(domain)

        return FailSafeTrackerStore(tracker_store)

    @staticmethod
    def _create_lock_store(store: Optional[LockStore]) -> LockStore:
        if store is not None:
            return store

        return InMemoryLockStore()

    @staticmethod
    def load_from_remote_storage(
        remote_storage: Text,
        model_name: Text,
        generator: Union[EndpointConfig, NaturalLanguageGenerator] = None,
        tracker_store: Optional[TrackerStore] = None,
        lock_store: Optional[LockStore] = None,
        action_endpoint: Optional[EndpointConfig] = None,
        model_server: Optional[EndpointConfig] = None,
    ) -> Optional["Agent"]:
        from rasa.nlu.persistor import get_persistor

        persistor = get_persistor(remote_storage)

        if persistor is not None:
            target_path = tempfile.mkdtemp()
            persistor.retrieve(model_name, target_path)

            return Agent.load(
                target_path,
                generator=generator,
                tracker_store=tracker_store,
                lock_store=lock_store,
                action_endpoint=action_endpoint,
                model_server=model_server,
                remote_storage=remote_storage,
            )

        return None

    # TODO: JUZL:
    def initialize_processor(self) -> None:
        processor = MessageProcessor(
            graph_runner=self.graph_runner,
            domain=self.domain,
            tracker_store=self.tracker_store,
            lock_store=self.lock_store,
            action_endpoint=self.action_endpoint,
            generator=self.nlg,
        )
        self.processor = processor
