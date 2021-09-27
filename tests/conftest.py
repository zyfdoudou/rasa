import asyncio
import copy
import os
import random
import textwrap

import pytest
import sys
import uuid

from _pytest.monkeypatch import MonkeyPatch
from _pytest.python import Function
from spacy import Language

from rasa.engine.caching import LocalTrainingCache
from rasa.engine.graph import ExecutionContext, GraphSchema
from rasa.engine.storage.local_model_storage import LocalModelStorage
from rasa.engine.storage.storage import ModelStorage
from sanic.request import Request

from typing import Iterator, Callable, Generator

from _pytest.tmpdir import TempPathFactory, TempdirFactory
from pathlib import Path
from sanic import Sanic
from typing import Text, List, Optional, Dict, Any
from unittest.mock import Mock

import rasa.shared.utils.io
from rasa.nlu.components import ComponentBuilder
from rasa.nlu.config import RasaNLUModelConfig
from rasa import server
from rasa.core.agent import Agent, load_agent
from rasa.core.brokers.broker import EventBroker
from rasa.core.channels import channel, RestInput

from rasa.nlu.model import Interpreter
from rasa.nlu.utils.spacy_utils import SpacyNLPGraphComponent
from rasa.shared.constants import LATEST_TRAINING_DATA_FORMAT_VERSION
from rasa.shared.core.domain import SessionConfig, Domain
from rasa.shared.core.events import UserUttered
from rasa.core.exporter import Exporter

import rasa.core.run
from rasa.core.tracker_store import InMemoryTrackerStore, TrackerStore
from rasa.model_training import train, train_nlu_async
from rasa.utils.common import TempDirectoryPath
from rasa.shared.exceptions import RasaException

# we reuse a bit of pytest's own testing machinery, this should eventually come
# from a separatedly installable pytest-cli plugin.
pytest_plugins = ["pytester"]


# these tests are run separately
collect_ignore_glob = ["docs/*.py"]

# Defines how tests are parallelized in the CI
PATH_PYTEST_MARKER_MAPPINGS = {
    "category_cli": [Path("tests", "cli").absolute()],
    "category_core_featurizers": [Path("tests", "core", "featurizers").absolute()],
    "category_policies": [
        Path("tests", "core", "test_policies.py").absolute(),
        Path("tests", "core", "policies").absolute(),
    ],
    "category_nlu_featurizers": [
        Path("tests", "nlu", "featurizers").absolute(),
        Path("tests", "nlu", "utils").absolute(),
    ],
    "category_nlu_predictors": [
        Path("tests", "nlu", "classifiers").absolute(),
        Path("tests", "nlu", "extractors").absolute(),
        Path("tests", "nlu", "selectors").absolute(),
    ],
    "category_full_model_training": [
        Path("tests", "test_model_training.py").absolute(),
        Path("tests", "nlu", "test_train.py").absolute(),
        Path("tests", "core", "test_training.py").absolute(),
        Path("tests", "core", "test_examples.py").absolute(),
    ],
    "category_performance": [Path("tests", "test_memory_leak.py").absolute()],
}


@pytest.fixture(scope="session")
def nlu_as_json_path() -> Text:
    return "data/examples/rasa/demo-rasa.json"


@pytest.fixture(scope="session")
def nlu_data_path() -> Text:
    return "data/test_moodbot/data/nlu.yml"


@pytest.fixture(scope="session")
def config_path() -> Text:
    return "rasa/shared/importers/default_config.yml"


@pytest.fixture(scope="session")
def domain_with_categorical_slot_path() -> Text:
    return "data/test_domains/domain_with_categorical_slot.yml"


@pytest.fixture(scope="session")
def domain_with_mapping_path() -> Text:
    return "data/test_domains/default_with_mapping.yml"


@pytest.fixture(scope="session")
def stories_path() -> Text:
    return "data/test_yaml_stories/stories_defaultdomain.yml"


@pytest.fixture(scope="session")
def e2e_stories_path() -> Text:
    return "data/test_yaml_stories/stories_e2e.yml"


@pytest.fixture(scope="session")
def simple_stories_path() -> Text:
    return "data/test_yaml_stories/stories_simple.yml"


@pytest.fixture(scope="session")
def stack_config_path() -> Text:
    return "data/test_config/stack_config.yml"


@pytest.fixture(scope="session")
def incorrect_nlu_data_path() -> Text:
    return "data/test/incorrect_nlu_format.yml"


@pytest.fixture(scope="session")
def end_to_end_story_path() -> Text:
    return "data/test_evaluations/test_end_to_end_story.yml"


@pytest.fixture(scope="session")
def e2e_story_file_unknown_entity_path() -> Text:
    return "data/test_evaluations/test_story_unknown_entity.yml"


@pytest.fixture(scope="session")
def domain_path() -> Text:
    return "data/test_domains/default_with_slots.yml"


@pytest.fixture(scope="session")
def story_file_trips_circuit_breaker_path() -> Text:
    return "data/test_evaluations/test_stories_trip_circuit_breaker.yml"


@pytest.fixture(scope="session")
def e2e_story_file_trips_circuit_breaker_path() -> Text:
    return "data/test_evaluations/test_end_to_end_trips_circuit_breaker.yml"


@pytest.fixture(scope="session")
def endpoints_path() -> Text:
    return "data/test_endpoints/example_endpoints.yml"


# https://github.com/pytest-dev/pytest-asyncio/issues/68
# this event_loop is used by pytest-asyncio, and redefining it
# is currently the only way of changing the scope of this fixture
@pytest.fixture(scope="session")
def event_loop(request: Request) -> Iterator[asyncio.AbstractEventLoop]:
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
async def _trained_default_agent(
    tmp_path_factory: TempPathFactory,
    stories_path: Text,
    nlu_data_path: Text,
    trained_async: Callable,
) -> Agent:
    project_path = tmp_path_factory.mktemp("project")

    config = textwrap.dedent(
        f"""
    version: "{LATEST_TRAINING_DATA_FORMAT_VERSION}"
    pipeline:
      - name: WhitespaceTokenizer
      - name: CountVectorsFeaturizer
      - name: DIETClassifier
        epochs: 10
    policies:
    - name: AugmentedMemoizationPolicy
      max_history: 3
    - name: RulePolicy
    """
    )
    config_path = project_path / "config.yml"
    rasa.shared.utils.io.write_text_file(config, config_path)
    model_path = train(
        "data/test_domains/default_with_slots.yml",
        str(config_path),
        [stories_path, nlu_data_path],
    ).model

    return await load_agent(model_path=model_path)


@pytest.fixture()
def empty_agent() -> Agent:
    agent = Agent("data/test_domains/default_with_slots.yml",)
    return agent


def reset_conversation_state(agent: Agent) -> Agent:
    # Clean tracker store after each test so tests don't affect each other
    agent.tracker_store = InMemoryTrackerStore(agent.domain)
    agent.domain.session_config = SessionConfig.default()
    agent.initialize_processor()
    return agent


@pytest.fixture
def default_agent(_trained_default_agent: Agent) -> Agent:
    return reset_conversation_state(_trained_default_agent)


@pytest.fixture(scope="session")
async def trained_moodbot_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/config.yml",
        training_files="data/test_moodbot/data/",
    )


@pytest.fixture(scope="session")
async def trained_unexpected_intent_policy_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/unexpected_intent_policy_config.yml",
        training_files="data/test_moodbot/data/",
    )


@pytest.fixture(scope="session")
async def trained_nlu_moodbot_path(trained_nlu_async: Callable) -> Text:
    return await trained_nlu_async(
        domain="data/test_moodbot/domain.yml",
        config="data/test_moodbot/config.yml",
        nlu_data="data/test_moodbot/data/nlu.yml",
    )


# TODO: JUZL:
@pytest.fixture(scope="session")
def unpacked_trained_moodbot_path(trained_moodbot_path: Text,) -> TempDirectoryPath:
    # return get_model(trained_moodbot_path)
    pass


@pytest.fixture(scope="session")
async def trained_spacybot_path(trained_async: Callable) -> Text:
    return await trained_async(
        domain="data/test_spacybot/domain.yml",
        config="data/test_spacybot/config.yml",
        training_files="data/test_spacybot/data/",
    )


@pytest.fixture(scope="session")
async def stack_agent(trained_rasa_model: Text) -> Agent:
    return await load_agent(model_path=trained_rasa_model)


@pytest.fixture(scope="session")
async def core_agent(trained_core_model: Text) -> Agent:
    return await load_agent(model_path=trained_core_model)


@pytest.fixture(scope="session")
async def nlu_agent(trained_nlu_model: Text) -> Agent:
    return await load_agent(model_path=trained_nlu_model)


@pytest.fixture(scope="module")
async def unexpected_intent_policy_agent(
    trained_unexpected_intent_policy_path: Text,
) -> Agent:
    return await load_agent(model_path=trained_unexpected_intent_policy_path)


@pytest.fixture(scope="module")
async def mood_agent(trained_moodbot_path: Text) -> Agent:
    return await load_agent(model_path=trained_moodbot_path)


@pytest.fixture(scope="session")
def _domain(domain_path: Text) -> Domain:
    return Domain.load(domain_path)


@pytest.fixture()
def domain(_domain: Domain) -> Domain:
    return copy.deepcopy(_domain)


@pytest.fixture(scope="session")
def trained_async(tmp_path_factory: TempPathFactory) -> Callable:
    async def _train(
        *args: Any,
        output_path: Optional[Text] = None,
        cache_dir: Optional[Path] = None,
        **kwargs: Any,
    ) -> Optional[Text]:

        old_get_cache_location = None
        if cache_dir:
            old_get_cache_location = LocalTrainingCache._get_cache_location
            LocalTrainingCache._get_cache_location = Mock(return_value=cache_dir)

        if output_path is None:
            output_path = str(tmp_path_factory.mktemp("models"))

        result = train(*args, output=output_path, **kwargs)

        if cache_dir:
            LocalTrainingCache._get_cache_location = old_get_cache_location

        return result.model

    return _train


@pytest.fixture(scope="session")
def trained_nlu_async(tmp_path_factory: TempPathFactory) -> Callable:
    async def _train_nlu(
        *args: Any, output_path: Optional[Text] = None, **kwargs: Any
    ) -> Optional[Text]:
        if output_path is None:
            output_path = str(tmp_path_factory.mktemp("models"))

        return await train_nlu_async(*args, output=output_path, **kwargs)

    return _train_nlu


@pytest.fixture(scope="session")
def trained_rasa_model(
    trained_async: Callable,
    domain_path: Text,
    nlu_data_path: Text,
    stories_path: Text,
    stack_config_path: Text,
) -> Text:
    trained_stack_model_path = rasa.api.train(
        domain=domain_path,
        config=stack_config_path,
        training_files=[nlu_data_path, stories_path],
    )

    return trained_stack_model_path.model


@pytest.fixture(scope="session")
async def trained_simple_rasa_model(
    trained_async: Callable,
    domain_path: Text,
    nlu_data_path: Text,
    simple_stories_path: Text,
    stack_config_path: Text,
) -> Text:
    trained_stack_model_path = await trained_async(
        domain=domain_path,
        config=stack_config_path,
        training_files=[nlu_data_path, simple_stories_path],
    )

    return trained_stack_model_path


# TODO: JUZL:
@pytest.fixture(scope="session")
def unpacked_trained_rasa_model(
    trained_rasa_model: Text,
) -> Generator[Text, None, None]:
    # with get_model(trained_rasa_model) as path:
    #     yield path
    pass


@pytest.fixture(scope="session")
async def trained_core_model(
    trained_async: Callable,
    domain_path: Text,
    stack_config_path: Text,
    stories_path: Text,
) -> Text:
    trained_core_model_path = await trained_async(
        domain=domain_path, config=stack_config_path, training_files=[stories_path],
    )

    return trained_core_model_path


@pytest.fixture(scope="session")
async def trained_nlu_model(
    trained_async: Callable,
    domain_path: Text,
    nlu_data_path: Text,
    stack_config_path: Text,
) -> Text:
    trained_nlu_model_path = await trained_async(
        domain=domain_path, config=stack_config_path, training_files=[nlu_data_path],
    )

    return trained_nlu_model_path


@pytest.fixture(scope="session")
def _trained_e2e_model_cache(tmp_path_factory: TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("cache")


@pytest.fixture()
def trained_e2e_model_cache(
    _trained_e2e_model_cache: Path,
    tmp_path_factory: TempPathFactory,
    monkeypatch: MonkeyPatch,
) -> Path:
    copied_cache = tmp_path_factory.mktemp("copy")
    rasa.utils.common.copy_directory(_trained_e2e_model_cache, copied_cache)

    monkeypatch.setattr(
        LocalTrainingCache, "_get_cache_location", Mock(return_value=copied_cache)
    )

    return copied_cache


@pytest.fixture(scope="session")
async def trained_e2e_model(
    trained_async: Callable,
    moodbot_domain_path: Text,
    e2e_bot_config_file: Path,
    nlu_data_path: Text,
    e2e_stories_path: Text,
    _trained_e2e_model_cache: Path,
) -> Text:
    return await trained_async(
        domain=moodbot_domain_path,
        config=str(e2e_bot_config_file),
        training_files=[nlu_data_path, e2e_stories_path],
        cache_dir=_trained_e2e_model_cache,
    )


# @pytest.fixture()
# def set_cache(monkeypatch: MonkeyPatch) -> Callable:
#     def set_cache(directory: Path) -> None:
#         monkeypatch.setattr(LocalTrainingCache, "._get_cache_location", directory)
#
#     return set_cache


@pytest.fixture(scope="session")
def moodbot_domain_path() -> Path:
    return Path("data", "test_moodbot", "domain.yml")


@pytest.fixture(scope="session")
def moodbot_domain(moodbot_domain_path: Path) -> Domain:
    return Domain.load(moodbot_domain_path)


@pytest.fixture(scope="session")
def moodbot_nlu_data_path() -> Path:
    return Path(os.getcwd()) / "data" / "test_moodbot" / "data" / "nlu.yml"


@pytest.fixture
def rasa_server(stack_agent: Agent) -> Sanic:
    app = server.create_app(agent=stack_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_non_trained_server(empty_agent: Agent) -> Sanic:
    app = server.create_app(agent=empty_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_core_server(core_agent: Agent) -> Sanic:
    app = server.create_app(agent=core_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_nlu_server(nlu_agent: Agent) -> Sanic:
    app = server.create_app(agent=nlu_agent)
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_server_secured(default_agent: Agent) -> Sanic:
    app = server.create_app(agent=default_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_non_trained_server_secured(empty_agent: Agent) -> Sanic:
    app = server.create_app(agent=empty_agent, auth_token="rasa", jwt_secret="core")
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture
def rasa_server_without_api() -> Sanic:
    app = rasa.core.run._create_app_without_api()
    channel.register([RestInput()], app, "/webhooks/")
    return app


@pytest.fixture(scope="session")
def project() -> Text:
    import tempfile
    from rasa.cli.scaffold import create_initial_project

    directory = tempfile.mkdtemp()
    create_initial_project(directory)

    return directory


@pytest.fixture(scope="session")
def component_builder():
    return ComponentBuilder()


@pytest.fixture(scope="session")
def spacy_nlp() -> Language:
    spacy_provider = SpacyNLPGraphComponent.create(
        {"model": "en_core_web_md"}, Mock(), Mock(), Mock()
    )

    return spacy_provider.provide().model


@pytest.fixture(scope="session")
def blank_config() -> RasaNLUModelConfig:
    return RasaNLUModelConfig({"language": "en", "pipeline": []})


@pytest.fixture(scope="session")
async def trained_response_selector_bot(trained_async: Callable) -> Path:
    zipped_model = await trained_async(
        domain="data/test_response_selector_bot/domain.yml",
        config="data/test_response_selector_bot/config.yml",
        training_files=[
            "data/test_response_selector_bot/data/rules.yml",
            "data/test_response_selector_bot/data/stories.yml",
            "data/test_response_selector_bot/data/nlu.yml",
        ],
    )

    if not zipped_model:
        raise RasaException("Model training for responseselectorbot failed.")

    return Path(zipped_model)


@pytest.fixture(scope="session")
def e2e_bot_domain_file() -> Path:
    return Path("data/test_e2ebot/domain.yml")


@pytest.fixture(scope="session")
def e2e_bot_config_file() -> Path:
    return Path("data/test_e2ebot/config.yml")


@pytest.fixture(scope="session")
def e2e_bot_training_files() -> List[Path]:
    return [
        Path("data/test_e2ebot/data/rules.yml"),
        Path("data/test_e2ebot/data/stories.yml"),
        Path("data/test_e2ebot/data/nlu.yml"),
    ]


@pytest.fixture(scope="session")
def e2e_bot_test_stories_with_unknown_bot_utterances() -> Path:
    return Path("data/test_e2ebot/tests/test_stories_with_unknown_bot_utterances.yml")


@pytest.fixture(scope="session")
async def e2e_bot(
    trained_async: Callable,
    e2e_bot_domain_file: Path,
    e2e_bot_config_file: Path,
    e2e_bot_training_files: List[Path],
) -> Path:
    zipped_model = await trained_async(
        domain=e2e_bot_domain_file,
        config=e2e_bot_config_file,
        training_files=e2e_bot_training_files,
    )

    if not zipped_model:
        raise RasaException("Model training for e2ebot failed.")

    return Path(zipped_model)


@pytest.fixture(scope="module")
def response_selector_agent(trained_response_selector_bot: Path,) -> Agent:
    return Agent.load_local_model(str(trained_response_selector_bot))


@pytest.fixture(scope="module")
def response_selector_interpreter(response_selector_agent: Agent,) -> Interpreter:
    return response_selector_agent.interpreter.interpreter


@pytest.fixture(scope="module")
def e2e_bot_agent(e2e_bot: Path) -> Agent:
    return Agent.load_local_model(str(e2e_bot))


def write_endpoint_config_to_yaml(
    path: Path, data: Dict[Text, Any], endpoints_filename: Text = "endpoints.yml"
) -> Path:
    endpoints_path = path / endpoints_filename

    # write endpoints config to file
    rasa.shared.utils.io.write_yaml(data, endpoints_path)
    return endpoints_path


def random_user_uttered_event(timestamp: Optional[float] = None) -> UserUttered:
    return UserUttered(
        uuid.uuid4().hex,
        timestamp=timestamp if timestamp is not None else random.random(),
    )


def pytest_runtest_setup(item: Function) -> None:
    if (
        "skip_on_windows" in [mark.name for mark in item.iter_markers()]
        and sys.platform == "win32"
    ):
        pytest.skip("cannot run on Windows")


class MockExporter(Exporter):
    """Mocked `Exporter` class."""

    def __init__(
        self,
        tracker_store: TrackerStore = Mock(),
        event_broker: EventBroker = Mock(),
        endpoints_path: Text = "",
    ) -> None:
        super().__init__(tracker_store, event_broker, endpoints_path)


class AsyncMock(Mock):
    """Helper class to mock async functions and methods."""

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return super().__call__(*args, **kwargs)


def _get_marker_for_ci_matrix(item: Function) -> Text:
    """Returns pytest marker which is used to parallelize the tests in GitHub actions.

    Args:
        item: The test case.

    Returns:
        A marker for this test based on which directory / python module the test is in.
    """
    test_path = Path(item.fspath).absolute()

    matching_markers = [
        marker
        for marker, paths_for_marker in PATH_PYTEST_MARKER_MAPPINGS.items()
        if any(
            path == test_path or path in test_path.parents for path in paths_for_marker
        )
    ]

    if not matching_markers:
        return "category_other_unit_tests"
    if len(matching_markers) > 1:
        raise ValueError(
            f"Each test should only be in one category. Test '{item.name}' is assigned "
            f"to these categories: {matching_markers}. Please fix the "
            "mapping in `PATH_PYTEST_MARKER_MAPPINGS`."
        )

    return matching_markers[0]


def pytest_collection_modifyitems(items: List[Function]) -> None:
    """Adds pytest markers dynamically when the tests are run.

    This is automatically called by pytest during its execution.

    Args:
        items: Tests to be run.
    """
    for item in items:
        marker = _get_marker_for_ci_matrix(item)
        item.add_marker(marker)


def create_test_file_with_size(directory: Path, size_in_mb: float) -> None:
    with open(directory / f"{uuid.uuid4().hex}", mode="wb") as f:
        f.seek(int(1024 * 1024 * size_in_mb))
        f.write(b"\0")


@pytest.fixture()
def default_model_storage(tmp_path: Path) -> ModelStorage:
    return LocalModelStorage.create(tmp_path)


@pytest.fixture()
def default_execution_context() -> ExecutionContext:
    return ExecutionContext(GraphSchema({}), uuid.uuid4().hex)


# @pytest.fixture(autouse=True)
# def use_temp_dir_for_cache(
#     monkeypatch: MonkeyPatch, tmp_path_factory: TempdirFactory
# ) -> Path:
#     cache_dir = tmp_path_factory.mktemp(uuid.uuid4().hex)
#     monkeypatch.setattr(
#         LocalTrainingCache,
#         LocalTrainingCache._get_cache_location.__name__,
#         Mock(cache_dir),
#     )
#
#     return cache_dir
