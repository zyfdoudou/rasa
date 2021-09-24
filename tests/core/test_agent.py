import asyncio
from pathlib import Path
from typing import Any, Dict, Text, List, Callable, Optional
from unittest import mock
from unittest.mock import Mock
import uuid

import pytest
from _pytest.logging import LogCaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from pytest_sanic.utils import TestClient
from sanic import Sanic, response
from sanic.request import Request
from sanic.response import StreamingHTTPResponse

import rasa.core
from rasa.engine.storage.storage import ModelMetadata
from rasa.exceptions import ModelNotFound
from rasa.shared.core.events import (
    ActionExecuted, BotUttered, DefinePrevUserUtteredFeaturization,
    SessionStarted, UserUtteranceReverted, UserUttered,
)
import rasa.shared.utils.common
from rasa.core.policies.rule_policy import RulePolicy
import rasa.utils.io
from rasa.core import jobs
from rasa.core.agent import Agent, load_agent
from rasa.core.channels.channel import UserMessage
from rasa.shared.core.domain import InvalidDomain, Domain
from rasa.shared.constants import INTENT_MESSAGE_PREFIX
from rasa.core.policies.ensemble import PolicyEnsemble
from rasa.utils.endpoints import EndpointConfig


def model_server_app(model_path: Text, model_hash: Text = "somehash") -> Sanic:
    app = Sanic(__name__)
    app.number_of_model_requests = 0

    @app.route("/model", methods=["GET"])
    async def model(request: Request) -> StreamingHTTPResponse:
        """Simple HTTP model server responding with a trained model."""

        if model_hash == request.headers.get("If-None-Match"):
            return response.text("", 204)

        app.number_of_model_requests += 1

        return await response.file_stream(
            location=model_path,
            headers={"ETag": model_hash, "filename": Path(model_path).name},
            mime_type="application/gzip",
        )

    return app


@pytest.fixture()
def model_server(
    loop: asyncio.AbstractEventLoop, sanic_client: Callable, trained_rasa_model: Text
) -> TestClient:
    app = model_server_app(trained_rasa_model, model_hash="somehash")
    return loop.run_until_complete(sanic_client(app))


async def test_agent_train(default_agent: Agent):
    domain = Domain.load("data/test_domains/default_with_slots.yml")

    assert default_agent.domain.action_names_or_texts == domain.action_names_or_texts
    assert default_agent.domain.intents == domain.intents
    assert default_agent.domain.entities == domain.entities
    assert default_agent.domain.responses == domain.responses
    assert [s.name for s in default_agent.domain.slots] == [
        s.name for s in domain.slots
    ]

    assert default_agent.processor
    assert default_agent.graph_runner


@pytest.mark.parametrize(
    "text_message_data, expected",
    [
        (
            '/greet{"name":"Rasa"}',
            {
                "text": '/greet{"name":"Rasa"}',
                "intent": {"name": "greet", "confidence": 1.0},
                "intent_ranking": [{"name": "greet", "confidence": 1.0}],
                "entities": [
                    {"entity": "name", "start": 6, "end": 21, "value": "Rasa"}
                ],
            },
        ),
    ],
)
async def test_agent_parse_message_using_nlu_interpreter(
    default_agent: Agent, text_message_data: Text, expected: Dict[Text, Any]
):
    result = await default_agent.parse_message(text_message_data)
    assert result == expected


async def test_agent_handle_text(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    result = await default_agent.handle_text(text, sender_id="test_agent_handle_text")
    assert result == [
        {"recipient_id": "test_agent_handle_text", "text": "hey there Rasa!"}
    ]


async def test_agent_handle_message(default_agent: Agent):
    text = INTENT_MESSAGE_PREFIX + 'greet{"name":"Rasa"}'
    message = UserMessage(text, sender_id="test_agent_handle_message")
    result = await default_agent.handle_message(message)
    assert result == [
        {"recipient_id": "test_agent_handle_message", "text": "hey there Rasa!"}
    ]


def test_agent_wrong_use_of_load():
    training_data_file = "data/test_moodbot/data/stories.yml"

    with pytest.raises(ModelNotFound):
        # try to load a model file from a data path, which is nonsense and
        # should fail properly
        Agent.load(training_data_file)


async def test_agent_with_model_server_in_thread(
    model_server: TestClient, domain: Domain, unpacked_trained_rasa_model: Text
):
    model_endpoint_config = EndpointConfig.from_dict(
        {"url": model_server.make_url("/model"), "wait_time_between_pulls": 2}
    )

    agent = Agent()
    agent = await rasa.core.agent.load_from_server(
        agent, model_server=model_endpoint_config
    )

    await asyncio.sleep(5)

    assert agent.fingerprint == "somehash"
    assert agent.domain.as_dict() == domain.as_dict()
    assert agent.graph_runner

    assert model_server.app.number_of_model_requests == 1
    jobs.kill_scheduler()


async def test_wait_time_between_pulls_without_interval(
    model_server: TestClient, monkeypatch: MonkeyPatch
):
    monkeypatch.setattr(
        "rasa.core.agent.schedule_model_pulling", lambda *args: 1 / 0
    )  # will raise an exception

    model_endpoint_config = EndpointConfig.from_dict(
        {"url": model_server.make_url("/model"), "wait_time_between_pulls": None}
    )

    agent = Agent()
    # should not call schedule_model_pulling, if it does, this will raise
    await rasa.core.agent.load_from_server(agent, model_server=model_endpoint_config)


async def test_load_agent(trained_rasa_model: Text):
    agent = await load_agent(model_path=trained_rasa_model)

    assert agent.tracker_store is not None
    assert agent.lock_store is not None
    assert agent.processor is not None
    assert agent.graph_runner is not None


# TODO: JUZL: more load_agent tests


# TODO: JUZL: handle_message with nlu only model
async def test_agent_handle_message_with_nlu_only_model(trained_nlu_model: Text):
    agent = await load_agent(model_path=trained_nlu_model)
    assert agent.domain is not None
    sender_id = "test_sender_id"
    message = UserMessage("hello", sender_id=sender_id)
    await agent.handle_message(message)
    tracker = agent.tracker_store.get_or_create_tracker(sender_id)

    # UserUttered event was added to tracker, with correct intent data
    assert tracker.events[3].intent["name"] == "greet"


async def test_load_agent_on_not_existing_path():
    agent = await load_agent(model_path="some-random-path")

    assert agent is None


@pytest.mark.parametrize(
    "model_path",
    [
        "non-existing-path",
        "data/test_domains/default_with_slots.yml",
        "not-existing-model.tar.gz",
        None,
    ],
)
async def test_agent_load_on_invalid_model_path(model_path: Optional[Text]):
    with pytest.raises(ModelNotFound):
        Agent.load(model_path)


async def test_agent_handle_message_2_normal(default_agent: Agent):
    sender_id = uuid.uuid4().hex
    message = UserMessage("hello", sender_id=sender_id)
    await default_agent.handle_message(message)
    tracker = default_agent.tracker_store.get_or_create_tracker(sender_id)
    expected_events = [
        ActionExecuted(action_name="action_session_start"),
        SessionStarted(),
        ActionExecuted(action_name="action_listen"),
        UserUttered(),
        DefinePrevUserUtteredFeaturization(False),
        ActionExecuted(action_name="utter_greet"),  # TODO: JUZL: why?
        BotUttered('hey there None!'),
        ActionExecuted(action_name="action_listen"),
    ]
    assert len(tracker.events) == len(expected_events)
    for e1, e2 in zip(tracker.events, expected_events):
        assert e1.__class__ == e2.__class__



async def test_agent_handle_message_2_only_nlu(trained_nlu_model: Text):
    agent = await load_agent(model_path=trained_nlu_model)
    sender_id = uuid.uuid4().hex
    message = UserMessage("hello", sender_id=sender_id)
    await agent.handle_message(message)
    tracker = agent.tracker_store.get_or_create_tracker(sender_id)
    expected_events = [
        ActionExecuted(action_name="action_session_start"),
        SessionStarted(),
        ActionExecuted(action_name="action_listen"),
        UserUttered(),
    ]
    assert len(tracker.events) == len(expected_events)
    for e1, e2 in zip(tracker.events, expected_events):
        assert e1.__class__ == e2.__class__  # TODO: JUZL: assert more than class


async def test_agent_handle_message_2_only_core(trained_core_model: Text):
    agent = await load_agent(model_path=trained_core_model)
    sender_id = uuid.uuid4().hex
    message = UserMessage("/greet", sender_id=sender_id)
    await agent.handle_message(message)
    tracker = agent.tracker_store.get_or_create_tracker(sender_id)
    expected_events = [
        ActionExecuted(action_name="action_session_start"),
        SessionStarted(),
        ActionExecuted(action_name="action_listen"),
        UserUttered(),
        DefinePrevUserUtteredFeaturization(False),
        ActionExecuted(action_name="utter_greet"),
        BotUttered(),
        ActionExecuted(action_name="action_listen"),
    ]
    import ipdb; ipdb.set_trace()
    assert len(tracker.events) == len(expected_events)
    for e1, e2 in zip(tracker.events, expected_events):
        assert e1.__class__ == e2.__class__  # TODO: JUZL: assert more than class


async def test_agent_handle_message_2_only_core_no_message(trained_core_model: Text):
    agent = await load_agent(model_path=trained_core_model)
    sender_id = uuid.uuid4().hex
    message = UserMessage("/greet", sender_id=sender_id)
    await agent.handle_message(message)
    tracker = agent.tracker_store.get_or_create_tracker(sender_id)
    expected_events = [
        ActionExecuted(action_name="action_session_start"),
        SessionStarted(),
        ActionExecuted(action_name="action_listen"),
        UserUttered(),
        DefinePrevUserUtteredFeaturization(False),
        ActionExecuted(action_name="utter_greet"),
        BotUttered(),
        ActionExecuted(action_name="action_listen"),
    ]
    import ipdb; ipdb.set_trace()
    assert len(tracker.events) == len(expected_events)
    for e1, e2 in zip(tracker.events, expected_events):
        assert e1.__class__ == e2.__class__  # TODO: JUZL: assert more than class

