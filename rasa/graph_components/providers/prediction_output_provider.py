from __future__ import annotations
import logging

from rasa.core.policies.policy import PolicyPrediction
from rasa.shared.core.events import UserUttered
from typing import Dict, Optional, Text, Any, List, Tuple

from rasa.core.channels.channel import UserMessage

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)


# TODO: JUZL: test
class PredictionOutputProvider(GraphComponent):
    """Provides the a unified output for model predictions."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> PredictionOutputProvider:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def provide(
        self, **kwargs
    ) -> Tuple[
        Optional[Message], Optional[DialogueStateTracker], Optional[PolicyPrediction]
    ]:
        """# TODO: JUZL: 
        """
        parsed_messages: List[Message] = kwargs.get("parsed_messages")
        parsed_message = parsed_messages[0] if parsed_messages else None

        tracker: DialogueStateTracker = kwargs.get("tracker_with_added_message")

        ensemble_output: Tuple[DialogueStateTracker, PolicyPrediction] = kwargs.get("ensemble_output")
        policy_prediction = None
        if ensemble_output:
            tracker, policy_prediction = ensemble_output


        return parsed_message, tracker, policy_prediction

