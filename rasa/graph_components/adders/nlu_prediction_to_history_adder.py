from __future__ import annotations
import logging
from rasa.shared.core.events import UserUttered
from typing import Dict, Optional, Text, Any, List

from rasa.core.channels.channel import UserMessage

from rasa.engine.graph import GraphComponent, ExecutionContext
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.constants import ENTITIES, INTENT, TEXT
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.core.trackers import DialogueStateTracker
from rasa.shared.core.domain import Domain

logger = logging.getLogger(__name__)


class NLUPredictionToHistoryAdder(GraphComponent):
    """Adds NLU predictions to DialogueStateTracker."""

    @classmethod
    def create(
        cls,
        config: Dict[Text, Any],
        model_storage: ModelStorage,
        resource: Resource,
        execution_context: ExecutionContext,
    ) -> NLUPredictionToHistoryAdder:
        """Creates component (see parent class for full docstring)."""
        return cls()

    def add(
        self,
        predictions: List[Message],
        tracker: Optional[DialogueStateTracker],
        original_messages: List[UserMessage],
        domain: Optional[Domain] = None,
    ) -> Optional[DialogueStateTracker]:
        """Adds NLU predictions to the tracker.

        Args:
            predictions: A list of NLU predictions wrapped as Messages
            tracker: The tracker the predictions should be attached to
            domain: The domain of the model.
            original_messages: An original message from the user with
                extra metadata to annotate the predictions (e.g. channel)

        Returns:
            The original tracker updated with events created from the predictions
        """
        # TODO: JUZL: test this
        if not tracker:
            return None

        for message, original_message in zip(predictions, original_messages):
            user_event = UserUttered(
                message.data.get(TEXT),
                message.data.get(INTENT),
                message.data.get(ENTITIES),
                message.as_dict(only_output_properties=True),
                input_channel=original_message.input_channel,
                message_id=message.data.get("message_id"),
                metadata=original_message.metadata,
            )
            tracker.update(user_event, domain)

            if user_event.entities:
                # Log currently set slots
                slot_values = "\n".join(
                    [f"\t{s.name}: {s.value}" for s in tracker.slots.values()]
                )
                if slot_values.strip():
                    logger.debug(f"Current slot values: \n{slot_values}")

        logger.debug(
            f"Logged {len(predictions)} UserUtterance(s) - \
                tracker now has {len(tracker.events)} events."
        )

        return tracker
