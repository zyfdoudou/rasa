import glob
import logging
import os
from typing import (
    Text,
    Optional,
)

from rasa.shared.constants import DEFAULT_MODELS_PATH

from rasa.exceptions import ModelNotFound


logger = logging.getLogger(__name__)


def get_local_model(model_path: Text = DEFAULT_MODELS_PATH) -> Text:
    """Returns verified path to local model archive.

    Args:
        model_path: Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Returns:
        Path to the zipped model. If it's a directory, the latest
                    trained model is returned.

    Raises:
        ModelNotFound Exception: When no model could be found at the provided path.

    """
    if not model_path:
        raise ModelNotFound("No path specified.")
    elif not os.path.exists(model_path):
        raise ModelNotFound(f"No file or directory at '{model_path}'.")

    if os.path.isdir(model_path):
        model_path = get_latest_model(model_path)
        if not model_path:
            raise ModelNotFound(
                f"Could not find any Rasa model files in '{model_path}'."
            )
    elif not model_path.endswith(".tar.gz"):
        raise ModelNotFound(f"Path '{model_path}' does not point to a Rasa model file.")

    return model_path


def get_latest_model(model_path: Text = DEFAULT_MODELS_PATH) -> Optional[Text]:
    """Get the latest model from a path.

    Args:
        model_path: Path to a directory containing zipped models.

    Returns:
        Path to latest model in the given directory.

    """
    if not os.path.exists(model_path) or os.path.isfile(model_path):
        model_path = os.path.dirname(model_path)

    list_of_files = glob.glob(os.path.join(model_path, "*.tar.gz"))

    if len(list_of_files) == 0:
        return None

    return max(list_of_files, key=os.path.getctime)
