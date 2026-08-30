import os
import re
from copy import deepcopy
from typing import Dict, List, Optional

from modules.utils.constants import AUTOMATIC_DETECTION, GRADIO_NONE_NUMBER_MAX
from modules.utils.files_manager import load_yaml, save_yaml
from modules.utils.paths import PRESETS_DIR
from modules.whisper.data_classes import (BGMSeparationParams, DiarizationParams, VadParams, WhisperParams)

PRESET_EXTENSION = ".yaml"

# The pipeline inputs are passed around as a flat list of gradio values, ordered the same way as
# `TranscriptionPipelineParams.to_list()` : one section after another, each in `model_fields` order.
PIPELINE_SECTIONS = (
    ("whisper", WhisperParams),
    ("vad", VadParams),
    ("diarization", DiarizationParams),
    ("bgm_separation", BGMSeparationParams),
)


def sanitize_preset_name(name: str) -> str:
    """A preset name is used as a file name, so keep it to something safe."""
    name = os.path.basename(name.strip()) if isinstance(name, str) else ""
    name = re.sub(r"[^\w\-. ]", "_", name).strip(" .")
    if not name:
        raise ValueError("Please enter a valid preset name.")
    return name


def get_preset_path(name: str) -> str:
    return os.path.join(PRESETS_DIR, sanitize_preset_name(name) + PRESET_EXTENSION)


def list_presets() -> List[str]:
    if not os.path.isdir(PRESETS_DIR):
        return []
    return sorted(os.path.splitext(file)[0] for file in os.listdir(PRESETS_DIR)
                  if file.endswith(PRESET_EXTENSION))


def load_preset(name: str) -> Dict:
    path = get_preset_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"There's no preset named \"{name}\".")
    return load_yaml(path) or {}


def save_preset(name: str, data: Dict) -> str:
    os.makedirs(PRESETS_DIR, exist_ok=True)
    return save_yaml(data, get_preset_path(name))


def delete_preset(name: str) -> None:
    path = get_preset_path(name)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"There's no preset named \"{name}\".")
    os.remove(path)


def pipeline_values_to_preset(pipeline_values: List,
                              file_format: Optional[str] = None,
                              add_timestamp: Optional[bool] = None) -> Dict:
    """
    Convert the values that are currently in the UI into a preset.

    The values are stored exactly as the gradio components hold them, so a preset file has the same
    schema as `configs/default_parameters.yaml` and loading one back restores the UI as it was.
    """
    preset, index = {}, 0
    for section, params in PIPELINE_SECTIONS:
        field_names = list(params.model_fields.keys())
        preset[section] = dict(zip(field_names, pipeline_values[index:index + len(field_names)]))
        index += len(field_names)

    if file_format is not None:
        preset["whisper"]["file_format"] = file_format
    if add_timestamp is not None:
        preset["whisper"]["add_timestamp"] = add_timestamp
    return preset


def preset_to_pipeline_values(preset: Dict, current_values: List) -> List:
    """
    Convert a preset back into the flat list of gradio values.

    Fields that the preset doesn't have keep the value they currently have in the UI, so that a
    preset written by an older version doesn't wipe the parameters it doesn't know about.
    """
    preset = normalize_preset(preset)

    values, index = [], 0
    for section, params in PIPELINE_SECTIONS:
        cached = preset.get(section) or {}
        for field_name in params.model_fields:
            values.append(cached[field_name] if field_name in cached else current_values[index])
            index += 1
    return values


def keep_available_choices(components: List, values: List, current_values: List) -> List:
    """
    A preset that was copied from another machine can name a device, a compute type or a model size
    that this one doesn't have. Those keep the value the UI already has instead of being restored.
    """
    for index, component in enumerate(components):
        choices = [choice for _, choice in getattr(component, "choices", None) or []]
        if (choices and not getattr(component, "allow_custom_value", False)
                and values[index] not in choices):
            values[index] = current_values[index]
    return values


def normalize_preset(preset: Dict) -> Dict:
    """
    Convert the values that can't be displayed as they are in the UI, mirroring
    `BaseTranscriptionPipeline.cache_parameters()`. Only hand written presets need this.
    """
    preset = deepcopy(preset)
    whisper_preset, vad_preset = preset.get("whisper") or {}, preset.get("vad") or {}

    if "lang" in whisper_preset and whisper_preset["lang"] is None:
        whisper_preset["lang"] = AUTOMATIC_DETECTION.unwrap()

    if isinstance(whisper_preset.get("suppress_tokens", None), list):
        whisper_preset["suppress_tokens"] = str(whisper_preset["suppress_tokens"])

    if vad_preset.get("max_speech_duration_s", None) == float("inf"):
        vad_preset["max_speech_duration_s"] = GRADIO_NONE_NUMBER_MAX

    return preset
