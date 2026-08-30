import os
import pytest

from modules.utils.constants import AUTOMATIC_DETECTION, GRADIO_NONE_NUMBER_MAX, GRADIO_NONE_STR
from modules.utils.paths import PRESETS_DIR
from modules.utils.preset_manager import *
from modules.whisper.data_classes import (BGMSeparationParams, DiarizationParams, TranscriptionPipelineParams,
                                          VadParams, WhisperParams)

TEST_PRESET_NAME = "test-preset"


def ui_values(**overrides):
    """The values as the gradio components hold them, which is what a preset is made of."""
    whisper = WhisperParams().to_dict()
    whisper.update(model_size="large-v3", lang="japanese", compute_type="int8_float32",
                   condition_on_previous_text=False, compression_ratio_threshold=2.0,
                   hallucination_silence_threshold=2.0, initial_prompt=GRADIO_NONE_STR,
                   prefix=GRADIO_NONE_STR, hotwords=GRADIO_NONE_STR, suppress_tokens="[-1]",
                   max_new_tokens=0)
    vad = VadParams().to_dict()
    vad.update(vad_filter=True, max_speech_duration_s=30)
    diarization = DiarizationParams().to_dict()
    bgm_separation = BGMSeparationParams().to_dict()
    bgm_separation.update(is_separate_bgm=True)

    values = list(whisper.values()) + list(vad.values()) + list(diarization.values()) + \
        list(bgm_separation.values())
    for index, value in overrides.items():
        values[index] = value
    return values


@pytest.fixture(autouse=True)
def clean_preset():
    yield
    path = os.path.join(PRESETS_DIR, TEST_PRESET_NAME + PRESET_EXTENSION)
    if os.path.exists(path):
        os.remove(path)


def test_preset_round_trip():
    values = ui_values()

    save_preset(TEST_PRESET_NAME, pipeline_values_to_preset(values, file_format="WebVTT",
                                                            add_timestamp=True))
    assert TEST_PRESET_NAME in list_presets()

    preset = load_preset(TEST_PRESET_NAME)
    assert preset["whisper"]["file_format"] == "WebVTT"
    assert preset["whisper"]["add_timestamp"] is True
    assert preset["whisper"]["lang"] == "japanese"
    assert preset["vad"]["max_speech_duration_s"] == 30
    assert preset["bgm_separation"]["is_separate_bgm"] is True

    # Every widget is restored to what it was, without going through the UI's current values
    assert preset_to_pipeline_values(preset, [None] * len(values)) == values

    delete_preset(TEST_PRESET_NAME)
    assert TEST_PRESET_NAME not in list_presets()


def test_preset_is_usable_by_the_pipeline():
    """A preset is made of raw UI values, so it has to survive the same conversion a job does."""
    values = preset_to_pipeline_values(
        pipeline_values_to_preset(ui_values()), [None] * len(ui_values())
    )
    params = TranscriptionPipelineParams.from_list(values)

    assert params.whisper.model_size == "large-v3"
    assert params.whisper.suppress_tokens == [-1]
    assert params.vad.max_speech_duration_s == 30
    assert params.bgm_separation.is_separate_bgm is True


def test_preset_keeps_current_values_for_unknown_fields():
    """A preset written by an older version shouldn't wipe the parameters it doesn't know about."""
    preset = pipeline_values_to_preset(ui_values())
    del preset["whisper"]["model_size"]
    del preset["vad"]
    del preset["bgm_separation"]["is_separate_bgm"]

    current = ui_values()
    current[0] = "tiny"  # whisper.model_size
    restored = preset_to_pipeline_values(preset, current)

    assert restored[0] == "tiny"
    vad_index = len(WhisperParams.model_fields)
    assert restored[vad_index:vad_index + len(VadParams.model_fields)] == \
        current[vad_index:vad_index + len(VadParams.model_fields)]
    assert restored == current


def test_hand_written_preset_is_normalized_for_the_ui():
    preset = pipeline_values_to_preset(ui_values())
    preset["whisper"]["lang"] = None
    preset["whisper"]["suppress_tokens"] = [-1]
    preset["vad"]["max_speech_duration_s"] = float("inf")

    values = preset_to_pipeline_values(preset, [None] * len(ui_values()))
    field_names = list(WhisperParams.model_fields.keys())

    assert values[field_names.index("lang")] == AUTOMATIC_DETECTION.unwrap()
    assert values[field_names.index("suppress_tokens")] == "[-1]"
    assert values[len(field_names) + list(VadParams.model_fields.keys()).index(
        "max_speech_duration_s")] == GRADIO_NONE_NUMBER_MAX


def test_preset_name_is_safe_to_use_as_a_file_name():
    assert sanitize_preset_name("  music anime  ") == "music anime"
    assert sanitize_preset_name("../../etc/passwd") == "passwd"
    assert os.path.dirname(get_preset_path("a/b")) == PRESETS_DIR

    for invalid_name in ["", "   ", None, "..", "/"]:
        with pytest.raises(ValueError):
            sanitize_preset_name(invalid_name)


def test_missing_preset_raises():
    with pytest.raises(FileNotFoundError):
        load_preset("this-preset-does-not-exist")
    with pytest.raises(FileNotFoundError):
        delete_preset("this-preset-does-not-exist")


def test_pipeline_inputs_match_the_preset_sections():
    """
    A preset maps the flat list of gradio values onto the params by position, so the widgets have to
    stay in `model_fields` order. `TranscriptionPipelineParams.from_list()` relies on this too.
    """
    whisper_inputs = WhisperParams.to_gradio_inputs(defaults={}, only_advanced=True,
                                                    available_compute_types=["float32"],
                                                    compute_type="float32")
    # `only_advanced` skips model_size, lang and is_translate, which the UI builds separately
    assert 3 + len(whisper_inputs) == len(WhisperParams.model_fields)
    assert len(VadParams.to_gradio_inputs(defaults={})) == len(VadParams.model_fields)
    assert len(DiarizationParams.to_gradio_inputs(defaults={}, available_devices=["cpu"],
                                                  device="cpu")) == len(DiarizationParams.model_fields)
    assert len(BGMSeparationParams.to_gradio_input(defaults={}, available_devices=["cpu"], device="cpu",
                                                   available_models=["UVR-MDX-NET-Inst_HQ_4"])) == \
        len(BGMSeparationParams.model_fields)


def test_values_this_machine_cant_offer_are_left_alone():
    """A preset copied from a machine with a GPU shouldn't put "cuda" in a CPU only dropdown."""
    import gradio as gr

    components = [gr.Dropdown(choices=["cpu", "cuda"], value="cpu"),
                  gr.Dropdown(choices=["float32"], value="float32"),
                  gr.Dropdown(choices=["large-v2"], value="large-v2", allow_custom_value=True),
                  gr.Number(value=5)]
    values = keep_available_choices(components, ["cuda", "int8_float32", "large-v3", 3],
                                    ["cpu", "float32", "large-v2", 5])

    assert values == ["cuda", "float32", "large-v3", 3]
