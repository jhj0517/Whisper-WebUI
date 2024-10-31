import faster_whisper.transcribe
import gradio as gr
import torch
from typing import Optional, Dict, List, Union, NamedTuple
from pydantic import BaseModel, Field, field_validator, ConfigDict
from gradio_i18n import Translate, gettext as _
from enum import Enum
from copy import deepcopy

import yaml

from modules.utils.constants import *


class WhisperImpl(Enum):
    WHISPER = "whisper"
    FASTER_WHISPER = "faster-whisper"
    INSANELY_FAST_WHISPER = "insanely_fast_whisper"


class Segment(BaseModel):
    id: Optional[int] = Field(default=None, description="Incremental id for the segment")
    seek: Optional[int] = Field(default=None, description="Seek of the segment from chunked audio")
    text: Optional[str] = Field(default=None, description="Transcription text of the segment")
    start: Optional[float] = Field(default=None, description="Start time of the segment")
    end: Optional[float] = Field(default=None, description="End time of the segment")
    tokens: Optional[List[int]] = Field(default=None, description="List of token IDs")
    temperature: Optional[float] = Field(default=None, description="Temperature used during the decoding process")
    avg_logprob: Optional[float] = Field(default=None, description="Average log probability of the tokens")
    compression_ratio: Optional[float] = Field(default=None, description="Compression ratio of the segment")
    no_speech_prob: Optional[float] = Field(default=None, description="Probability that it's not speech")
    words: Optional[List['Word']] = Field(default=None, description="List of words contained in the segment")

    @classmethod
    def from_faster_whisper(cls,
                            seg: faster_whisper.transcribe.Segment):
        if seg.words is not None:
            words = [
                Word(
                    start=w.start,
                    end=w.end,
                    word=w.word,
                    probability=w.probability
                ) for w in seg.words
            ]
        else:
            words = None

        return cls(
            id=seg.id,
            seek=seg.seek,
            text=seg.text,
            start=seg.start,
            end=seg.end,
            tokens=seg.tokens,
            temperature=seg.temperature,
            avg_logprob=seg.avg_logprob,
            compression_ratio=seg.compression_ratio,
            no_speech_prob=seg.no_speech_prob,
            words=words
        )


class Word(BaseModel):
    start: Optional[float] = Field(default=None, description="Start time of the word")
    end: Optional[float] = Field(default=None, description="Start time of the word")
    word: Optional[str] = Field(default=None, description="Word text")
    probability: Optional[float] = Field(default=None, description="Probability of the word")


class BaseParams(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    def to_dict(self) -> Dict:
        return self.model_dump()

    def to_list(self) -> List:
        return list(self.model_dump().values())

    @classmethod
    def from_list(cls, data_list: List) -> 'BaseParams':
        field_names = list(cls.model_fields.keys())
        return cls(**dict(zip(field_names, data_list)))


class VadParams(BaseParams):
    """Voice Activity Detection parameters"""
    vad_filter: bool = Field(default=False, description="Enable voice activity detection to filter out non-speech parts")
    threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Speech threshold for Silero VAD. Probabilities above this value are considered speech"
    )
    min_speech_duration_ms: int = Field(
        default=250,
        ge=0,
        description="Final speech chunks shorter than this are discarded"
    )
    max_speech_duration_s: float = Field(
        default=float("inf"),
        gt=0,
        description="Maximum duration of speech chunks in seconds"
    )
    min_silence_duration_ms: int = Field(
        default=2000,
        ge=0,
        description="Minimum silence duration between speech chunks"
    )
    speech_pad_ms: int = Field(
        default=400,
        ge=0,
        description="Padding added to each side of speech chunks"
    )

    @classmethod
    def to_gradio_inputs(cls, defaults: Optional[Dict] = None) -> List[gr.components.base.FormComponent]:
        return [
            gr.Checkbox(
                label=_("Enable Silero VAD Filter"),
                value=defaults.get("vad_filter", cls.__fields__["vad_filter"].default),
                interactive=True,
                info=_("Enable this to transcribe only detected voice")
            ),
            gr.Slider(
                minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold",
                value=defaults.get("threshold", cls.__fields__["threshold"].default),
                info="Lower it to be more sensitive to small sounds."
            ),
            gr.Number(
                label="Minimum Speech Duration (ms)", precision=0,
                value=defaults.get("min_speech_duration_ms", cls.__fields__["min_speech_duration_ms"].default),
                info="Final speech chunks shorter than this time are thrown out"
            ),
            gr.Number(
                label="Maximum Speech Duration (s)",
                value=defaults.get("max_speech_duration_s", GRADIO_NONE_NUMBER_MAX),
                info="Maximum duration of speech chunks in \"seconds\"."
            ),
            gr.Number(
                label="Minimum Silence Duration (ms)", precision=0,
                value=defaults.get("min_silence_duration_ms", cls.__fields__["min_silence_duration_ms"].default),
                info="In the end of each speech chunk wait for this time before separating it"
            ),
            gr.Number(
                label="Speech Padding (ms)", precision=0,
                value=defaults.get("speech_pad_ms", cls.__fields__["speech_pad_ms"].default),
                info="Final speech chunks are padded by this time each side"
            )
        ]


class DiarizationParams(BaseParams):
    """Speaker diarization parameters"""
    is_diarize: bool = Field(default=False, description="Enable speaker diarization")
    device: str = Field(default="cuda", description="Device to run Diarization model.")
    hf_token: str = Field(
        default="",
        description="Hugging Face token for downloading diarization models"
    )

    @classmethod
    def to_gradio_inputs(cls,
                         defaults: Optional[Dict] = None,
                         available_devices: Optional[List] = None,
                         device: Optional[str] = None) -> List[gr.components.base.FormComponent]:
        return [
            gr.Checkbox(
                label=_("Enable Diarization"),
                value=defaults.get("is_diarize", cls.__fields__["is_diarize"].default),
            ),
            gr.Dropdown(
                label=_("Device"),
                choices=["cpu", "cuda"] if available_devices is None else available_devices,
                value=defaults.get("device", device),
            ),
            gr.Textbox(
                label=_("HuggingFace Token"),
                value=defaults.get("hf_token", cls.__fields__["hf_token"].default),
                info=_("This is only needed the first time you download the model")
            ),
        ]


class BGMSeparationParams(BaseParams):
    """Background music separation parameters"""
    is_separate_bgm: bool = Field(default=False, description="Enable background music separation")
    model_size: str = Field(
        default="UVR-MDX-NET-Inst_HQ_4",
        description="UVR model size"
    )
    device: str = Field(default="cuda", description="Device to run UVR model.")
    segment_size: int = Field(
        default=256,
        gt=0,
        description="Segment size for UVR model"
    )
    save_file: bool = Field(
        default=False,
        description="Whether to save separated audio files"
    )
    enable_offload: bool = Field(
        default=True,
        description="Offload UVR model after transcription"
    )

    @classmethod
    def to_gradio_input(cls,
                        defaults: Optional[Dict] = None,
                        available_devices: Optional[List] = None,
                        device: Optional[str] = None,
                        available_models: Optional[List] = None) -> List[gr.components.base.FormComponent]:
        return [
            gr.Checkbox(
                label=_("Enable Background Music Remover Filter"),
                value=defaults.get("is_separate_bgm", cls.__fields__["is_separate_bgm"].default),
                interactive=True,
                info=_("Enabling this will remove background music")
            ),
            gr.Dropdown(
                label=_("Model"),
                choices=["UVR-MDX-NET-Inst_HQ_4",
                         "UVR-MDX-NET-Inst_3"] if available_models is None else available_models,
                value=defaults.get("model_size", cls.__fields__["model_size"].default),
            ),
            gr.Dropdown(
                label=_("Device"),
                choices=["cpu", "cuda"] if available_devices is None else available_devices,
                value=defaults.get("device", device),
            ),
            gr.Number(
                label="Segment Size",
                value=defaults.get("segment_size", cls.__fields__["segment_size"].default),
                precision=0,
                info="Segment size for UVR model"
            ),
            gr.Checkbox(
                label=_("Save separated files to output"),
                value=defaults.get("save_file", cls.__fields__["save_file"].default),
            ),
            gr.Checkbox(
                label=_("Offload sub model after removing background music"),
                value=defaults.get("enable_offload", cls.__fields__["enable_offload"].default),
            )
        ]


class WhisperParams(BaseParams):
    """Whisper parameters"""
    model_size: str = Field(default="large-v2", description="Whisper model size")
    lang: Optional[str] = Field(default=None, description="Source language of the file to transcribe")
    is_translate: bool = Field(default=False, description="Translate speech to English end-to-end")
    beam_size: int = Field(default=5, ge=1, description="Beam size for decoding")
    log_prob_threshold: float = Field(
        default=-1.0,
        description="Threshold for average log probability of sampled tokens"
    )
    no_speech_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Threshold for detecting silence"
    )
    compute_type: str = Field(default="float16", description="Computation type for transcription")
    best_of: int = Field(default=5, ge=1, description="Number of candidates when sampling")
    patience: float = Field(default=1.0, gt=0, description="Beam search patience factor")
    condition_on_previous_text: bool = Field(
        default=True,
        description="Use previous output as prompt for next window"
    )
    prompt_reset_on_temperature: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Temperature threshold for resetting prompt"
    )
    initial_prompt: Optional[str] = Field(default=None, description="Initial prompt for first window")
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        description="Temperature for sampling"
    )
    compression_ratio_threshold: float = Field(
        default=2.4,
        gt=0,
        description="Threshold for gzip compression ratio"
    )
    length_penalty: float = Field(default=1.0, gt=0, description="Exponential length penalty")
    repetition_penalty: float = Field(default=1.0, gt=0, description="Penalty for repeated tokens")
    no_repeat_ngram_size: int = Field(default=0, ge=0, description="Size of n-grams to prevent repetition")
    prefix: Optional[str] = Field(default=None, description="Prefix text for first window")
    suppress_blank: bool = Field(
        default=True,
        description="Suppress blank outputs at start of sampling"
    )
    suppress_tokens: Optional[Union[List[int], str]] = Field(default=[-1], description="Token IDs to suppress")
    max_initial_timestamp: float = Field(
        default=1.0,
        ge=0.0,
        description="Maximum initial timestamp"
    )
    word_timestamps: bool = Field(default=False, description="Extract word-level timestamps")
    prepend_punctuations: Optional[str] = Field(
        default="\"'“¿([{-",
        description="Punctuations to merge with next word"
    )
    append_punctuations: Optional[str] = Field(
        default="\"'.。,，!！?？:：”)]}、",
        description="Punctuations to merge with previous word"
    )
    max_new_tokens: Optional[int] = Field(default=None, description="Maximum number of new tokens per chunk")
    chunk_length: Optional[int] = Field(default=30, description="Length of audio segments in seconds")
    hallucination_silence_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for skipping silent periods in hallucination detection"
    )
    hotwords: Optional[str] = Field(default=None, description="Hotwords/hint phrases for the model")
    language_detection_threshold: Optional[float] = Field(
        default=None,
        description="Threshold for language detection probability"
    )
    language_detection_segments: int = Field(
        default=1,
        gt=0,
        description="Number of segments for language detection"
    )
    batch_size: int = Field(default=24, gt=0, description="Batch size for processing")

    @field_validator('lang')
    def validate_lang(cls, v):
        from modules.utils.constants import AUTOMATIC_DETECTION
        return None if v == AUTOMATIC_DETECTION.unwrap() else v

    @field_validator('suppress_tokens')
    def validate_supress_tokens(cls, v):
        import ast
        try:
            if isinstance(v, str):
                suppress_tokens = ast.literal_eval(v)
                if not isinstance(suppress_tokens, list):
                    raise ValueError("Invalid Suppress Tokens. The value must be type of List[int]")
                return suppress_tokens
            if isinstance(v, list):
                return v
        except Exception as e:
            raise ValueError(f"Invalid Suppress Tokens. The value must be type of List[int]: {e}")

    @classmethod
    def to_gradio_inputs(cls,
                         defaults: Optional[Dict] = None,
                         only_advanced: Optional[bool] = True,
                         whisper_type: Optional[str] = None,
                         available_models: Optional[List] = None,
                         available_langs: Optional[List] = None,
                         available_compute_types: Optional[List] = None,
                         compute_type: Optional[str] = None):
        whisper_type = WhisperImpl.FASTER_WHISPER.value if whisper_type is None else whisper_type.strip().lower()

        inputs = []
        if not only_advanced:
            inputs += [
                gr.Dropdown(
                    label=_("Model"),
                    choices=available_models,
                    value=defaults.get("model_size", cls.__fields__["model_size"].default),
                ),
                gr.Dropdown(
                    label=_("Language"),
                    choices=available_langs,
                    value=defaults.get("lang", AUTOMATIC_DETECTION),
                ),
                gr.Checkbox(
                    label=_("Translate to English?"),
                    value=defaults.get("is_translate", cls.__fields__["is_translate"].default),
                ),
            ]

        inputs += [
            gr.Number(
                label="Beam Size",
                value=defaults.get("beam_size", cls.__fields__["beam_size"].default),
                precision=0,
                info="Beam size for decoding"
            ),
            gr.Number(
                label="Log Probability Threshold",
                value=defaults.get("log_prob_threshold", cls.__fields__["log_prob_threshold"].default),
                info="Threshold for average log probability of sampled tokens"
            ),
            gr.Number(
                label="No Speech Threshold",
                value=defaults.get("no_speech_threshold", cls.__fields__["no_speech_threshold"].default),
                info="Threshold for detecting silence"
            ),
            gr.Dropdown(
                label="Compute Type",
                choices=["float16", "int8", "int16"] if available_compute_types is None else available_compute_types,
                value=defaults.get("compute_type", compute_type),
                info="Computation type for transcription"
            ),
            gr.Number(
                label="Best Of",
                value=defaults.get("best_of", cls.__fields__["best_of"].default),
                precision=0,
                info="Number of candidates when sampling"
            ),
            gr.Number(
                label="Patience",
                value=defaults.get("patience", cls.__fields__["patience"].default),
                info="Beam search patience factor"
            ),
            gr.Checkbox(
                label="Condition On Previous Text",
                value=defaults.get("condition_on_previous_text", cls.__fields__["condition_on_previous_text"].default),
                info="Use previous output as prompt for next window"
            ),
            gr.Slider(
                label="Prompt Reset On Temperature",
                value=defaults.get("prompt_reset_on_temperature",
                                   cls.__fields__["prompt_reset_on_temperature"].default),
                minimum=0,
                maximum=1,
                step=0.01,
                info="Temperature threshold for resetting prompt"
            ),
            gr.Textbox(
                label="Initial Prompt",
                value=defaults.get("initial_prompt", GRADIO_NONE_STR),
                info="Initial prompt for first window"
            ),
            gr.Slider(
                label="Temperature",
                value=defaults.get("temperature", cls.__fields__["temperature"].default),
                minimum=0.0,
                step=0.01,
                maximum=1.0,
                info="Temperature for sampling"
            ),
            gr.Number(
                label="Compression Ratio Threshold",
                value=defaults.get("compression_ratio_threshold",
                                   cls.__fields__["compression_ratio_threshold"].default),
                info="Threshold for gzip compression ratio"
            )
        ]

        faster_whisper_inputs = [
            gr.Number(
                label="Length Penalty",
                value=defaults.get("length_penalty", cls.__fields__["length_penalty"].default),
                info="Exponential length penalty",
            ),
            gr.Number(
                label="Repetition Penalty",
                value=defaults.get("repetition_penalty", cls.__fields__["repetition_penalty"].default),
                info="Penalty for repeated tokens"
            ),
            gr.Number(
                label="No Repeat N-gram Size",
                value=defaults.get("no_repeat_ngram_size", cls.__fields__["no_repeat_ngram_size"].default),
                precision=0,
                info="Size of n-grams to prevent repetition"
            ),
            gr.Textbox(
                label="Prefix",
                value=defaults.get("prefix", GRADIO_NONE_STR),
                info="Prefix text for first window"
            ),
            gr.Checkbox(
                label="Suppress Blank",
                value=defaults.get("suppress_blank", cls.__fields__["suppress_blank"].default),
                info="Suppress blank outputs at start of sampling"
            ),
            gr.Textbox(
                label="Suppress Tokens",
                value=defaults.get("suppress_tokens", "[-1]"),
                info="Token IDs to suppress"
            ),
            gr.Number(
                label="Max Initial Timestamp",
                value=defaults.get("max_initial_timestamp", cls.__fields__["max_initial_timestamp"].default),
                info="Maximum initial timestamp"
            ),
            gr.Checkbox(
                label="Word Timestamps",
                value=defaults.get("word_timestamps", cls.__fields__["word_timestamps"].default),
                info="Extract word-level timestamps"
            ),
            gr.Textbox(
                label="Prepend Punctuations",
                value=defaults.get("prepend_punctuations", cls.__fields__["prepend_punctuations"].default),
                info="Punctuations to merge with next word"
            ),
            gr.Textbox(
                label="Append Punctuations",
                value=defaults.get("append_punctuations", cls.__fields__["append_punctuations"].default),
                info="Punctuations to merge with previous word"
            ),
            gr.Number(
                label="Max New Tokens",
                value=defaults.get("max_new_tokens", GRADIO_NONE_NUMBER_MIN),
                precision=0,
                info="Maximum number of new tokens per chunk"
            ),
            gr.Number(
                label="Chunk Length (s)",
                value=defaults.get("chunk_length", cls.__fields__["chunk_length"].default),
                precision=0,
                info="Length of audio segments in seconds"
            ),
            gr.Number(
                label="Hallucination Silence Threshold (sec)",
                value=defaults.get("hallucination_silence_threshold",
                                   GRADIO_NONE_NUMBER_MIN),
                info="Threshold for skipping silent periods in hallucination detection"
            ),
            gr.Textbox(
                label="Hotwords",
                value=defaults.get("hotwords", cls.__fields__["hotwords"].default),
                info="Hotwords/hint phrases for the model"
            ),
            gr.Number(
                label="Language Detection Threshold",
                value=defaults.get("language_detection_threshold",
                                   GRADIO_NONE_NUMBER_MIN),
                info="Threshold for language detection probability"
            ),
            gr.Number(
                label="Language Detection Segments",
                value=defaults.get("language_detection_segments",
                                   cls.__fields__["language_detection_segments"].default),
                precision=0,
                info="Number of segments for language detection"
            )
        ]

        insanely_fast_whisper_inputs = [
            gr.Number(
                label="Batch Size",
                value=defaults.get("batch_size", cls.__fields__["batch_size"].default),
                precision=0,
                info="Batch size for processing"
            )
        ]

        if whisper_type != WhisperImpl.FASTER_WHISPER.value:
            for input_component in faster_whisper_inputs:
                input_component.visible = False

        if whisper_type != WhisperImpl.INSANELY_FAST_WHISPER.value:
            for input_component in insanely_fast_whisper_inputs:
                input_component.visible = False

        inputs += faster_whisper_inputs + insanely_fast_whisper_inputs

        return inputs


class TranscriptionPipelineParams(BaseModel):
    """Transcription pipeline parameters"""
    whisper: WhisperParams = Field(default_factory=WhisperParams)
    vad: VadParams = Field(default_factory=VadParams)
    diarization: DiarizationParams = Field(default_factory=DiarizationParams)
    bgm_separation: BGMSeparationParams = Field(default_factory=BGMSeparationParams)

    def to_dict(self) -> Dict:
        data = {
            "whisper": self.whisper.to_dict(),
            "vad": self.vad.to_dict(),
            "diarization": self.diarization.to_dict(),
            "bgm_separation": self.bgm_separation.to_dict()
        }
        return data

    def to_list(self) -> List:
        """
        Convert data class to the list because I have to pass the parameters as a list in the gradio.
        Related Gradio issue: https://github.com/gradio-app/gradio/issues/2471
        See more about Gradio pre-processing: https://www.gradio.app/docs/components
        """
        whisper_list = self.whisper.to_list()
        vad_list = self.vad.to_list()
        diarization_list = self.diarization.to_list()
        bgm_sep_list = self.bgm_separation.to_list()
        return whisper_list + vad_list + diarization_list + bgm_sep_list

    @staticmethod
    def from_list(pipeline_list: List) -> 'TranscriptionPipelineParams':
        """Convert list to the data class again to use it in a function."""
        data_list = deepcopy(pipeline_list)

        whisper_list = data_list[0:len(WhisperParams.__annotations__)]
        data_list = data_list[len(WhisperParams.__annotations__):]

        vad_list = data_list[0:len(VadParams.__annotations__)]
        data_list = data_list[len(VadParams.__annotations__):]

        diarization_list = data_list[0:len(DiarizationParams.__annotations__)]
        data_list = data_list[len(DiarizationParams.__annotations__):]

        bgm_sep_list = data_list[0:len(BGMSeparationParams.__annotations__)]

        return TranscriptionPipelineParams(
            whisper=WhisperParams.from_list(whisper_list),
            vad=VadParams.from_list(vad_list),
            diarization=DiarizationParams.from_list(diarization_list),
            bgm_separation=BGMSeparationParams.from_list(bgm_sep_list)
        )
