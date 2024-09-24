from dataclasses import dataclass, fields
import gradio as gr
from typing import Optional, Dict
import yaml


@dataclass
class WhisperParameters:
    model_size: gr.Dropdown
    lang: gr.Dropdown
    is_translate: gr.Checkbox
    beam_size: gr.Number
    log_prob_threshold: gr.Number
    no_speech_threshold: gr.Number
    compute_type: gr.Dropdown
    best_of: gr.Number
    patience: gr.Number
    condition_on_previous_text: gr.Checkbox
    prompt_reset_on_temperature: gr.Slider
    initial_prompt: gr.Textbox
    temperature: gr.Slider
    compression_ratio_threshold: gr.Number
    vad_filter: gr.Checkbox
    threshold: gr.Slider
    min_speech_duration_ms: gr.Number
    max_speech_duration_s: gr.Number
    min_silence_duration_ms: gr.Number
    speech_pad_ms: gr.Number
    batch_size: gr.Number
    is_diarize: gr.Checkbox
    hf_token: gr.Textbox
    diarization_device: gr.Dropdown
    length_penalty: gr.Number
    repetition_penalty: gr.Number
    no_repeat_ngram_size: gr.Number
    prefix: gr.Textbox
    suppress_blank: gr.Checkbox
    suppress_tokens: gr.Textbox
    max_initial_timestamp: gr.Number
    word_timestamps: gr.Checkbox
    prepend_punctuations: gr.Textbox
    append_punctuations: gr.Textbox
    max_new_tokens: gr.Number
    chunk_length: gr.Number
    hallucination_silence_threshold: gr.Number
    hotwords: gr.Textbox
    language_detection_threshold: gr.Number
    language_detection_segments: gr.Number
    is_bgm_separate: gr.Checkbox
    uvr_model_size: gr.Dropdown
    uvr_device: gr.Dropdown
    uvr_segment_size: gr.Number
    uvr_save_file: gr.Checkbox
    uvr_enable_offload: gr.Checkbox
    """
    A data class for Gradio components of the Whisper Parameters. Use "before" Gradio pre-processing.
    This data class is used to mitigate the key-value problem between Gradio components and function parameters.
    Related Gradio issue: https://github.com/gradio-app/gradio/issues/2471
    See more about Gradio pre-processing: https://www.gradio.app/docs/components

    Attributes
    ----------
    model_size: gr.Dropdown
        Whisper model size.
        
    lang: gr.Dropdown
        Source language of the file to transcribe.
        
    is_translate: gr.Checkbox
        Boolean value that determines whether to translate to English.
        It's Whisper's feature to translate speech from another language directly into English end-to-end.
        
    beam_size: gr.Number
        Int value that is used for decoding option.
        
    log_prob_threshold: gr.Number
        If the average log probability over sampled tokens is below this value, treat as failed.
        
    no_speech_threshold: gr.Number
        If the no_speech probability is higher than this value AND 
        the average log probability over sampled tokens is below `log_prob_threshold`,
        consider the segment as silent.
        
    compute_type: gr.Dropdown
        compute type for transcription.
        see more info : https://opennmt.net/CTranslate2/quantization.html
        
    best_of: gr.Number
        Number of candidates when sampling with non-zero temperature.
        
    patience: gr.Number
        Beam search patience factor.
        
    condition_on_previous_text: gr.Checkbox
        if True, the previous output of the model is provided as a prompt for the next window;
        disabling may make the text inconsistent across windows, but the model becomes less prone to
        getting stuck in a failure loop, such as repetition looping or timestamps going out of sync.
        
    initial_prompt: gr.Textbox
        Optional text to provide as a prompt for the first window. This can be used to provide, or
        "prompt-engineer" a context for transcription, e.g. custom vocabularies or proper nouns
        to make it more likely to predict those word correctly.
        
    temperature: gr.Slider 
        Temperature for sampling. It can be a tuple of temperatures,
        which will be successively used upon failures according to either
        `compression_ratio_threshold` or `log_prob_threshold`.
            
    compression_ratio_threshold: gr.Number
        If the gzip compression ratio is above this value, treat as failed
        
    vad_filter: gr.Checkbox
        Enable the voice activity detection (VAD) to filter out parts of the audio
        without speech. This step is using the Silero VAD model
        https://github.com/snakers4/silero-vad.
        
    threshold: gr.Slider
        This parameter is related with Silero VAD. Speech threshold. 
        Silero VAD outputs speech probabilities for each audio chunk,
        probabilities ABOVE this value are considered as SPEECH. It is better to tune this
        parameter for each dataset separately, but "lazy" 0.5 is pretty good for most datasets.
        
    min_speech_duration_ms: gr.Number
        This parameter is related with Silero VAD. Final speech chunks shorter min_speech_duration_ms are thrown out.
        
    max_speech_duration_s: gr.Number
        This parameter is related with Silero VAD. Maximum duration of speech chunks in seconds. Chunks longer
        than max_speech_duration_s will be split at the timestamp of the last silence that
        lasts more than 100ms (if any), to prevent aggressive cutting. Otherwise, they will be
        split aggressively just before max_speech_duration_s.
    
    min_silence_duration_ms: gr.Number
        This parameter is related with Silero VAD. In the end of each speech chunk wait for min_silence_duration_ms
        before separating it
        
    speech_pad_ms: gr.Number
        This parameter is related with Silero VAD. Final speech chunks are padded by speech_pad_ms each side    
        
    batch_size: gr.Number
        This parameter is related with insanely-fast-whisper pipe. Batch size to pass to the pipe
        
    is_diarize: gr.Checkbox
        This parameter is related with whisperx. Boolean value that determines whether to diarize or not.
        
    hf_token: gr.Textbox
        This parameter is related with whisperx. Huggingface token is needed to download diarization models.
        Read more about : https://huggingface.co/pyannote/speaker-diarization-3.1#requirements
        
    diarization_device: gr.Dropdown
        This parameter is related with whisperx. Device to run diarization model
        
    length_penalty: gr.Number
        This parameter is related to faster-whisper. Exponential length penalty constant.
    
    repetition_penalty: gr.Number
        This parameter is related to faster-whisper. Penalty applied to the score of previously generated tokens
        (set > 1 to penalize).

    no_repeat_ngram_size: gr.Number
        This parameter is related to faster-whisper. Prevent repetitions of n-grams with this size (set 0 to disable).

    prefix: gr.Textbox
        This parameter is related to faster-whisper. Optional text to provide as a prefix for the first window.

    suppress_blank: gr.Checkbox
        This parameter is related to faster-whisper. Suppress blank outputs at the beginning of the sampling.

    suppress_tokens: gr.Textbox
        This parameter is related to faster-whisper. List of token IDs to suppress. -1 will suppress a default set
        of symbols as defined in the model config.json file.

    max_initial_timestamp: gr.Number
        This parameter is related to faster-whisper. The initial timestamp cannot be later than this.

    word_timestamps: gr.Checkbox
        This parameter is related to faster-whisper. Extract word-level timestamps using the cross-attention pattern
        and dynamic time warping, and include the timestamps for each word in each segment.

    prepend_punctuations: gr.Textbox
        This parameter is related to faster-whisper. If word_timestamps is True, merge these punctuation symbols
        with the next word.

    append_punctuations: gr.Textbox
        This parameter is related to faster-whisper. If word_timestamps is True, merge these punctuation symbols
        with the previous word.

    max_new_tokens: gr.Number
        This parameter is related to faster-whisper. Maximum number of new tokens to generate per-chunk. If not set,
        the maximum will be set by the default max_length.

    chunk_length: gr.Number
        This parameter is related to faster-whisper and insanely-fast-whisper. The length of audio segments in seconds.
         If it is not None, it will overwrite the default chunk_length of the FeatureExtractor.

    hallucination_silence_threshold: gr.Number
        This parameter is related to faster-whisper. When word_timestamps is True, skip silent periods longer than this threshold
        (in seconds) when a possible hallucination is detected.

    hotwords: gr.Textbox
        This parameter is related to faster-whisper. Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.

    language_detection_threshold: gr.Number
        This parameter is related to faster-whisper. If the maximum probability of the language tokens is higher than this value, the language is detected.

    language_detection_segments: gr.Number
        This parameter is related to faster-whisper. Number of segments to consider for the language detection.
        
    is_separate_bgm: gr.Checkbox
        This parameter is related to UVR. Boolean value that determines whether to separate bgm or not.
        
    uvr_model_size: gr.Dropdown
        This parameter is related to UVR. UVR model size.
    
    uvr_device: gr.Dropdown
        This parameter is related to UVR. Device to run UVR model.
        
    uvr_segment_size: gr.Number
        This parameter is related to UVR. Segment size for UVR model.
        
    uvr_save_file: gr.Checkbox
        This parameter is related to UVR. Boolean value that determines whether to save the file or not.
        
    uvr_enable_offload: gr.Checkbox
        This parameter is related to UVR. Boolean value that determines whether to offload the UVR model or not
        after each transcription.
    """

    def as_list(self) -> list:
        """
        Converts the data class attributes into a list, Use in Gradio UI before Gradio pre-processing.
        See more about Gradio pre-processing: : https://www.gradio.app/docs/components

        Returns
        ----------
        A list of Gradio components
        """
        return [getattr(self, f.name) for f in fields(self)]

    @staticmethod
    def as_value(*args) -> 'WhisperValues':
        """
        To use Whisper parameters in function after Gradio post-processing.
        See more about Gradio post-processing: : https://www.gradio.app/docs/components

        Returns
        ----------
        WhisperValues
           Data class that has values of parameters
        """
        return WhisperValues(*args)


@dataclass
class WhisperValues:
    model_size: str = "large-v2"
    lang: Optional[str] = None
    is_translate: bool = False
    beam_size: int = 5
    log_prob_threshold: float = -1.0
    no_speech_threshold: float = 0.6
    compute_type: str = "float16"
    best_of: int = 5
    patience: float = 1.0
    condition_on_previous_text: bool = True
    prompt_reset_on_temperature: float = 0.5
    initial_prompt: Optional[str] = None
    temperature: float = 0.0
    compression_ratio_threshold: float = 2.4
    vad_filter: bool = False
    threshold: float = 0.5
    min_speech_duration_ms: int = 250
    max_speech_duration_s: float = float("inf")
    min_silence_duration_ms: int = 2000
    speech_pad_ms: int = 400
    batch_size: int = 24
    is_diarize: bool = False
    hf_token: str = ""
    diarization_device: str = "cuda"
    length_penalty: float = 1.0
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    prefix: Optional[str] = None
    suppress_blank: bool = True
    suppress_tokens: Optional[str] = "[-1]"
    max_initial_timestamp: float = 0.0
    word_timestamps: bool = False
    prepend_punctuations: Optional[str] = "\"'“¿([{-"
    append_punctuations: Optional[str] = "\"'.。,，!！?？:：”)]}、"
    max_new_tokens: Optional[int] = None
    chunk_length: Optional[int] = 30
    hallucination_silence_threshold: Optional[float] = None
    hotwords: Optional[str] = None
    language_detection_threshold: Optional[float] = None
    language_detection_segments: int = 1
    is_bgm_separate: bool = False
    uvr_model_size: str = "UVR-MDX-NET-Inst_HQ_4"
    uvr_device: str = "cuda"
    uvr_segment_size: int = 256
    uvr_save_file: bool = False
    uvr_enable_offload: bool = True
    """
    A data class to use Whisper parameters.
    """

    def to_yaml(self) -> Dict:
        data = {
            "whisper": {
                "model_size": self.model_size,
                "lang": "Automatic Detection" if self.lang is None else self.lang,
                "is_translate": self.is_translate,
                "beam_size": self.beam_size,
                "log_prob_threshold": self.log_prob_threshold,
                "no_speech_threshold": self.no_speech_threshold,
                "best_of": self.best_of,
                "patience": self.patience,
                "condition_on_previous_text": self.condition_on_previous_text,
                "prompt_reset_on_temperature": self.prompt_reset_on_temperature,
                "initial_prompt": None if not self.initial_prompt else self.initial_prompt,
                "temperature": self.temperature,
                "compression_ratio_threshold": self.compression_ratio_threshold,
                "batch_size": self.batch_size,
                "length_penalty": self.length_penalty,
                "repetition_penalty": self.repetition_penalty,
                "no_repeat_ngram_size": self.no_repeat_ngram_size,
                "prefix": None if not self.prefix else self.prefix,
                "suppress_blank": self.suppress_blank,
                "suppress_tokens": self.suppress_tokens,
                "max_initial_timestamp": self.max_initial_timestamp,
                "word_timestamps": self.word_timestamps,
                "prepend_punctuations": self.prepend_punctuations,
                "append_punctuations": self.append_punctuations,
                "max_new_tokens": self.max_new_tokens,
                "chunk_length": self.chunk_length,
                "hallucination_silence_threshold": self.hallucination_silence_threshold,
                "hotwords": None if not self.hotwords else self.hotwords,
                "language_detection_threshold": self.language_detection_threshold,
                "language_detection_segments": self.language_detection_segments,
            },
            "vad": {
                "vad_filter": self.vad_filter,
                "threshold": self.threshold,
                "min_speech_duration_ms": self.min_speech_duration_ms,
                "max_speech_duration_s": self.max_speech_duration_s,
                "min_silence_duration_ms": self.min_silence_duration_ms,
                "speech_pad_ms": self.speech_pad_ms,
            },
            "diarization": {
                "is_diarize": self.is_diarize,
                "hf_token": self.hf_token
            },
            "bgm_separation": {
                "is_separate_bgm": self.is_bgm_separate,
                "model_size": self.uvr_model_size,
                "segment_size": self.uvr_segment_size,
                "save_file": self.uvr_save_file,
                "enable_offload": self.uvr_enable_offload
            },
        }
        return data
