from dataclasses import dataclass, fields
import gradio as gr
from typing import Optional


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
    initial_prompt: gr.Textbox
    temperature: gr.Slider
    compression_ratio_threshold: gr.Number
    vad_filter: gr.Checkbox
    threshold: gr.Slider
    min_speech_duration_ms: gr.Number
    max_speech_duration_s: gr.Number
    min_silence_duration_ms: gr.Number
    window_size_sample: gr.Number
    speech_pad_ms: gr.Number
    chunk_length_s: gr.Number
    batch_size: gr.Number
    is_diarize: gr.Checkbox
    hf_token: gr.Textbox
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
        
    window_size_samples: gr.Number
        This parameter is related with Silero VAD. Audio chunks of window_size_samples size are fed to the silero VAD model.
        WARNING! Silero VAD models were trained using 512, 1024, 1536 samples for 16000 sample rate.
        Values other than these may affect model performance!!
        
    speech_pad_ms: gr.Number
        This parameter is related with Silero VAD. Final speech chunks are padded by speech_pad_ms each side    
        
    chunk_length_s: gr.Number
        This parameter is related with insanely-fast-whisper pipe.
        Maximum length of each chunk
        
    batch_size: gr.Number
        This parameter is related with insanely-fast-whisper pipe. Batch size to pass to the pipe
        
    is_diarize: gr.Checkbox
        This parameter is related with whisperx. Boolean value that determines whether to diarize or not.
        
    hf_token: gr.Textbox
        This parameter is related with whisperx. Huggingface token is needed to download diarization models.
        Read more about : https://huggingface.co/pyannote/speaker-diarization-3.1#requirements
    """

    def to_list(self) -> list:
        """
        Converts the data class attributes into a list, Use in Gradio UI before Gradio pre-processing.
        See more about Gradio pre-processing: : https://www.gradio.app/docs/components

        Returns
        ----------
        A list of Gradio components
        """
        return [getattr(self, f.name) for f in fields(self)]

    @staticmethod
    def post_process(*args) -> 'WhisperValues':
        """
        To use Whisper parameters in function after Gradio post-processing.
        See more about Gradio post-processing: : https://www.gradio.app/docs/components

        Returns
        ----------
        WhisperValues
           Data class that has values of parameters
        """
        return WhisperValues(
            model_size=args[0],
            lang=args[1],
            is_translate=args[2],
            beam_size=args[3],
            log_prob_threshold=args[4],
            no_speech_threshold=args[5],
            compute_type=args[6],
            best_of=args[7],
            patience=args[8],
            condition_on_previous_text=args[9],
            initial_prompt=args[10],
            temperature=args[11],
            compression_ratio_threshold=args[12],
            vad_filter=args[13],
            threshold=args[14],
            min_speech_duration_ms=args[15],
            max_speech_duration_s=args[16],
            min_silence_duration_ms=args[17],
            window_size_samples=args[18],
            speech_pad_ms=args[19],
            chunk_length_s=args[20],
            batch_size=args[21],
            is_diarize=args[22],
            hf_token=args[23],
        )


@dataclass
class WhisperValues:
    model_size: str
    lang: str
    is_translate: bool
    beam_size: int
    log_prob_threshold: float
    no_speech_threshold: float
    compute_type: str
    best_of: int
    patience: float
    condition_on_previous_text: bool
    initial_prompt: Optional[str]
    temperature: float
    compression_ratio_threshold: float
    vad_filter: bool
    threshold: float
    min_speech_duration_ms: int
    max_speech_duration_s: float
    min_silence_duration_ms: int
    window_size_samples: int
    speech_pad_ms: int
    chunk_length_s: int
    batch_size: int
    is_diarize: bool
    hf_token: str
    """
    A data class to use Whisper parameters.
    """
