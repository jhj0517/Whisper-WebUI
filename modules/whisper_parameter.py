from dataclasses import dataclass, fields
import gradio as gr
from typing import Optional


@dataclass
class WhisperGradioComponents:
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
    """
    A data class for Gradio components of the Whisper Parameters. Use "before" Gradio pre-processing.
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
    """

    def to_list(self) -> list:
        """
        Converts the data class attributes into a list. Use "before" Gradio pre-processing.
        See more about Gradio pre-processing: : https://www.gradio.app/docs/components

        Returns
        ----------
        A list of Gradio components
        """
        return [getattr(self, f.name) for f in fields(self)]


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
    """
    A data class to use Whisper parameters. Use "after" Gradio pre-processing.
    See more about Gradio pre-processing: : https://www.gradio.app/docs/components
    """
