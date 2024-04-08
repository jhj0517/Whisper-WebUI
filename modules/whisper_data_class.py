from dataclasses import dataclass, fields
import gradio as gr


@dataclass
class WhisperGradioComponents:
    model_size: gr.Dropdown
    lang: gr.Dropdown
    is_translate: gr.Checkbox
    beam_size: gr.Number
    log_prob_threshold: gr.Number
    no_speech_threshold: gr.Number
    compute_type: gr.Dropdown
    """
    A data class to pass Gradio components to the function before Gradio pre-processing.
    See this documentation for more info: https://www.gradio.app/docs/components

    Parameters
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
    """

    def to_list(self) -> list:
        """
        Use this to pass parameters before Gradio pre-processing to the function.

        Returns
        ----------
        A list of Gradio components
        """
        return [getattr(self, f.name) for f in fields(self)]

    @staticmethod
    def to_values(*params):
        """
        Use this to use parameters after Gradio pre-processing in the function.

        Returns
        ----------
        A WhisperValues data class
        """
        return WhisperValues(*params)


@dataclass
class WhisperValues:
    model_size: str
    lang: str
    is_translate: bool
    beam_size: int
    log_prob_threshold: float
    no_speech_threshold: float
    compute_type: str
    """
    A data class to use Whisper parameters in the function after gradio pre-processing.
    See this documentation for more info : https://www.gradio.app/docs/components
    """
