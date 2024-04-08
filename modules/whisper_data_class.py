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

    def to_list(self):
        return [getattr(self, f.name) for f in fields(self)]

    @staticmethod
    def to_values(*params):
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
