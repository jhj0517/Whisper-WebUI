import os
import argparse
import gradio as gr

from modules.whisper.whisper_Inference import WhisperInference
from modules.whisper.faster_whisper_inference import FasterWhisperInference
from modules.whisper.insanely_fast_whisper_inference import InsanelyFastWhisperInference
from modules.translation.nllb_inference import NLLBInference
from ui.htmls import *
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.whisper_parameter import *


class App:
    def __init__(self, args):
        self.args = args
        self.app = gr.Blocks(css=CSS, theme=self.args.theme)
        self.whisper_inf = self.init_whisper()
        print(f"Use \"{self.args.whisper_type}\" implementation")
        print(f"Device \"{self.whisper_inf.device}\" is detected")
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=self.args.output_dir
        )

    def init_whisper(self):
        # Temporal fix of the issue : https://github.com/jhj0517/Whisper-WebUI/issues/144
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

        whisper_type = self.args.whisper_type.lower().strip()

        if whisper_type in ["faster_whisper", "faster-whisper", "fasterwhisper"]:
            whisper_inf = FasterWhisperInference(
                model_dir=self.args.faster_whisper_model_dir,
                output_dir=self.args.output_dir,
                args=self.args
            )
        elif whisper_type in ["whisper"]:
            whisper_inf = WhisperInference(
                model_dir=self.args.whisper_model_dir,
                output_dir=self.args.output_dir,
                args=self.args
            )
        elif whisper_type in ["insanely_fast_whisper", "insanely-fast-whisper", "insanelyfastwhisper",
                              "insanely_faster_whisper", "insanely-faster-whisper", "insanelyfasterwhisper"]:
            whisper_inf = InsanelyFastWhisperInference(
                model_dir=self.args.insanely_fast_whisper_model_dir,
                output_dir=self.args.output_dir,
                args=self.args
            )
        else:
            whisper_inf = FasterWhisperInference(
                model_dir=self.args.faster_whisper_model_dir,
                output_dir=self.args.output_dir,
                args=self.args
            )
        return whisper_inf

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            print(f"The folder {folder_path} does not exist.")

    @staticmethod
    def on_change_models(model_size: str):
        translatable_model = ["large", "large-v1", "large-v2", "large-v3"]
        if model_size not in translatable_model:
            return gr.Checkbox(visible=False, value=False, interactive=False)
        else:
            return gr.Checkbox(visible=True, value=False, label="Translate to English?", interactive=True)

    def launch(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    gr.Markdown(MARKDOWN, elem_id="md_project")
            with gr.Tabs():
                with gr.TabItem("File"):  # tab1
                    with gr.Column():
                        input_file = gr.Files(type="filepath", label="Upload File here")
                        tb_input_folder = gr.Textbox(label="Input Folder Path (Optional)",
                                                     info="Optional: Specify the folder path where the input files are located, if you prefer to use local files instead of uploading them."
                                                          " Leave this field empty if you do not wish to use a local path.",
                                                     visible=self.args.colab,
                                                     value="")
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v2",
                                               label="Model")
                        dd_lang = gr.Dropdown(choices=["Automatic Detection"] + self.whisper_inf.available_langs,
                                              value="Automatic Detection", label="Language")
                        dd_file_format = gr.Dropdown(["SRT", "WebVTT", "txt"], value="SRT", label="File Format")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
                    with gr.Row():
                        cb_timestamp = gr.Checkbox(value=True, label="Add a timestamp to the end of the filename",
                                                   interactive=True)
                    with gr.Accordion("Advanced Parameters", open=False):
                        nb_beam_size = gr.Number(label="Beam Size", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="Log Probability Threshold", value=-1.0,
                                                          interactive=True)
                        nb_no_speech_threshold = gr.Number(label="No Speech Threshold", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="Compute Type",
                                                      choices=self.whisper_inf.available_compute_types,
                                                      value=self.whisper_inf.current_compute_type, interactive=True)
                        nb_best_of = gr.Number(label="Best Of", value=5, interactive=True)
                        nb_patience = gr.Number(label="Patience", value=1, interactive=True)
                        cb_condition_on_previous_text = gr.Checkbox(label="Condition On Previous Text", value=True,
                                                                    interactive=True)
                        tb_initial_prompt = gr.Textbox(label="Initial Prompt", value=None, interactive=True)
                        sd_temperature = gr.Slider(label="Temperature", value=0, step=0.01, maximum=1.0,
                                                   interactive=True)
                        nb_compression_ratio_threshold = gr.Number(label="Compression Ratio Threshold", value=2.4,
                                                                   interactive=True)
                        with gr.Group(visible=isinstance(self.whisper_inf, FasterWhisperInference)):
                            with gr.Column():
                                nb_length_penalty = gr.Number(label="Length Penalty", value=1,
                                                              info="Exponential length penalty constant.")
                                nb_repetition_penalty = gr.Number(label="Repetition Penalty", value=1,
                                                                  info="Penalty applied to the score of previously generated tokens (set > 1 to penalize).")
                                nb_no_repeat_ngram_size = gr.Number(label="No Repeat N-gram Size", value=0, precision=0,
                                                                    info="Prevent repetitions of n-grams with this size (set 0 to disable).")
                                tb_prefix = gr.Textbox(label="Prefix", value=lambda: None,  # Bug Fix https://github.com/gradio-app/gradio/issues/6728
                                                       info="Optional text to provide as a prefix for the first window.")
                                cb_suppress_blank = gr.Checkbox(label="Suppress Blank", value=True,
                                                                info="Suppress blank outputs at the beginning of the sampling.")
                                tb_suppress_tokens = gr.Textbox(label="Suppress Tokens", value="[-1]",
                                                                info="List of token IDs to suppress. -1 will suppress a default set of symbols as defined in the model config.json file.")
                                nb_max_initial_timestamp = gr.Number(label="Max Initial Timestamp", value=1.0,
                                                                     info="The initial timestamp cannot be later than this.")
                                cb_word_timestamps = gr.Checkbox(label="Word Timestamps", value=False,
                                                                 info="Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.")
                                tb_prepend_punctuations = gr.Textbox(label="Prepend Punctuations", value="\"'“¿([{-",
                                                                     info="If 'Word Timestamps' is True, merge these punctuation symbols with the next word.")
                                tb_append_punctuations = gr.Textbox(label="Append Punctuations",
                                                                    value="\"'.。,，!！?？:：”)]}、",
                                                                    info="If 'Word Timestamps' is True, merge these punctuation symbols with the previous word.")
                                nb_max_new_tokens = gr.Number(label="Max New Tokens", value=lambda: None, precision=0,
                                                              info="Maximum number of new tokens to generate per-chunk. If not set, the maximum will be set by the default max_length.")
                                nb_chunk_length = gr.Number(label="Chunk Length", value=lambda: None, precision=0,
                                                            info="The length of audio segments. If it is not None, it will overwrite the default chunk_length of the FeatureExtractor.")
                                nb_hallucination_silence_threshold = gr.Number(label="Hallucination Silence Threshold (sec)",
                                                                               value=lambda: None,
                                                                               info="When 'Word Timestamps' is True, skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected.")
                                tb_hotwords = gr.Textbox(label="Hotwords", value=None,
                                                         info="Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.")
                                nb_language_detection_threshold = gr.Number(label="Language Detection Threshold",
                                                                            value=None,
                                                                            info="If the maximum probability of the language tokens is higher than this value, the language is detected.")
                                nb_language_detection_segments = gr.Number(label="Language Detection Segments", value=1,
                                                                           precision=0,
                                                                           info="Number of segments to consider for the language detection.")

                        with gr.Group(visible=isinstance(self.whisper_inf, InsanelyFastWhisperInference)):
                            nb_chunk_length_s = gr.Number(label="Chunk Lengths (sec)", value=30, precision=0)
                            nb_batch_size = gr.Number(label="Batch Size", value=24, precision=0)
                    with gr.Accordion("VAD", open=False):
                        cb_vad_filter = gr.Checkbox(label="Enable Silero VAD Filter", value=False, interactive=True)
                        sd_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold",
                                                 value=0.5, info="Lower it to be more sensitive to small sounds.")
                        nb_min_speech_duration_ms = gr.Number(label="Minimum Speech Duration (ms)", precision=0,
                                                              value=250)
                        nb_max_speech_duration_s = gr.Number(label="Maximum Speech Duration (s)", value=9999)
                        nb_min_silence_duration_ms = gr.Number(label="Minimum Silence Duration (ms)", precision=0,
                                                               value=2000)
                        nb_speech_pad_ms = gr.Number(label="Speech Padding (ms)", precision=0, value=400)
                    with gr.Accordion("Diarization", open=False):
                        cb_diarize = gr.Checkbox(label="Enable Diarization")
                        tb_hf_token = gr.Text(label="HuggingFace Token", value="",
                                              info="This is only needed the first time you download the model. If you already have models, you don't need to enter. "
                                                   "To download the model, you must manually go to \"https://huggingface.co/pyannote/speaker-diarization-3.1\" and agree to their requirement.")
                        dd_diarization_device = gr.Dropdown(label="Device",
                                                            choices=self.whisper_inf.diarizer.get_available_device(),
                                                            value=self.whisper_inf.diarizer.get_device())
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Files(label="Downloadable output file", scale=3, interactive=False)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [input_file, tb_input_folder, dd_file_format, cb_timestamp]
                    whisper_params = WhisperParameters(
                        model_size=dd_model,
                        lang=dd_lang,
                        is_translate=cb_translate,
                        beam_size=nb_beam_size,
                        log_prob_threshold=nb_log_prob_threshold,
                        no_speech_threshold=nb_no_speech_threshold,
                        compute_type=dd_compute_type,
                        best_of=nb_best_of,
                        patience=nb_patience,
                        condition_on_previous_text=cb_condition_on_previous_text,
                        initial_prompt=tb_initial_prompt,
                        temperature=sd_temperature,
                        compression_ratio_threshold=nb_compression_ratio_threshold,
                        vad_filter=cb_vad_filter,
                        threshold=sd_threshold,
                        min_speech_duration_ms=nb_min_speech_duration_ms,
                        max_speech_duration_s=nb_max_speech_duration_s,
                        min_silence_duration_ms=nb_min_silence_duration_ms,
                        speech_pad_ms=nb_speech_pad_ms,
                        chunk_length_s=nb_chunk_length_s,
                        batch_size=nb_batch_size,
                        is_diarize=cb_diarize,
                        hf_token=tb_hf_token,
                        diarization_device=dd_diarization_device,
                        length_penalty=nb_length_penalty,
                        repetition_penalty=nb_repetition_penalty,
                        no_repeat_ngram_size=nb_no_repeat_ngram_size,
                        prefix=tb_prefix,
                        suppress_blank=cb_suppress_blank,
                        suppress_tokens=tb_suppress_tokens,
                        max_initial_timestamp=nb_max_initial_timestamp,
                        word_timestamps=cb_word_timestamps,
                        prepend_punctuations=tb_prepend_punctuations,
                        append_punctuations=tb_append_punctuations,
                        max_new_tokens=nb_max_new_tokens,
                        chunk_length=nb_chunk_length,
                        hallucination_silence_threshold=nb_hallucination_silence_threshold,
                        hotwords=tb_hotwords,
                        language_detection_threshold=nb_language_detection_threshold,
                        language_detection_segments=nb_language_detection_segments
                    )

                    btn_run.click(fn=self.whisper_inf.transcribe_file,
                                  inputs=params + whisper_params.as_list(),
                                  outputs=[tb_indicator, files_subtitles])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
                    dd_model.change(fn=self.on_change_models, inputs=[dd_model], outputs=[cb_translate])

                with gr.TabItem("Youtube"):  # tab2
                    with gr.Row():
                        tb_youtubelink = gr.Textbox(label="Youtube Link")
                    with gr.Row(equal_height=True):
                        with gr.Column():
                            img_thumbnail = gr.Image(label="Youtube Thumbnail")
                        with gr.Column():
                            tb_title = gr.Label(label="Youtube Title")
                            tb_description = gr.Textbox(label="Youtube Description", max_lines=15)
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v2",
                                               label="Model")
                        dd_lang = gr.Dropdown(choices=["Automatic Detection"] + self.whisper_inf.available_langs,
                                              value="Automatic Detection", label="Language")
                        dd_file_format = gr.Dropdown(choices=["SRT", "WebVTT", "txt"], value="SRT", label="File Format")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
                    with gr.Row():
                        cb_timestamp = gr.Checkbox(value=True, label="Add a timestamp to the end of the filename",
                                                   interactive=True)
                    with gr.Accordion("Advanced Parameters", open=False):
                        nb_beam_size = gr.Number(label="Beam Size", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="Log Probability Threshold", value=-1.0,
                                                          interactive=True)
                        nb_no_speech_threshold = gr.Number(label="No Speech Threshold", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="Compute Type",
                                                      choices=self.whisper_inf.available_compute_types,
                                                      value=self.whisper_inf.current_compute_type, interactive=True)
                        nb_best_of = gr.Number(label="Best Of", value=5, interactive=True)
                        nb_patience = gr.Number(label="Patience", value=1, interactive=True)
                        cb_condition_on_previous_text = gr.Checkbox(label="Condition On Previous Text", value=True,
                                                                    interactive=True)
                        tb_initial_prompt = gr.Textbox(label="Initial Prompt", value=None, interactive=True)
                        sd_temperature = gr.Slider(label="Temperature", value=0, step=0.01, maximum=1.0,
                                                   interactive=True)
                        nb_compression_ratio_threshold = gr.Number(label="Compression Ratio Threshold", value=2.4,
                                                                   interactive=True)
                        with gr.Group(visible=isinstance(self.whisper_inf, FasterWhisperInference)):
                            with gr.Column():
                                nb_length_penalty = gr.Number(label="Length Penalty", value=1,
                                                              info="Exponential length penalty constant.")
                                nb_repetition_penalty = gr.Number(label="Repetition Penalty", value=1,
                                                                  info="Penalty applied to the score of previously generated tokens (set > 1 to penalize).")
                                nb_no_repeat_ngram_size = gr.Number(label="No Repeat N-gram Size", value=0, precision=0,
                                                                    info="Prevent repetitions of n-grams with this size (set 0 to disable).")
                                tb_prefix = gr.Textbox(label="Prefix", value=lambda: None, # Bug Fix https://github.com/gradio-app/gradio/issues/6728
                                                       info="Optional text to provide as a prefix for the first window.")
                                cb_suppress_blank = gr.Checkbox(label="Suppress Blank", value=True,
                                                                info="Suppress blank outputs at the beginning of the sampling.")
                                tb_suppress_tokens = gr.Textbox(label="Suppress Tokens", value="[-1]",
                                                                info="List of token IDs to suppress. -1 will suppress a default set of symbols as defined in the model config.json file.")
                                nb_max_initial_timestamp = gr.Number(label="Max Initial Timestamp", value=1.0,
                                                                     info="The initial timestamp cannot be later than this.")
                                cb_word_timestamps = gr.Checkbox(label="Word Timestamps", value=False,
                                                                 info="Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.")
                                tb_prepend_punctuations = gr.Textbox(label="Prepend Punctuations", value="\"'“¿([{-",
                                                                     info="If 'Word Timestamps' is True, merge these punctuation symbols with the next word.")
                                tb_append_punctuations = gr.Textbox(label="Append Punctuations",
                                                                    value="\"'.。,，!！?？:：”)]}、",
                                                                    info="If 'Word Timestamps' is True, merge these punctuation symbols with the previous word.")
                                nb_max_new_tokens = gr.Number(label="Max New Tokens", value=lambda: None, precision=0,
                                                              info="Maximum number of new tokens to generate per-chunk. If not set, the maximum will be set by the default max_length.")
                                nb_chunk_length = gr.Number(label="Chunk Length", value=lambda: None, precision=0,
                                                            info="The length of audio segments. If it is not None, it will overwrite the default chunk_length of the FeatureExtractor.")
                                nb_hallucination_silence_threshold = gr.Number(
                                    label="Hallucination Silence Threshold (sec)",
                                    value=lambda: None,
                                    info="When 'Word Timestamps' is True, skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected.")
                                tb_hotwords = gr.Textbox(label="Hotwords", value=None,
                                                         info="Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.")
                                nb_language_detection_threshold = gr.Number(label="Language Detection Threshold",
                                                                            value=None,
                                                                            info="If the maximum probability of the language tokens is higher than this value, the language is detected.")
                                nb_language_detection_segments = gr.Number(label="Language Detection Segments", value=1,
                                                                           precision=0,
                                                                           info="Number of segments to consider for the language detection.")

                        with gr.Group(visible=isinstance(self.whisper_inf, InsanelyFastWhisperInference)):
                            nb_chunk_length_s = gr.Number(label="Chunk Lengths (sec)", value=30, precision=0)
                            nb_batch_size = gr.Number(label="Batch Size", value=24, precision=0)
                    with gr.Accordion("VAD", open=False):
                        cb_vad_filter = gr.Checkbox(label="Enable Silero VAD Filter", value=False, interactive=True)
                        sd_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold",
                                                 value=0.5, info="Lower it to be more sensitive to small sounds.")
                        nb_min_speech_duration_ms = gr.Number(label="Minimum Speech Duration (ms)", precision=0,
                                                              value=250)
                        nb_max_speech_duration_s = gr.Number(label="Maximum Speech Duration (s)", value=9999)
                        nb_min_silence_duration_ms = gr.Number(label="Minimum Silence Duration (ms)", precision=0,
                                                               value=2000)
                        nb_speech_pad_ms = gr.Number(label="Speech Padding (ms)", precision=0, value=400)
                    with gr.Accordion("Diarization", open=False):
                        cb_diarize = gr.Checkbox(label="Enable Diarization")
                        tb_hf_token = gr.Text(label="HuggingFace Token", value="",
                                              info="This is only needed the first time you download the model. If you already have models, you don't need to enter. "
                                                   "To download the model, you must manually go to \"https://huggingface.co/pyannote/speaker-diarization-3.1\" and agree to their requirement.")
                        dd_diarization_device = gr.Dropdown(label="Device",
                                                            choices=self.whisper_inf.diarizer.get_available_device(),
                                                            value=self.whisper_inf.diarizer.get_device())
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Files(label="Downloadable output file", scale=3)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [tb_youtubelink, dd_file_format, cb_timestamp]
                    whisper_params = WhisperParameters(
                        model_size=dd_model,
                        lang=dd_lang,
                        is_translate=cb_translate,
                        beam_size=nb_beam_size,
                        log_prob_threshold=nb_log_prob_threshold,
                        no_speech_threshold=nb_no_speech_threshold,
                        compute_type=dd_compute_type,
                        best_of=nb_best_of,
                        patience=nb_patience,
                        condition_on_previous_text=cb_condition_on_previous_text,
                        initial_prompt=tb_initial_prompt,
                        temperature=sd_temperature,
                        compression_ratio_threshold=nb_compression_ratio_threshold,
                        vad_filter=cb_vad_filter,
                        threshold=sd_threshold,
                        min_speech_duration_ms=nb_min_speech_duration_ms,
                        max_speech_duration_s=nb_max_speech_duration_s,
                        min_silence_duration_ms=nb_min_silence_duration_ms,
                        speech_pad_ms=nb_speech_pad_ms,
                        chunk_length_s=nb_chunk_length_s,
                        batch_size=nb_batch_size,
                        is_diarize=cb_diarize,
                        hf_token=tb_hf_token,
                        diarization_device=dd_diarization_device,
                        length_penalty=nb_length_penalty,
                        repetition_penalty=nb_repetition_penalty,
                        no_repeat_ngram_size=nb_no_repeat_ngram_size,
                        prefix=tb_prefix,
                        suppress_blank=cb_suppress_blank,
                        suppress_tokens=tb_suppress_tokens,
                        max_initial_timestamp=nb_max_initial_timestamp,
                        word_timestamps=cb_word_timestamps,
                        prepend_punctuations=tb_prepend_punctuations,
                        append_punctuations=tb_append_punctuations,
                        max_new_tokens=nb_max_new_tokens,
                        chunk_length=nb_chunk_length,
                        hallucination_silence_threshold=nb_hallucination_silence_threshold,
                        hotwords=tb_hotwords,
                        language_detection_threshold=nb_language_detection_threshold,
                        language_detection_segments=nb_language_detection_segments
                    )

                    btn_run.click(fn=self.whisper_inf.transcribe_youtube,
                                  inputs=params + whisper_params.as_list(),
                                  outputs=[tb_indicator, files_subtitles])
                    tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink],
                                          outputs=[img_thumbnail, tb_title, tb_description])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
                    dd_model.change(fn=self.on_change_models, inputs=[dd_model], outputs=[cb_translate])

                with gr.TabItem("Mic"):  # tab3
                    with gr.Row():
                        mic_input = gr.Microphone(label="Record with Mic", type="filepath", interactive=True)
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v2",
                                               label="Model")
                        dd_lang = gr.Dropdown(choices=["Automatic Detection"] + self.whisper_inf.available_langs,
                                              value="Automatic Detection", label="Language")
                        dd_file_format = gr.Dropdown(["SRT", "WebVTT", "txt"], value="SRT", label="File Format")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
                    with gr.Accordion("Advanced Parameters", open=False):
                        nb_beam_size = gr.Number(label="Beam Size", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="Log Probability Threshold", value=-1.0,
                                                          interactive=True)
                        nb_no_speech_threshold = gr.Number(label="No Speech Threshold", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="Compute Type",
                                                      choices=self.whisper_inf.available_compute_types,
                                                      value=self.whisper_inf.current_compute_type, interactive=True)
                        nb_best_of = gr.Number(label="Best Of", value=5, interactive=True)
                        nb_patience = gr.Number(label="Patience", value=1, interactive=True)
                        cb_condition_on_previous_text = gr.Checkbox(label="Condition On Previous Text", value=True,
                                                                    interactive=True)
                        tb_initial_prompt = gr.Textbox(label="Initial Prompt", value=None, interactive=True)
                        sd_temperature = gr.Slider(label="Temperature", value=0, step=0.01, maximum=1.0,
                                                   interactive=True)

                        with gr.Group(visible=isinstance(self.whisper_inf, FasterWhisperInference)):
                            with gr.Column():
                                nb_length_penalty = gr.Number(label="Length Penalty", value=1,
                                                              info="Exponential length penalty constant.")
                                nb_repetition_penalty = gr.Number(label="Repetition Penalty", value=1,
                                                                  info="Penalty applied to the score of previously generated tokens (set > 1 to penalize).")
                                nb_no_repeat_ngram_size = gr.Number(label="No Repeat N-gram Size", value=0, precision=0,
                                                                    info="Prevent repetitions of n-grams with this size (set 0 to disable).")
                                tb_prefix = gr.Textbox(label="Prefix", value=lambda: None,  # Bug Fix https://github.com/gradio-app/gradio/issues/6728
                                                       info="Optional text to provide as a prefix for the first window.")
                                cb_suppress_blank = gr.Checkbox(label="Suppress Blank", value=True,
                                                                info="Suppress blank outputs at the beginning of the sampling.")
                                tb_suppress_tokens = gr.Textbox(label="Suppress Tokens", value="[-1]",
                                                                info="List of token IDs to suppress. -1 will suppress a default set of symbols as defined in the model config.json file.")
                                nb_max_initial_timestamp = gr.Number(label="Max Initial Timestamp", value=1.0,
                                                                     info="The initial timestamp cannot be later than this.")
                                cb_word_timestamps = gr.Checkbox(label="Word Timestamps", value=False,
                                                                 info="Extract word-level timestamps using the cross-attention pattern and dynamic time warping, and include the timestamps for each word in each segment.")
                                tb_prepend_punctuations = gr.Textbox(label="Prepend Punctuations", value="\"'“¿([{-",
                                                                     info="If 'Word Timestamps' is True, merge these punctuation symbols with the next word.")
                                tb_append_punctuations = gr.Textbox(label="Append Punctuations",
                                                                    value="\"'.。,，!！?？:：”)]}、",
                                                                    info="If 'Word Timestamps' is True, merge these punctuation symbols with the previous word.")
                                nb_max_new_tokens = gr.Number(label="Max New Tokens", value=lambda: None, precision=0,
                                                              info="Maximum number of new tokens to generate per-chunk. If not set, the maximum will be set by the default max_length.")
                                nb_chunk_length = gr.Number(label="Chunk Length", value=lambda: None, precision=0,
                                                            info="The length of audio segments. If it is not None, it will overwrite the default chunk_length of the FeatureExtractor.")
                                nb_hallucination_silence_threshold = gr.Number(
                                    label="Hallucination Silence Threshold (sec)",
                                    value=lambda: None,
                                    info="When 'Word Timestamps' is True, skip silent periods longer than this threshold (in seconds) when a possible hallucination is detected.")
                                tb_hotwords = gr.Textbox(label="Hotwords", value=None,
                                                         info="Hotwords/hint phrases to provide the model with. Has no effect if prefix is not None.")
                                nb_language_detection_threshold = gr.Number(label="Language Detection Threshold",
                                                                            value=None,
                                                                            info="If the maximum probability of the language tokens is higher than this value, the language is detected.")
                                nb_language_detection_segments = gr.Number(label="Language Detection Segments", value=1,
                                                                           precision=0,
                                                                           info="Number of segments to consider for the language detection.")

                        with gr.Group(visible=isinstance(self.whisper_inf, InsanelyFastWhisperInference)):
                            nb_chunk_length_s = gr.Number(label="Chunk Lengths (sec)", value=30, precision=0)
                            nb_batch_size = gr.Number(label="Batch Size", value=24, precision=0)

                    with gr.Accordion("VAD", open=False):
                        cb_vad_filter = gr.Checkbox(label="Enable Silero VAD Filter", value=False, interactive=True)
                        sd_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold",
                                                 value=0.5, info="Lower it to be more sensitive to small sounds.")
                        nb_min_speech_duration_ms = gr.Number(label="Minimum Speech Duration (ms)", precision=0,
                                                              value=250)
                        nb_max_speech_duration_s = gr.Number(label="Maximum Speech Duration (s)", value=9999)
                        nb_min_silence_duration_ms = gr.Number(label="Minimum Silence Duration (ms)", precision=0,
                                                               value=2000)
                        nb_speech_pad_ms = gr.Number(label="Speech Padding (ms)", precision=0, value=400)
                    with gr.Accordion("Diarization", open=False):
                        cb_diarize = gr.Checkbox(label="Enable Diarization")
                        tb_hf_token = gr.Text(label="HuggingFace Token", value="",
                                              info="This is only needed the first time you download the model. If you already have models, you don't need to enter. "
                                                   "To download the model, you must manually go to \"https://huggingface.co/pyannote/speaker-diarization-3.1\" and agree to their requirement.")
                        dd_diarization_device = gr.Dropdown(label="Device",
                                                            choices=self.whisper_inf.diarizer.get_available_device(),
                                                            value=self.whisper_inf.diarizer.get_device())
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Files(label="Downloadable output file", scale=3)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [mic_input, dd_file_format]
                    whisper_params = WhisperParameters(
                        model_size=dd_model,
                        lang=dd_lang,
                        is_translate=cb_translate,
                        beam_size=nb_beam_size,
                        log_prob_threshold=nb_log_prob_threshold,
                        no_speech_threshold=nb_no_speech_threshold,
                        compute_type=dd_compute_type,
                        best_of=nb_best_of,
                        patience=nb_patience,
                        condition_on_previous_text=cb_condition_on_previous_text,
                        initial_prompt=tb_initial_prompt,
                        temperature=sd_temperature,
                        compression_ratio_threshold=nb_compression_ratio_threshold,
                        vad_filter=cb_vad_filter,
                        threshold=sd_threshold,
                        min_speech_duration_ms=nb_min_speech_duration_ms,
                        max_speech_duration_s=nb_max_speech_duration_s,
                        min_silence_duration_ms=nb_min_silence_duration_ms,
                        speech_pad_ms=nb_speech_pad_ms,
                        chunk_length_s=nb_chunk_length_s,
                        batch_size=nb_batch_size,
                        is_diarize=cb_diarize,
                        hf_token=tb_hf_token,
                        diarization_device=dd_diarization_device,
                        length_penalty=nb_length_penalty,
                        repetition_penalty=nb_repetition_penalty,
                        no_repeat_ngram_size=nb_no_repeat_ngram_size,
                        prefix=tb_prefix,
                        suppress_blank=cb_suppress_blank,
                        suppress_tokens=tb_suppress_tokens,
                        max_initial_timestamp=nb_max_initial_timestamp,
                        word_timestamps=cb_word_timestamps,
                        prepend_punctuations=tb_prepend_punctuations,
                        append_punctuations=tb_append_punctuations,
                        max_new_tokens=nb_max_new_tokens,
                        chunk_length=nb_chunk_length,
                        hallucination_silence_threshold=nb_hallucination_silence_threshold,
                        hotwords=tb_hotwords,
                        language_detection_threshold=nb_language_detection_threshold,
                        language_detection_segments=nb_language_detection_segments
                    )

                    btn_run.click(fn=self.whisper_inf.transcribe_mic,
                                  inputs=params + whisper_params.as_list(),
                                  outputs=[tb_indicator, files_subtitles])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
                    dd_model.change(fn=self.on_change_models, inputs=[dd_model], outputs=[cb_translate])

                with gr.TabItem("T2T Translation"):  # tab 4
                    with gr.Row():
                        file_subs = gr.Files(type="filepath", label="Upload Subtitle Files to translate here",
                                             file_types=['.vtt', '.srt'])

                    with gr.TabItem("DeepL API"):  # sub tab1
                        with gr.Row():
                            tb_authkey = gr.Textbox(label="Your Auth Key (API KEY)",
                                                    value="")
                        with gr.Row():
                            dd_deepl_sourcelang = gr.Dropdown(label="Source Language", value="Automatic Detection",
                                                              choices=list(
                                                                  self.deepl_api.available_source_langs.keys()))
                            dd_deepl_targetlang = gr.Dropdown(label="Target Language", value="English",
                                                              choices=list(
                                                                  self.deepl_api.available_target_langs.keys()))
                        with gr.Row():
                            cb_deepl_ispro = gr.Checkbox(label="Pro User?", value=False)
                        with gr.Row():
                            btn_run = gr.Button("TRANSLATE SUBTITLE FILE", variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label="Output", scale=5)
                            files_subtitles = gr.Files(label="Downloadable output file", scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)

                    btn_run.click(fn=self.deepl_api.translate_deepl,
                                  inputs=[tb_authkey, file_subs, dd_deepl_sourcelang, dd_deepl_targetlang,
                                          cb_deepl_ispro],
                                  outputs=[tb_indicator, files_subtitles])

                    btn_openfolder.click(fn=lambda: self.open_folder(os.path.join("outputs", "translations")),
                                         inputs=None,
                                         outputs=None)

                    with gr.TabItem("NLLB"):  # sub tab2
                        with gr.Row():
                            dd_nllb_model = gr.Dropdown(label="Model", value="facebook/nllb-200-1.3B",
                                                        choices=self.nllb_inf.available_models)
                            dd_nllb_sourcelang = gr.Dropdown(label="Source Language",
                                                             choices=self.nllb_inf.available_source_langs)
                            dd_nllb_targetlang = gr.Dropdown(label="Target Language",
                                                             choices=self.nllb_inf.available_target_langs)
                        with gr.Row():
                            nb_max_length = gr.Number(label="Max Length Per Line", value=200, precision=0)
                        with gr.Row():
                            cb_timestamp = gr.Checkbox(value=True, label="Add a timestamp to the end of the filename",
                                                       interactive=True)
                        with gr.Row():
                            btn_run = gr.Button("TRANSLATE SUBTITLE FILE", variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label="Output", scale=5)
                            files_subtitles = gr.Files(label="Downloadable output file", scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)
                        with gr.Column():
                            md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                    btn_run.click(fn=self.nllb_inf.translate_file,
                                  inputs=[file_subs, dd_nllb_model, dd_nllb_sourcelang, dd_nllb_targetlang,
                                          nb_max_length, cb_timestamp],
                                  outputs=[tb_indicator, files_subtitles])

                    btn_openfolder.click(fn=lambda: self.open_folder(os.path.join("outputs", "translations")),
                                         inputs=None,
                                         outputs=None)

        # Launch the app with optional gradio settings
        launch_args = {}
        if self.args.share:
            launch_args['share'] = self.args.share
        if self.args.server_name:
            launch_args['server_name'] = self.args.server_name
        if self.args.server_port:
            launch_args['server_port'] = self.args.server_port
        if self.args.username and self.args.password:
            launch_args['auth'] = (self.args.username, self.args.password)
        if self.args.root_path:
            launch_args['root_path'] = self.args.root_path
        launch_args['inbrowser'] = True

        self.app.queue(api_open=False).launch(**launch_args)


# Create the parser for command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--whisper_type', type=str, default="faster-whisper",
                    help='A type of the whisper implementation between: ["whisper", "faster-whisper", "insanely-fast-whisper"]')
parser.add_argument('--share', type=bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=bool, default=False, nargs='?', const=True, help='enable api or not')
parser.add_argument('--whisper_model_dir', type=str, default=os.path.join("models", "Whisper"),
                    help='Directory path of the whisper model')
parser.add_argument('--faster_whisper_model_dir', type=str, default=os.path.join("models", "Whisper", "faster-whisper"),
                    help='Directory path of the faster-whisper model')
parser.add_argument('--insanely_fast_whisper_model_dir', type=str,
                    default=os.path.join("models", "Whisper", "insanely-fast-whisper"),
                    help='Directory path of the insanely-fast-whisper model')
parser.add_argument('--diarization_model_dir', type=str, default=os.path.join("models", "Diarization"),
                    help='Directory path of the diarization model')
parser.add_argument('--nllb_model_dir', type=str, default=os.path.join("models", "NLLB"),
                    help='Directory path of the Facebook NLLB model')
parser.add_argument('--output_dir', type=str, default=os.path.join("outputs"), help='Directory path of the outputs')
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
