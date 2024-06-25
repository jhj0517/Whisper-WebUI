import gradio as gr
import os
import argparse

from modules.whisper_Inference import WhisperInference
from modules.faster_whisper_inference import FasterWhisperInference
from modules.insanely_fast_whisper_inference import InsanelyFastWhisperInference
from modules.nllb_inference import NLLBInference
from ui.htmls import *
from modules.youtube_manager import get_ytmetas
from modules.deepl_api import DeepLAPI
from modules.whisper_parameter import *


class App:
    def __init__(self, args):
        self.args = args
        self.app = gr.Blocks(css=CSS, theme=self.args.theme)
        self.whisper_inf = self.init_whisper()
        print(f"Use \"{self.args.whisper_type}\" implementation")
        print(f"Device \"{self.whisper_inf.device}\" is detected")
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=self.args.output_dir
        )
        self.deepl_api = DeepLAPI(
            output_dir=self.args.output_dir
        )

    def init_whisper(self):
        whisper_type = self.args.whisper_type.lower().strip()

        if whisper_type in ["faster_whisper", "faster-whisper", "fasterwhisper"]:
            whisper_inf = FasterWhisperInference(
                model_dir=self.args.faster_whisper_model_dir,
                output_dir=self.args.output_dir
            )
        elif whisper_type in ["whisper"]:
            whisper_inf = WhisperInference(
                model_dir=self.args.whisper_model_dir,
                output_dir=self.args.output_dir
            )
        elif whisper_type in ["insanely_fast_whisper", "insanely-fast-whisper", "insanelyfastwhisper",
                              "insanely_faster_whisper", "insanely-faster-whisper", "insanelyfasterwhisper"]:
            whisper_inf = InsanelyFastWhisperInference(
                model_dir=self.args.insanely_fast_whisper_model_dir,
                output_dir=self.args.output_dir
            )
        else:
            whisper_inf = FasterWhisperInference(
                model_dir=self.args.faster_whisper_model_dir,
                output_dir=self.args.output_dir
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
                    with gr.Row():
                        input_file = gr.Files(type="filepath", label="Upload File here")
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v2",
                                               label="Model")
                        dd_lang = gr.Dropdown(choices=["Automatic Detection"] + self.whisper_inf.available_langs,
                                              value="Automatic Detection", label="Language")
                        dd_file_format = gr.Dropdown(["SRT", "WebVTT", "txt"], value="SRT", label="File Format")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
                    with gr.Row():
                        cb_timestamp = gr.Checkbox(value=True, label="Add a timestamp to the end of the filename", interactive=True)
                    with gr.Accordion("Advanced_Parameters", open=False):
                        nb_beam_size = gr.Number(label="Beam Size", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="Log Probability Threshold", value=-1.0, interactive=True)
                        nb_no_speech_threshold = gr.Number(label="No Speech Threshold", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="Compute Type", choices=self.whisper_inf.available_compute_types, value=self.whisper_inf.current_compute_type, interactive=True)
                        nb_best_of = gr.Number(label="Best Of", value=5, interactive=True)
                        nb_patience = gr.Number(label="Patience", value=1, interactive=True)
                        cb_condition_on_previous_text = gr.Checkbox(label="Condition On Previous Text", value=True, interactive=True)
                        tb_initial_prompt = gr.Textbox(label="Initial Prompt", value=None, interactive=True)
                        sd_temperature = gr.Slider(label="Temperature", value=0, step=0.01, maximum=1.0, interactive=True)
                        nb_compression_ratio_threshold = gr.Number(label="Compression Ratio Threshold", value=2.4, interactive=True)
                    with gr.Accordion("VAD Options", open=False, visible=isinstance(self.whisper_inf, FasterWhisperInference)):
                        cb_vad_filter = gr.Checkbox(label="Enable Silero VAD Filter", value=False, interactive=True)
                        sd_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold", value=0.5)
                        nb_min_speech_duration_ms = gr.Number(label="Minimum Speech Duration (ms)", precision=0, value=250)
                        nb_max_speech_duration_s = gr.Number(label="Maximum Speech Duration (s)", value=9999)
                        nb_min_silence_duration_ms = gr.Number(label="Minimum Silence Duration (ms)", precision=0, value=2000)
                        nb_window_size_sample = gr.Number(label="Window Size (samples)", precision=0, value=1024)
                        nb_speech_pad_ms = gr.Number(label="Speech Padding (ms)", precision=0, value=400)
                    with gr.Accordion("Insanely Fast Whisper Parameters", open=False, visible=isinstance(self.whisper_inf, InsanelyFastWhisperInference)):
                        nb_chunk_length_s = gr.Number(label="Chunk Lengths (sec)", value=30, precision=0)
                        nb_batch_size = gr.Number(label="Batch Size", value=24, precision=0)
                    with gr.Accordion("Diarization Parameters", open=False):
                        cb_diarize = gr.Checkbox(label="Enable Diarization")
                        tb_hf_token = gr.Text(label="HuggingFace Token", value="")
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Files(label="Downloadable output file", scale=3, interactive=False)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [input_file, dd_file_format, cb_timestamp]
                    whisper_params = WhisperParameters(model_size=dd_model,
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
                                                       window_size_sample=nb_window_size_sample,
                                                       speech_pad_ms=nb_speech_pad_ms,
                                                       chunk_length_s=nb_chunk_length_s,
                                                       batch_size=nb_batch_size,
                                                       is_diarize=cb_diarize,
                                                       hf_token=tb_hf_token)

                    btn_run.click(fn=self.whisper_inf.transcribe_file,
                                  inputs=params + whisper_params.to_list(),
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
                    with gr.Accordion("Advanced_Parameters", open=False):
                        nb_beam_size = gr.Number(label="Beam Size", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="Log Probability Threshold", value=-1.0, interactive=True)
                        nb_no_speech_threshold = gr.Number(label="No Speech Threshold", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="Compute Type", choices=self.whisper_inf.available_compute_types, value=self.whisper_inf.current_compute_type, interactive=True)
                        nb_best_of = gr.Number(label="Best Of", value=5, interactive=True)
                        nb_patience = gr.Number(label="Patience", value=1, interactive=True)
                        cb_condition_on_previous_text = gr.Checkbox(label="Condition On Previous Text", value=True, interactive=True)
                        tb_initial_prompt = gr.Textbox(label="Initial Prompt", value=None, interactive=True)
                        sd_temperature = gr.Slider(label="Temperature", value=0, step=0.01, maximum=1.0, interactive=True)
                        nb_compression_ratio_threshold = gr.Number(label="Compression Ratio Threshold", value=2.4, interactive=True)
                    with gr.Accordion("VAD Options", open=False, visible=isinstance(self.whisper_inf, FasterWhisperInference)):
                        cb_vad_filter = gr.Checkbox(label="Enable Silero VAD Filter", value=False, interactive=True)
                        sd_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold", value=0.5)
                        nb_min_speech_duration_ms = gr.Number(label="Minimum Speech Duration (ms)", precision=0, value=250)
                        nb_max_speech_duration_s = gr.Number(label="Maximum Speech Duration (s)", value=9999)
                        nb_min_silence_duration_ms = gr.Number(label="Minimum Silence Duration (ms)", precision=0, value=2000)
                        nb_window_size_sample = gr.Number(label="Window Size (samples)", precision=0, value=1024)
                        nb_speech_pad_ms = gr.Number(label="Speech Padding (ms)", precision=0, value=400)
                    with gr.Accordion("Insanely Fast Whisper Parameters", open=False,
                                      visible=isinstance(self.whisper_inf, InsanelyFastWhisperInference)):
                        nb_chunk_length_s = gr.Number(label="Chunk Lengths (sec)", value=30, precision=0)
                        nb_batch_size = gr.Number(label="Batch Size", value=24, precision=0)
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Files(label="Downloadable output file", scale=3)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [tb_youtubelink, dd_file_format, cb_timestamp]
                    whisper_params = WhisperParameters(model_size=dd_model,
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
                                                       window_size_sample=nb_window_size_sample,
                                                       speech_pad_ms=nb_speech_pad_ms,
                                                       chunk_length_s=nb_chunk_length_s,
                                                       batch_size=nb_batch_size,
                                                       is_diarize=cb_diarize,
                                                       hf_token=tb_hf_token)

                    btn_run.click(fn=self.whisper_inf.transcribe_youtube,
                                  inputs=params + whisper_params.to_list(),
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
                    with gr.Accordion("Advanced_Parameters", open=False):
                        nb_beam_size = gr.Number(label="Beam Size", value=1, precision=0, interactive=True)
                        nb_log_prob_threshold = gr.Number(label="Log Probability Threshold", value=-1.0, interactive=True)
                        nb_no_speech_threshold = gr.Number(label="No Speech Threshold", value=0.6, interactive=True)
                        dd_compute_type = gr.Dropdown(label="Compute Type", choices=self.whisper_inf.available_compute_types, value=self.whisper_inf.current_compute_type, interactive=True)
                        nb_best_of = gr.Number(label="Best Of", value=5, interactive=True)
                        nb_patience = gr.Number(label="Patience", value=1, interactive=True)
                        cb_condition_on_previous_text = gr.Checkbox(label="Condition On Previous Text", value=True, interactive=True)
                        tb_initial_prompt = gr.Textbox(label="Initial Prompt", value=None, interactive=True)
                        sd_temperature = gr.Slider(label="Temperature", value=0, step=0.01, maximum=1.0, interactive=True)
                    with gr.Accordion("VAD Options", open=False, visible=isinstance(self.whisper_inf, FasterWhisperInference)):
                        cb_vad_filter = gr.Checkbox(label="Enable Silero VAD Filter", value=False, interactive=True)
                        sd_threshold = gr.Slider(minimum=0.0, maximum=1.0, step=0.01, label="Speech Threshold", value=0.5)
                        nb_min_speech_duration_ms = gr.Number(label="Minimum Speech Duration (ms)", precision=0, value=250)
                        nb_max_speech_duration_s = gr.Number(label="Maximum Speech Duration (s)", value=9999)
                        nb_min_silence_duration_ms = gr.Number(label="Minimum Silence Duration (ms)", precision=0, value=2000)
                        nb_window_size_sample = gr.Number(label="Window Size (samples)", precision=0, value=1024)
                        nb_speech_pad_ms = gr.Number(label="Speech Padding (ms)", precision=0, value=400)
                    with gr.Accordion("Insanely Fast Whisper Parameters", open=False,
                                      visible=isinstance(self.whisper_inf, InsanelyFastWhisperInference)):
                        nb_chunk_length_s = gr.Number(label="Chunk Lengths (sec)", value=30, precision=0)
                        nb_batch_size = gr.Number(label="Batch Size", value=24, precision=0)
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=5)
                        files_subtitles = gr.Files(label="Downloadable output file", scale=3)
                        btn_openfolder = gr.Button('📂', scale=1)

                    params = [mic_input, dd_file_format]
                    whisper_params = WhisperParameters(model_size=dd_model,
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
                                                       window_size_sample=nb_window_size_sample,
                                                       speech_pad_ms=nb_speech_pad_ms,
                                                       chunk_length_s=nb_chunk_length_s,
                                                       batch_size=nb_batch_size,
                                                       is_diarize=cb_diarize,
                                                       hf_token=tb_hf_token)

                    btn_run.click(fn=self.whisper_inf.transcribe_mic,
                                  inputs=params + whisper_params.to_list(),
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
                                  inputs=[file_subs, dd_nllb_model, dd_nllb_sourcelang, dd_nllb_targetlang, cb_timestamp],
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
parser.add_argument('--whisper_type', type=str, default="faster-whisper", help='A type of the whisper implementation between: ["whisper", "faster-whisper", "insanely-fast-whisper"]')
parser.add_argument('--share', type=bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=bool, default=False, nargs='?', const=True, help='enable api or not')
parser.add_argument('--whisper_model_dir', type=str, default=os.path.join("models", "Whisper"), help='Directory path of the whisper model')
parser.add_argument('--faster_whisper_model_dir', type=str, default=os.path.join("models", "Whisper", "faster-whisper"), help='Directory path of the faster-whisper model')
parser.add_argument('--insanely_fast_whisper_model_dir', type=str, default=os.path.join("models", "Whisper", "insanely-fast-whisper"), help='Directory path of the insanely-fast-whisper model')
parser.add_argument('--nllb_model_dir', type=str, default=os.path.join("models", "NLLB"), help='Directory path of the Facebook NLLB model')
parser.add_argument('--output_dir', type=str, default=os.path.join("outputs"), help='Directory path of the outputs')
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
