import gradio as gr
import os
import argparse

from modules.whisper_Inference import WhisperInference
from modules.nllb_inference import NLLBInference
from ui.htmls import *
from modules.youtube_manager import get_ytmetas


class App:
    def __init__(self, args):
        self.args = args
        self.app = gr.Blocks(css=CSS)
        self.whisper_inf = WhisperInference()
        self.nllb_inf = NLLBInference()

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            print(f"The folder {folder_path} does not exist.")

    @staticmethod
    def on_change_models(model_size: str):
        translatable_model = ["large", "large-v1", "large-v2"]
        if model_size not in translatable_model:
            return gr.Checkbox.update(visible=False, value=False, interactive=False)
        else:
            return gr.Checkbox.update(visible=True, value=False, label="Translate to English?", interactive=True)

    def launch(self):
        with self.app:
            with gr.Row():
                with gr.Column():
                    gr.Markdown(MARKDOWN, elem_id="md_project")
            with gr.Tabs():
                with gr.TabItem("File"):  # tab1
                    with gr.Row():
                        input_file = gr.Files(type="file", label="Upload File here")
                    with gr.Row():
                        dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value="large-v2",
                                               label="Model")
                        dd_lang = gr.Dropdown(choices=["Automatic Detection"] + self.whisper_inf.available_langs,
                                              value="Automatic Detection", label="Language")
                        dd_subformat = gr.Dropdown(["SRT", "WebVTT"], value="SRT", label="Subtitle Format")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
                    with gr.Row():
                        cb_timestamp = gr.Checkbox(value=True, label="Add a timestamp to the end of the filename", interactive=True)
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=8)
                        btn_openfolder = gr.Button('📂', scale=2)

                    btn_run.click(fn=self.whisper_inf.transcribe_file,
                                  inputs=[input_file, dd_model, dd_lang, dd_subformat, cb_translate, cb_timestamp],
                                  outputs=[tb_indicator])
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
                        dd_subformat = gr.Dropdown(choices=["SRT", "WebVTT"], value="SRT", label="Subtitle Format")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
                    with gr.Row():
                        cb_timestamp = gr.Checkbox(value=True, label="Add a timestamp to the end of the filename",
                                                   interactive=True)
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=8)
                        btn_openfolder = gr.Button('📂', scale=2)

                    btn_run.click(fn=self.whisper_inf.transcribe_youtube,
                                  inputs=[tb_youtubelink, dd_model, dd_lang, dd_subformat, cb_translate, cb_timestamp],
                                  outputs=[tb_indicator])
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
                        dd_subformat = gr.Dropdown(["SRT", "WebVTT"], value="SRT", label="Subtitle Format")
                    with gr.Row():
                        cb_translate = gr.Checkbox(value=False, label="Translate to English?", interactive=True)
                    with gr.Row():
                        btn_run = gr.Button("GENERATE SUBTITLE FILE", variant="primary")
                    with gr.Row():
                        tb_indicator = gr.Textbox(label="Output", scale=8)
                        btn_openfolder = gr.Button('📂', scale=2)

                    btn_run.click(fn=self.whisper_inf.transcribe_mic,
                                  inputs=[mic_input, dd_model, dd_lang, dd_subformat, cb_translate],
                                  outputs=[tb_indicator])
                    btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)
                    dd_model.change(fn=self.on_change_models, inputs=[dd_model], outputs=[cb_translate])

                with gr.TabItem("T2T Translation"):  # tab 4
                    with gr.Row():
                        file_subs = gr.Files(type="file", label="Upload Subtitle Files to translate here",
                                             file_types=['.vtt', '.srt'])

                    with gr.TabItem("NLLB"):  # sub tab1
                        with gr.Row():
                            dd_nllb_model = gr.Dropdown(label="Model", value=self.nllb_inf.default_model_size,
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
                            tb_indicator = gr.Textbox(label="Output", scale=8)
                            btn_openfolder = gr.Button('📂', scale=2)
                        with gr.Column():
                            md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                    btn_run.click(fn=self.nllb_inf.translate_file,
                                  inputs=[file_subs, dd_nllb_model, dd_nllb_sourcelang, dd_nllb_targetlang, cb_timestamp],
                                  outputs=[tb_indicator])
                    btn_openfolder.click(fn=lambda: self.open_folder(os.path.join("outputs", "translations")),
                                         inputs=None,
                                         outputs=None)

        if self.args.share:
            self.app.queue(api_open=False).launch(share=True)
        else:
            self.app.queue(api_open=False).launch()


# Create the parser
parser = argparse.ArgumentParser()
parser.add_argument('--share', type=bool, default=False, nargs='?', const=True,
                    help='Share value')
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
