import os
import argparse
import gradio as gr
from gradio_i18n import Translate, gettext as _
import yaml

from modules.utils.paths import (FASTER_WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, WHISPER_MODELS_DIR,
                                 INSANELY_FAST_WHISPER_MODELS_DIR, NLLB_MODELS_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR, I18N_YAML_PATH)
from modules.utils.files_manager import load_yaml, MEDIA_EXTENSION
from modules.utils.preset_manager import (list_presets, load_preset, save_preset, delete_preset,
                                          sanitize_preset_name, pipeline_values_to_preset,
                                          preset_to_pipeline_values, keep_available_choices)
from modules.whisper.whisper_factory import WhisperFactory
from modules.translation.nllb_inference import NLLBInference
from modules.ui.htmls import *
from modules.utils.cli_manager import str2bool
from modules.utils.youtube_manager import get_ytmetas
from modules.translation.deepl_api import DeepLAPI
from modules.whisper.data_classes import *
from modules.utils.logger import get_logger


logger = get_logger()


class App:
    def __init__(self, args):
        self.args = args
        # Check every 1 hour (3600) for cached files and delete them if older than 1 day (86400)
        self.app = gr.Blocks(css=CSS, theme=self.args.theme, delete_cache=(3600, 86400))
        self.whisper_inf = WhisperFactory.create_whisper_inference(
            whisper_type=self.args.whisper_type,
            whisper_model_dir=self.args.whisper_model_dir,
            faster_whisper_model_dir=self.args.faster_whisper_model_dir,
            insanely_fast_whisper_model_dir=self.args.insanely_fast_whisper_model_dir,
            uvr_model_dir=self.args.uvr_model_dir,
            output_dir=self.args.output_dir,
        )
        self.nllb_inf = NLLBInference(
            model_dir=self.args.nllb_model_dir,
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.deepl_api = DeepLAPI(
            output_dir=os.path.join(self.args.output_dir, "translations")
        )
        self.preset_dropdowns = []
        self.preset_events = []
        self.i18n = load_yaml(I18N_YAML_PATH)
        self.default_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        logger.info(f"Use \"{self.args.whisper_type}\" implementation\n"
                    f"Device \"{self.whisper_inf.device}\" is detected")

    def create_pipeline_inputs(self):
        whisper_params = self.default_params["whisper"]
        vad_params = self.default_params["vad"]
        diarization_params = self.default_params["diarization"]
        uvr_params = self.default_params["bgm_separation"]

        with gr.Row(equal_height=True):
            dd_preset = gr.Dropdown(choices=list_presets(), value=None, label=_("Preset"),
                                    info=_("Save the settings below as a preset, or load a saved one"),
                                    allow_custom_value=True, scale=4)
            self.preset_dropdowns.append(dd_preset)
            btn_preset_load = gr.Button(_("LOAD"), scale=1)
            btn_preset_save = gr.Button(_("SAVE"), scale=1)
            btn_preset_delete = gr.Button(_("DELETE"), scale=1)

        with gr.Row():
            dd_model = gr.Dropdown(choices=self.whisper_inf.available_models, value=whisper_params["model_size"],
                                   label=_("Model"), allow_custom_value=True)
            dd_lang = gr.Dropdown(choices=self.whisper_inf.available_langs + [AUTOMATIC_DETECTION],
                                  value=AUTOMATIC_DETECTION if whisper_params["lang"] == AUTOMATIC_DETECTION.unwrap()
                                  else whisper_params["lang"], label=_("Language"))
            dd_file_format = gr.Dropdown(choices=["SRT", "WebVTT", "txt", "LRC"], value=whisper_params["file_format"], label=_("File Format"))
        with gr.Row():
            cb_translate = gr.Checkbox(value=whisper_params["is_translate"], label=_("Translate to English?"),
                                       interactive=True)
        with gr.Row():
            cb_timestamp = gr.Checkbox(value=whisper_params["add_timestamp"],
                                       label=_("Add a timestamp to the end of the filename"),
                                       interactive=True)

        with gr.Accordion(_("Advanced Parameters"), open=False):
            whisper_inputs = WhisperParams.to_gradio_inputs(defaults=whisper_params, only_advanced=True,
                                                            whisper_type=self.args.whisper_type,
                                                            available_compute_types=self.whisper_inf.available_compute_types,
                                                            compute_type=self.whisper_inf.current_compute_type)

        with gr.Accordion(_("Background Music Remover Filter"), open=False):
            uvr_inputs = BGMSeparationParams.to_gradio_input(defaults=uvr_params,
                                                             available_models=self.whisper_inf.music_separator.available_models,
                                                             available_devices=self.whisper_inf.music_separator.available_devices,
                                                             device=self.whisper_inf.music_separator.device)

        with gr.Accordion(_("Voice Detection Filter"), open=False):
            vad_inputs = VadParams.to_gradio_inputs(defaults=vad_params)

        with gr.Accordion(_("Diarization"), open=False):
            diarization_inputs = DiarizationParams.to_gradio_inputs(defaults=diarization_params,
                                                                    available_devices=self.whisper_inf.diarizer.available_device,
                                                                    device=self.whisper_inf.diarizer.device)

        pipeline_inputs = [dd_model, dd_lang, cb_translate] + whisper_inputs + vad_inputs + diarization_inputs + uvr_inputs

        # The preset holds whatever is in the UI right now, so every handler takes the pipeline
        # inputs and gives them back. Each tab builds its own widgets, hence its own wiring.
        preset_params = [dd_file_format, cb_timestamp] + pipeline_inputs

        def on_preset_load(preset_name, file_format, add_timestamp, *pipeline_values):
            try:
                preset = load_preset(preset_name)
            except (ValueError, FileNotFoundError, OSError) as e:
                raise gr.Error(str(e))

            whisper_preset = preset.get("whisper") or {}
            current_values = [file_format, add_timestamp] + list(pipeline_values)
            values = [whisper_preset.get("file_format", file_format),
                      whisper_preset.get("add_timestamp", add_timestamp)] + \
                preset_to_pipeline_values(preset, list(pipeline_values))

            gr.Info(_("Loaded preset") + f" : {sanitize_preset_name(preset_name)}")
            return keep_available_choices(preset_params, values, current_values)

        btn_preset_load.click(fn=on_preset_load, inputs=[dd_preset] + preset_params,
                              outputs=preset_params)
        # Saving and deleting change the list every tab shows, so they are wired in `launch()`
        # once the other tabs exist. Loading only touches this tab, so it is wired here.
        self.preset_events.append((dd_preset, btn_preset_save, btn_preset_delete, preset_params))

        return (
            pipeline_inputs,
            dd_file_format,
            cb_timestamp
        )

    def preset_choices(self, selected: str = None, selected_index: int = None):
        """
        Update every tab's preset dropdown. Only the tab that was used gets its selection changed;
        the others just get the new list, so they keep whatever they had selected.
        """
        presets = list_presets()
        return [gr.Dropdown(choices=presets, value=selected) if index == selected_index
                else gr.Dropdown(choices=presets)
                for index in range(len(self.preset_dropdowns))]

    def preset_saver(self, index: int):
        """Build the SAVE handler for one tab. `index` is the dropdown that keeps the selection."""
        def on_preset_save(preset_name, file_format, add_timestamp, *pipeline_values):
            try:
                preset_name = sanitize_preset_name(preset_name)
                save_preset(preset_name, pipeline_values_to_preset(list(pipeline_values),
                                                                   file_format, add_timestamp))
            except (ValueError, OSError) as e:
                raise gr.Error(str(e))

            gr.Info(_("Saved preset") + f" : {preset_name}")
            return self.preset_choices(selected=preset_name, selected_index=index)
        return on_preset_save

    def preset_deleter(self, index: int):
        """Build the DELETE handler for one tab."""
        def on_preset_delete(preset_name):
            try:
                delete_preset(preset_name)
            except (ValueError, FileNotFoundError, OSError) as e:
                raise gr.Error(str(e))

            gr.Info(_("Deleted preset") + f" : {sanitize_preset_name(preset_name)}")
            return self.preset_choices(selected=None, selected_index=index)
        return on_preset_delete

    def launch(self):
        translation_params = self.default_params["translation"]
        deepl_params = translation_params["deepl"]
        nllb_params = translation_params["nllb"]
        uvr_params = self.default_params["bgm_separation"]

        with self.app:
            lang = gr.Radio(choices=list(self.i18n.keys()),
                            label=_("Language"), interactive=True,
                            visible=False,  # Set it by development purpose.
                            )
            with Translate(self.i18n):  # Add `lang = lang` here to test dynamic change of the languages.
                with gr.Row():
                    with gr.Column():
                        gr.Markdown(MARKDOWN, elem_id="md_project")
                with gr.Tabs():
                    with gr.TabItem(_("File")):  # tab1
                        with gr.Column():
                            input_file = gr.Files(type="filepath", label=_("Upload File here"), file_types=MEDIA_EXTENSION)
                            tb_input_folder = gr.Textbox(label="Input Folder Path (Optional)",
                                                         info="Optional: Specify the folder path where the input files are located, if you prefer to use local files instead of uploading them."
                                                              " Leave this field empty if you do not wish to use a local path.",
                                                         visible=self.args.colab,
                                                         value="")
                            cb_include_subdirectory = gr.Checkbox(label="Include Subdirectory Files",
                                                                  info="When using Input Folder Path above, whether to include all files in the subdirectory or not.",
                                                                  visible=self.args.colab,
                                                                  value=False)
                            cb_save_same_dir = gr.Checkbox(label="Save outputs at same directory",
                                                           info="When using Input Folder Path above, whether to save output in the same directory as inputs or not, in addition to the original"
                                                                " output directory.",
                                                           visible=self.args.colab,
                                                           value=True)
                        pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3, interactive=False)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [input_file, tb_input_folder, cb_include_subdirectory, cb_save_same_dir,
                                  dd_file_format, cb_timestamp]
                        params = params + pipeline_params
                        btn_run.click(fn=self.whisper_inf.transcribe_file,
                                      inputs=params,
                                      outputs=[tb_indicator, files_subtitles])
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("Youtube")):  # tab2
                        with gr.Row():
                            tb_youtubelink = gr.Textbox(label=_("Youtube Link"))
                        with gr.Row(equal_height=True):
                            with gr.Column():
                                img_thumbnail = gr.Image(label=_("Youtube Thumbnail"))
                            with gr.Column():
                                tb_title = gr.Label(label=_("Youtube Title"))
                                tb_description = gr.Textbox(label=_("Youtube Description"), max_lines=15)

                        pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [tb_youtubelink, dd_file_format, cb_timestamp]

                        btn_run.click(fn=self.whisper_inf.transcribe_youtube,
                                      inputs=params + pipeline_params,
                                      outputs=[tb_indicator, files_subtitles])
                        tb_youtubelink.change(get_ytmetas, inputs=[tb_youtubelink],
                                              outputs=[img_thumbnail, tb_title, tb_description])
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("Mic")):  # tab3
                        with gr.Row():
                            mic_input = gr.Microphone(label=_("Record with Mic"), type="filepath", interactive=True,
                                                      show_download_button=True)

                        pipeline_params, dd_file_format, cb_timestamp = self.create_pipeline_inputs()

                        with gr.Row():
                            btn_run = gr.Button(_("GENERATE SUBTITLE FILE"), variant="primary")
                        with gr.Row():
                            tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                            files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                            btn_openfolder = gr.Button('📂', scale=1)

                        params = [mic_input, dd_file_format, cb_timestamp]

                        btn_run.click(fn=self.whisper_inf.transcribe_mic,
                                      inputs=params + pipeline_params,
                                      outputs=[tb_indicator, files_subtitles])
                        btn_openfolder.click(fn=lambda: self.open_folder("outputs"), inputs=None, outputs=None)

                    with gr.TabItem(_("T2T Translation")):  # tab 4
                        with gr.Row():
                            file_subs = gr.Files(type="filepath", label=_("Upload Subtitle Files to translate here"))

                        with gr.TabItem(_("DeepL API")):  # sub tab1
                            with gr.Row():
                                tb_api_key = gr.Textbox(label=_("Your Auth Key (API KEY)"),
                                                        value=deepl_params["api_key"])
                            with gr.Row():
                                dd_source_lang = gr.Dropdown(label=_("Source Language"),
                                                             value=AUTOMATIC_DETECTION if deepl_params["source_lang"] == AUTOMATIC_DETECTION.unwrap()
                                                             else deepl_params["source_lang"],
                                                             choices=list(self.deepl_api.available_source_langs.keys()))
                                dd_target_lang = gr.Dropdown(label=_("Target Language"),
                                                             value=deepl_params["target_lang"],
                                                             choices=list(self.deepl_api.available_target_langs.keys()))
                            with gr.Row():
                                cb_is_pro = gr.Checkbox(label=_("Pro User?"), value=deepl_params["is_pro"])
                            with gr.Row():
                                cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                           label=_("Add a timestamp to the end of the filename"),
                                                           interactive=True)
                            with gr.Row():
                                btn_run = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                                files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                                btn_openfolder = gr.Button('📂', scale=1)

                        btn_run.click(fn=self.deepl_api.translate_deepl,
                                      inputs=[tb_api_key, file_subs, dd_source_lang, dd_target_lang,
                                              cb_is_pro, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles])

                        btn_openfolder.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None)

                        with gr.TabItem(_("NLLB")):  # sub tab2
                            with gr.Row():
                                dd_model_size = gr.Dropdown(label=_("Model"), value=nllb_params["model_size"],
                                                            choices=self.nllb_inf.available_models)
                                dd_source_lang = gr.Dropdown(label=_("Source Language"),
                                                             value=nllb_params["source_lang"],
                                                             choices=self.nllb_inf.available_source_langs)
                                dd_target_lang = gr.Dropdown(label=_("Target Language"),
                                                             value=nllb_params["target_lang"],
                                                             choices=self.nllb_inf.available_target_langs)
                            with gr.Row():
                                nb_max_length = gr.Number(label="Max Length Per Line", value=nllb_params["max_length"],
                                                          precision=0)
                            with gr.Row():
                                cb_timestamp = gr.Checkbox(value=translation_params["add_timestamp"],
                                                           label=_("Add a timestamp to the end of the filename"),
                                                           interactive=True)
                            with gr.Row():
                                btn_run = gr.Button(_("TRANSLATE SUBTITLE FILE"), variant="primary")
                            with gr.Row():
                                tb_indicator = gr.Textbox(label=_("Output"), scale=5)
                                files_subtitles = gr.Files(label=_("Downloadable output file"), scale=3)
                                btn_openfolder = gr.Button('📂', scale=1)
                            with gr.Column():
                                md_vram_table = gr.HTML(NLLB_VRAM_TABLE, elem_id="md_nllb_vram_table")

                        btn_run.click(fn=self.nllb_inf.translate_file,
                                      inputs=[file_subs, dd_model_size, dd_source_lang, dd_target_lang,
                                              nb_max_length, cb_timestamp],
                                      outputs=[tb_indicator, files_subtitles])

                        btn_openfolder.click(
                            fn=lambda: self.open_folder(os.path.join(self.args.output_dir, "translations")),
                            inputs=None,
                            outputs=None)

                    with gr.TabItem(_("BGM Separation")):
                        files_audio = gr.Files(type="filepath", label=_("Upload Audio Files to separate background music"))
                        dd_uvr_device = gr.Dropdown(label=_("Device"), value=self.whisper_inf.music_separator.device,
                                                    choices=self.whisper_inf.music_separator.available_devices)
                        dd_uvr_model_size = gr.Dropdown(label=_("Model"), value=uvr_params["uvr_model_size"],
                                                        choices=self.whisper_inf.music_separator.available_models)
                        nb_uvr_segment_size = gr.Number(label="Segment Size", value=uvr_params["segment_size"],
                                                        precision=0)
                        cb_uvr_save_file = gr.Checkbox(label=_("Save separated files to output"),
                                                       value=True, visible=False)
                        btn_run = gr.Button(_("SEPARATE BACKGROUND MUSIC"), variant="primary")
                        with gr.Column():
                            with gr.Row():
                                ad_instrumental = gr.Audio(label=_("Instrumental"), scale=8)
                                btn_open_instrumental_folder = gr.Button('📂', scale=1)
                            with gr.Row():
                                ad_vocals = gr.Audio(label=_("Vocals"), scale=8)
                                btn_open_vocals_folder = gr.Button('📂', scale=1)

                        btn_run.click(fn=self.whisper_inf.music_separator.separate_files,
                                      inputs=[files_audio, dd_uvr_model_size, dd_uvr_device, nb_uvr_segment_size,
                                              cb_uvr_save_file],
                                      outputs=[ad_instrumental, ad_vocals])
                        btn_open_instrumental_folder.click(inputs=None,
                                                           outputs=None,
                                                           fn=lambda: self.open_folder(os.path.join(
                                                               self.args.output_dir, "UVR", "instrumental"
                                                           )))
                        btn_open_vocals_folder.click(inputs=None,
                                                     outputs=None,
                                                     fn=lambda: self.open_folder(os.path.join(
                                                         self.args.output_dir, "UVR", "vocals"
                                                     )))

            # Every tab builds its own dropdown, so saving or deleting has to write to all of
            # them, which is only possible now that they all exist.
            for index, (dd_preset, btn_save, btn_delete, preset_params) in enumerate(self.preset_events):
                btn_save.click(fn=self.preset_saver(index), inputs=[dd_preset] + preset_params,
                               outputs=self.preset_dropdowns)
                btn_delete.click(fn=self.preset_deleter(index), inputs=dd_preset,
                                 outputs=self.preset_dropdowns)

            # Presets can also appear or disappear on disk while the app runs, and the choices are
            # baked in when the UI is built, so refresh them whenever a page loads. Don't do this
            # on the dropdown's own focus event: a `choices` update re-filters the list against the
            # text in the box, which would hide every preset but the selected one.
            self.app.load(fn=lambda: self.preset_choices(), inputs=None,
                          outputs=self.preset_dropdowns)

        # Launch the app with optional gradio settings
        args = self.args
        self.app.queue(
            api_open=args.api_open
        ).launch(
            share=args.share,
            server_name=args.server_name,
            server_port=args.server_port,
            auth=(args.username, args.password) if args.username and args.password else None,
            root_path=args.root_path,
            inbrowser=args.inbrowser,
            ssl_verify=args.ssl_verify,
            ssl_keyfile=args.ssl_keyfile,
            ssl_keyfile_password=args.ssl_keyfile_password,
            ssl_certfile=args.ssl_certfile,
            allowed_paths=eval(args.allowed_paths) if args.allowed_paths else None
        )

    @staticmethod
    def open_folder(folder_path: str):
        if os.path.exists(folder_path):
            os.system(f"start {folder_path}")
        else:
            os.makedirs(folder_path, exist_ok=True)
            logger.info(f"The directory path {folder_path} has newly created.")


parser = argparse.ArgumentParser()
parser.add_argument('--whisper_type', type=str, default=WhisperImpl.FASTER_WHISPER.value,
                    choices=[item.value for item in WhisperImpl],
                    help='A type of the whisper implementation (Github repo name)')
parser.add_argument('--share', type=str2bool, default=False, nargs='?', const=True, help='Gradio share value')
parser.add_argument('--server_name', type=str, default=None, help='Gradio server host')
parser.add_argument('--server_port', type=int, default=None, help='Gradio server port')
parser.add_argument('--root_path', type=str, default=None, help='Gradio root path')
parser.add_argument('--username', type=str, default=None, help='Gradio authentication username')
parser.add_argument('--password', type=str, default=None, help='Gradio authentication password')
parser.add_argument('--theme', type=str, default=None, help='Gradio Blocks theme')
parser.add_argument('--colab', type=str2bool, default=False, nargs='?', const=True, help='Is colab user or not')
parser.add_argument('--api_open', type=str2bool, default=False, nargs='?', const=True,
                    help='Enable api or not in Gradio')
parser.add_argument('--allowed_paths', type=str, default=None, help='Gradio allowed paths')
parser.add_argument('--inbrowser', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to automatically start Gradio app or not')
parser.add_argument('--ssl_verify', type=str2bool, default=True, nargs='?', const=True,
                    help='Whether to verify SSL or not')
parser.add_argument('--ssl_keyfile', type=str, default=None, help='SSL Key file location')
parser.add_argument('--ssl_keyfile_password', type=str, default=None, help='SSL Key file password')
parser.add_argument('--ssl_certfile', type=str, default=None, help='SSL cert file location')
parser.add_argument('--whisper_model_dir', type=str, default=WHISPER_MODELS_DIR,
                    help='Directory path of the whisper model')
parser.add_argument('--faster_whisper_model_dir', type=str, default=FASTER_WHISPER_MODELS_DIR,
                    help='Directory path of the faster-whisper model')
parser.add_argument('--insanely_fast_whisper_model_dir', type=str,
                    default=INSANELY_FAST_WHISPER_MODELS_DIR,
                    help='Directory path of the insanely-fast-whisper model')
parser.add_argument('--diarization_model_dir', type=str, default=DIARIZATION_MODELS_DIR,
                    help='Directory path of the diarization model')
parser.add_argument('--nllb_model_dir', type=str, default=NLLB_MODELS_DIR,
                    help='Directory path of the Facebook NLLB model')
parser.add_argument('--uvr_model_dir', type=str, default=UVR_MODELS_DIR,
                    help='Directory path of the UVR model')
parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help='Directory path of the outputs')
_args = parser.parse_args()

if __name__ == "__main__":
    app = App(args=_args)
    app.launch()
