import whisper
import gradio as gr
import os
from datetime import datetime

from .base_interface import BaseInterface
from modules.subtitle_manager import get_srt, get_vtt, get_txt, write_file, safe_filename
from modules.youtube_manager import get_ytdata, get_ytaudio

DEFAULT_MODEL_SIZE = "large-v2"


class WhisperInference(BaseInterface):
    def __init__(self):
        super().__init__()
        self.current_model_size = None
        self.model = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))

    def transcribe_file(self,
                        fileobjs: list,
                        model_size: str,
                        lang: str,
                        subformat: str,
                        istranslate: bool,
                        add_timestamp: bool,
                        progress=gr.Progress()):
        """
        Write subtitle file from Files

        Parameters
        ----------
        fileobjs: list
            List of files to transcribe from gr.Files()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        subformat: str
            Subtitle format to write from gr.Dropdown(). Supported format: [SRT, WebVTT]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
            I use a forked version of whisper for this. To see more info : https://github.com/jhj0517/jhj0517-whisper/tree/add-progress-callback
        """
        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        try:
            if model_size != self.current_model_size or self.model is None:
                progress(0, desc="Initializing Model..")
                self.current_model_size = model_size
                self.model = whisper.load_model(name=model_size, download_root=os.path.join("models", "Whisper"))

            if lang == "Automatic Detection":
                lang = None

            progress(0, desc="Loading Audio..")

            files_info = {}
            for fileobj in fileobjs:

                audio = whisper.load_audio(fileobj.name)

                translatable_model = ["large", "large-v1", "large-v2"]
                if istranslate and self.current_model_size in translatable_model:
                    result = self.model.transcribe(audio=audio, language=lang, verbose=False, task="translate",
                                                   progress_callback=progress_callback)
                else:
                    result = self.model.transcribe(audio=audio, language=lang, verbose=False,
                                                   progress_callback=progress_callback)

                progress(1, desc="Completed!")

                file_name, file_ext = os.path.splitext(os.path.basename(fileobj.orig_name))
                file_name = safe_filename(file_name)
                timestamp = datetime.now().strftime("%m%d%H%M%S")
                if add_timestamp:
                    output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
                else:
                    output_path = os.path.join("outputs", f"{file_name}")

                if subformat == "SRT":
                    subtitle = get_srt(result["segments"])
                    write_file(subtitle, f"{output_path}.srt")
                elif subformat == "WebVTT":
                    subtitle = get_vtt(result["segments"])
                    write_file(subtitle, f"{output_path}.vtt")
                else :
                    subtitle = get_txt(result["segments"])
                    write_file(subtitle, f"{output_path}.txt")

                files_info[file_name] = subtitle

            total_result = ''
            for file_name, subtitle in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{subtitle}'

            return f"Done! Subtitle is in the outputs folder.\n\n{total_result}"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            self.release_cuda_memory()
            self.remove_input_files([fileobj.name for fileobj in fileobjs])

    


    def transcribe_youtube(self,
                           youtubelink: str,
                           model_size: str,
                           lang: str,
                           subformat: str,
                           istranslate: bool,
                           add_timestamp: bool,
                           progress=gr.Progress()):
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtubelink: str
            Link of Youtube to transcribe from gr.Textbox()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        subformat: str
            Subtitle format to write from gr.Dropdown(). Supported format: [SRT, WebVTT]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
            I use a forked version of whisper for this. To see more info : https://github.com/jhj0517/jhj0517-whisper/tree/add-progress-callback
        """
        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        try:
            if model_size != self.current_model_size or self.model is None:
                progress(0, desc="Initializing Model..")
                self.current_model_size = model_size
                self.model = whisper.load_model(name=model_size, download_root=os.path.join("models", "Whisper"))

            if lang == "Automatic Detection":
                lang = None

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtubelink)
            audio = whisper.load_audio(get_ytaudio(yt))

            translatable_model = ["large", "large-v1", "large-v2"]
            if istranslate and self.current_model_size in translatable_model:
                result = self.model.transcribe(audio=audio, language=lang, verbose=False, task="translate",
                                               progress_callback=progress_callback)
            else:
                result = self.model.transcribe(audio=audio, language=lang, verbose=False,
                                               progress_callback=progress_callback)

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            timestamp = datetime.now().strftime("%m%d%H%M%S")
            if add_timestamp:
                output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
            else:
                output_path = os.path.join("outputs", f"{file_name}")

            if subformat == "SRT":
                subtitle = get_srt(result["segments"])
                write_file(subtitle, f"{output_path}.srt")
            elif subformat == "WebVTT":
                subtitle = get_vtt(result["segments"])
                write_file(subtitle, f"{output_path}.vtt")

            return f"Done! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            yt = get_ytdata(youtubelink)
            file_path = get_ytaudio(yt)
            self.release_cuda_memory()
            self.remove_input_files([file_path])

    def transcribe_mic(self,
                       micaudio: str,
                       model_size: str,
                       lang: str,
                       subformat: str,
                       istranslate: bool,
                       progress=gr.Progress()):
        """
        Write subtitle file from microphone

        Parameters
        ----------
        micaudio: str
            Audio file path from gr.Microphone()
        model_size: str
            Whisper model size from gr.Dropdown()
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        subformat: str
            Subtitle format to write from gr.Dropdown(). Supported format: [SRT, WebVTT]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
            I use a forked version of whisper for this. To see more info : https://github.com/jhj0517/jhj0517-whisper/tree/add-progress-callback
        """
        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        try:
            if model_size != self.current_model_size or self.model is None:
                progress(0, desc="Initializing Model..")
                self.current_model_size = model_size
                self.model = whisper.load_model(name=model_size, download_root=os.path.join("models", "Whisper"))

            if lang == "Automatic Detection":
                lang = None

            progress(0, desc="Loading Audio..")

            translatable_model = ["large", "large-v1", "large-v2"]
            if istranslate and self.current_model_size in translatable_model:
                result = self.model.transcribe(audio=micaudio, language=lang, verbose=False, task="translate",
                                               progress_callback=progress_callback)
            else:
                result = self.model.transcribe(audio=micaudio, language=lang, verbose=False,
                                               progress_callback=progress_callback)

            progress(1, desc="Completed!")

            timestamp = datetime.now().strftime("%m%d%H%M%S")
            output_path = os.path.join("outputs", f"Mic-{timestamp}")

            if subformat == "SRT":
                subtitle = get_srt(result["segments"])
                write_file(subtitle, f"{output_path}.srt")
            elif subformat == "WebVTT":
                subtitle = get_vtt(result["segments"])
                write_file(subtitle, f"{output_path}.vtt")

            return f"Done! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            self.release_cuda_memory()
            self.remove_input_files([micaudio])
