import whisper
import gradio as gr
import time
import os
from typing import BinaryIO, Union, Tuple
import numpy as np
from datetime import datetime
import torch

from .base_interface import BaseInterface
from modules.subtitle_manager import get_srt, get_vtt, write_file, safe_filename
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

        try:
            if model_size != self.current_model_size or self.model is None:
                self.initialize_model(model_size=model_size, progress=progress)

            files_info = {}
            for fileobj in fileobjs:
                progress(0, desc="Loading Audio..")
                audio = whisper.load_audio(fileobj.name)

                result, elapsed_time = self.transcribe(audio=audio,
                                                       lang=lang,
                                                       istranslate=istranslate,
                                                       progress=progress)
                progress(1, desc="Completed!")

                file_name, file_ext = os.path.splitext(os.path.basename(fileobj.orig_name))
                file_name = safe_filename(file_name)
                subtitle = self.generate_and_write_subtitle(
                    file_name=file_name,
                    transcribed_segments=result,
                    add_timestamp=add_timestamp,
                    subformat=subformat
                )

                files_info[file_name] = {"subtitle": subtitle, "elapsed_time": elapsed_time}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f"{info['subtitle']}"
                total_time += info["elapsed_time"]

            return f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
        except Exception as e:
            print(f"Error transcribing file: {str(e)}")
            return f"Error transcribing file: {str(e)}"
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
        try:
            if model_size != self.current_model_size or self.model is None:
                self.initialize_model(model_size=model_size, progress=progress)

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtubelink)
            audio = whisper.load_audio(get_ytaudio(yt))

            result, elapsed_time = self.transcribe(audio=audio,
                                                   lang=lang,
                                                   istranslate=istranslate,
                                                   progress=progress)
            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle = self.generate_and_write_subtitle(
                file_name=file_name,
                transcribed_segments=result,
                add_timestamp=add_timestamp,
                subformat=subformat
            )

            return f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            print(f"Error transcribing youtube video: {str(e)}")
            return f"Error transcribing youtube video: {str(e)}"
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

        try:
            if model_size != self.current_model_size or self.model is None:
                self.initialize_model(model_size=model_size, progress=progress)

            result, elapsed_time = self.transcribe(audio=micaudio,
                                                   lang=lang,
                                                   istranslate=istranslate,
                                                   progress=progress)
            progress(1, desc="Completed!")

            subtitle = self.generate_and_write_subtitle(
                file_name="Mic",
                transcribed_segments=result,
                add_timestamp=True,
                subformat=subformat
            )

            return f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            print(f"Error transcribing mic: {str(e)}")
            return f"Error transcribing mic: {str(e)}"
        finally:
            self.release_cuda_memory()
            self.remove_input_files([micaudio])

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
                   lang: str,
                   istranslate: bool,
                   progress: gr.Progress
                   ) -> Tuple[list[dict], float]:
        """
        transcribe method for OpenAI's Whisper implementation.

        Parameters
        ----------
        audio: Union[str, BinaryIO, torch.Tensor]
            Audio path or file binary or Audio numpy array
        lang: str
            Source language of the file to transcribe from gr.Dropdown()
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        segments_result: list[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()

        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        if lang == "Automatic Detection":
            lang = None

        translatable_model = ["large", "large-v1", "large-v2"]
        segments_result = self.model.transcribe(audio=audio,
                                                language=lang,
                                                verbose=False,
                                                task="translate" if istranslate and self.current_model_size in translatable_model else "transcribe",
                                                progress_callback=progress_callback)["segments"]
        elapsed_time = time.time() - start_time

        return segments_result, elapsed_time

    def initialize_model(self,
                         model_size: str,
                         progress: gr.Progress
                         ):
        """
        Initialize model if it doesn't match with current model size
        """
        progress(0, desc="Initializing Model..")
        self.current_model_size = model_size
        self.model = whisper.load_model(name=model_size, download_root=os.path.join("models", "Whisper"))

    @staticmethod
    def generate_and_write_subtitle(file_name: str,
                                    transcribed_segments: list,
                                    add_timestamp: bool,
                                    subformat: str,
                                    ) -> str:
        """
        This method writes subtitle file and returns str to gr.Textbox
        """
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        if add_timestamp:
            output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
        else:
            output_path = os.path.join("outputs", f"{file_name}")

        if subformat == "SRT":
            subtitle = get_srt(transcribed_segments)
            write_file(subtitle, f"{output_path}.srt")
        elif subformat == "WebVTT":
            subtitle = get_vtt(transcribed_segments)
            write_file(subtitle, f"{output_path}.vtt")
        return subtitle

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)

        time_str = ""
        if hours:
            time_str += f"{hours} hours "
        if minutes:
            time_str += f"{minutes} minutes "
        seconds = round(seconds)
        time_str += f"{seconds} seconds"

        return time_str.strip()
