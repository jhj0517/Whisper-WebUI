import os

import tqdm
import time
import numpy as np
from typing import BinaryIO, Union, Tuple
from datetime import datetime, timedelta

import faster_whisper
import whisper
import torch
import gradio as gr

from .base_interface import BaseInterface
from modules.subtitle_manager import get_srt, get_vtt, write_file, safe_filename
from modules.youtube_manager import get_ytdata, get_ytaudio


class FasterWhisperInference(BaseInterface):
    def __init__(self):
        super().__init__()
        self.current_model_size = None
        self.model = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.translatable_models = ["large", "large-v1", "large-v2"]
        self.default_beam_size = 1
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def transcribe_file(self,
                        fileobjs: list,
                        model_size: str,
                        lang: str,
                        subformat: str,
                        istranslate: bool,
                        add_timestamp: bool,
                        progress=gr.Progress()
                        ) -> str:
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

        Returns
        ----------
        String to return to gr.Textbox()
        """
        try:
            if model_size != self.current_model_size or self.model is None:
                self.initialize_model(model_size=model_size, progress=progress)

            if lang == "Automatic Detection":
                lang = None

            files_info = {}
            for fileobj in fileobjs:
                transcribed_segments, time_for_task = self.transcribe(
                    audio=fileobj.name,
                    lang=lang,
                    istranslate=istranslate,
                    progress=progress
                )

                file_name, file_ext = os.path.splitext(os.path.basename(fileobj.orig_name))
                file_name = safe_filename(file_name)
                subtitle = self.generate_and_write_subtitle(
                    file_name=file_name,
                    transcribed_segments=transcribed_segments,
                    add_timestamp=add_timestamp,
                    subformat=subformat
                )
                files_info[file_name] = {"subtitle": subtitle, "time_for_task": time_for_task}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
                total_time += info["time_for_task"]

            return f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"

        except Exception as e:
            print(f"Error transcribing file on line {e}")
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
                           progress=gr.Progress()
                           ) -> str:
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

        Returns
        ----------
        String to return to gr.Textbox()
        """
        try:
            if model_size != self.current_model_size or self.model is None:
                self.initialize_model(model_size=model_size, progress=progress)

            if lang == "Automatic Detection":
                lang = None

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtubelink)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.transcribe(
                audio=audio,
                lang=lang,
                istranslate=istranslate,
                progress=progress
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle = self.generate_and_write_subtitle(
                file_name=file_name,
                transcribed_segments=transcribed_segments,
                add_timestamp=add_timestamp,
                subformat=subformat
            )
            return f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
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
                       progress=gr.Progress()
                       ) -> str:
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

        Returns
        ----------
        String to return to gr.Textbox()
        """
        try:
            if model_size != self.current_model_size or self.model is None:
                self.initialize_model(model_size=model_size, progress=progress)

            if lang == "Automatic Detection":
                lang = None

            progress(0, desc="Loading Audio..")

            transcribed_segments, time_for_task = self.transcribe(
                audio=micaudio,
                lang=lang,
                istranslate=istranslate,
                progress=progress
            )
            progress(1, desc="Completed!")

            subtitle = self.generate_and_write_subtitle(
                file_name="Mic",
                transcribed_segments=transcribed_segments,
                add_timestamp=True,
                subformat=subformat
            )
            return f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            self.release_cuda_memory()
            self.remove_input_files([micaudio])

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   lang: str,
                   istranslate: bool,
                   progress: gr.Progress
                   ) -> Tuple[list, float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
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
        if lang:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            lang = language_code_dict[lang]
        segments, info = self.model.transcribe(
            audio=audio,
            language=lang,
            beam_size=self.default_beam_size,
            task="translate" if istranslate and self.current_model_size in self.translatable_models else "transcribe"
        )
        progress(0, desc="Loading audio..")

        segments_result = []
        for segment in segments:
            progress(segment.start / info.duration, desc="Transcribing..")
            segments_result.append({
                "start": segment.start,
                "end": segment.end,
                "text": segment.text
            })

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
        self.model = faster_whisper.WhisperModel(
            device=self.device,
            model_size_or_path=model_size,
            download_root=os.path.join("models", "Whisper", "faster-whisper"),
            compute_type="float16"
        )

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
