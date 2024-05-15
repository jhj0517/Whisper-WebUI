import os

import time
import numpy as np
from typing import BinaryIO, Union, Tuple, List
from datetime import datetime

import faster_whisper
import ctranslate2
import whisper
import torch
import gradio as gr

from .base_interface import BaseInterface
from modules.subtitle_manager import get_srt, get_vtt, get_txt, write_file, safe_filename
from modules.youtube_manager import get_ytdata, get_ytaudio
from modules.whisper_parameter import *

# Temporal fix of the issue : https://github.com/jhj0517/Whisper-WebUI/issues/144
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class FasterWhisperInference(BaseInterface):
    def __init__(self):
        super().__init__()
        self.current_model_size = None
        self.model = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.translatable_models = ["large", "large-v1", "large-v2", "large-v3"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_compute_types = ctranslate2.get_supported_compute_types(
            "cuda") if self.device == "cuda" else ctranslate2.get_supported_compute_types("cpu")
        self.current_compute_type = "float16" if self.device == "cuda" else "float32"
        self.default_beam_size = 1

    def transcribe_file(self,
                        files: list,
                        file_format: str,
                        add_timestamp: bool,
                        progress=gr.Progress(),
                        *whisper_params,
                        ) -> list:
        """
        Write subtitle file from Files

        Parameters
        ----------
        files: list
            List of files to transcribe from gr.Files()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the subtitle filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *whisper_params: tuple
            Gradio components related to Whisper. see whisper_data_class.py for details.

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            files_info = {}
            for file in files:
                transcribed_segments, time_for_task = self.transcribe(
                    file.name,
                    progress,
                    *whisper_params,
                )

                file_name, file_ext = os.path.splitext(os.path.basename(file.name))
                file_name = safe_filename(file_name)
                subtitle, file_path = self.generate_and_write_file(
                    file_name=file_name,
                    transcribed_segments=transcribed_segments,
                    add_timestamp=add_timestamp,
                    file_format=file_format
                )
                files_info[file_name] = {"subtitle": subtitle, "time_for_task": time_for_task, "path": file_path}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
                total_time += info["time_for_task"]

            result_str = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            result_file_path = [info['path'] for info in files_info.values()]

            return [result_str, result_file_path]

        except Exception as e:
            print(f"Error transcribing file: {e}")
        finally:
            self.release_cuda_memory()
            if not files:
                self.remove_input_files([file.name for file in files])

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: str,
                           add_timestamp: bool,
                           progress=gr.Progress(),
                           *whisper_params,
                           ) -> list:
        """
        Write subtitle file from Youtube

        Parameters
        ----------
        youtube_link: str
            URL of the Youtube video to transcribe from gr.Textbox()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *whisper_params: tuple
            Gradio components related to Whisper. see whisper_data_class.py for details.

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtube_link)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.transcribe(
                audio,
                progress,
                *whisper_params,
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle, result_file_path = self.generate_and_write_file(
                file_name=file_name,
                transcribed_segments=transcribed_segments,
                add_timestamp=add_timestamp,
                file_format=file_format
            )
            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"

            return [result_str, result_file_path]

        except Exception as e:
            print(f"Error transcribing file: {e}")
        finally:
            try:
                if 'yt' not in locals():
                    yt = get_ytdata(youtube_link)
                    file_path = get_ytaudio(yt)
                else:
                    file_path = get_ytaudio(yt)

                self.release_cuda_memory()
                self.remove_input_files([file_path])
            except Exception as cleanup_error:
                pass

    def transcribe_mic(self,
                       mic_audio: str,
                       file_format: str,
                       progress=gr.Progress(),
                       *whisper_params,
                       ) -> list:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        mic_audio: str
            Audio file path from gr.Microphone()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *whisper_params: tuple
            Gradio components related to Whisper. see whisper_data_class.py for details.

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.transcribe(
                mic_audio,
                progress,
                *whisper_params,
            )
            progress(1, desc="Completed!")

            subtitle, result_file_path = self.generate_and_write_file(
                file_name="Mic",
                transcribed_segments=transcribed_segments,
                add_timestamp=True,
                file_format=file_format
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return [result_str, result_file_path]
        except Exception as e:
            print(f"Error transcribing file: {e}")
        finally:
            self.release_cuda_memory()
            self.remove_input_files([mic_audio])

    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress,
                   *whisper_params,
                   ) -> Tuple[List[dict], float]:
        """
        transcribe method for faster-whisper.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *whisper_params: tuple
            Gradio components related to Whisper. see whisper_data_class.py for details.

        Returns
        ----------
        segments_result: List[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for transcription
        """
        start_time = time.time()

        params = WhisperValues(*whisper_params)

        if params.model_size != self.current_model_size or self.model is None or self.current_compute_type != params.compute_type:
            self.update_model(params.model_size, params.compute_type, progress)

        if params.lang == "Automatic Detection":
            params.lang = None
        else:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            params.lang = language_code_dict[params.lang]

        segments, info = self.model.transcribe(
            audio=audio,
            language=params.lang,
            task="translate" if params.is_translate and self.current_model_size in self.translatable_models else "transcribe",
            beam_size=params.beam_size,
            log_prob_threshold=params.log_prob_threshold,
            no_speech_threshold=params.no_speech_threshold,
            best_of=params.best_of,
            patience=params.patience
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

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress
                     ):
        """
        Update current model setting

        Parameters
        ----------
        model_size: str
            Size of whisper model
        compute_type: str
            Compute type for transcription.
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        """
        progress(0, desc="Initializing Model..")
        self.current_model_size = model_size
        self.current_compute_type = compute_type
        self.model = faster_whisper.WhisperModel(
            device=self.device,
            model_size_or_path=model_size,
            download_root=os.path.join("models", "Whisper", "faster-whisper"),
            compute_type=self.current_compute_type
        )

    @staticmethod
    def generate_and_write_file(file_name: str,
                                transcribed_segments: list,
                                add_timestamp: bool,
                                file_format: str,
                                ) -> str:
        """
        Writes subtitle file

        Parameters
        ----------
        file_name: str
            Output file name
        transcribed_segments: list
            Text segments transcribed from audio
        add_timestamp: bool
            Determines whether to add a timestamp to the end of the filename.
        file_format: str
            File format to write. Supported formats: [SRT, WebVTT, txt]

        Returns
        ----------
        content: str
            Result of the transcription
        output_path: str
            output file path
        """
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        if add_timestamp:
            output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
        else:
            output_path = os.path.join("outputs", f"{file_name}")

        if file_format == "SRT":
            content = get_srt(transcribed_segments)
            output_path += '.srt'
            write_file(content, output_path)

        elif file_format == "WebVTT":
            content = get_vtt(transcribed_segments)
            output_path += '.vtt'
            write_file(content, output_path)

        elif file_format == "txt":
            content = get_txt(transcribed_segments)
            output_path += '.txt'
            write_file(content, output_path)
        return content, output_path

    @staticmethod
    def format_time(elapsed_time: float) -> str:
        """
        Get {hours} {minutes} {seconds} time format string

        Parameters
        ----------
        elapsed_time: str
            Elapsed time for transcription

        Returns
        ----------
        Time format string
        """
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
