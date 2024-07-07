import os
import torch
import whisper
import gradio as gr
from abc import ABC, abstractmethod
from typing import BinaryIO, Union, Tuple, List
import numpy as np
from datetime import datetime
from argparse import Namespace
from faster_whisper.vad import VadOptions
from dataclasses import astuple

from modules.utils.subtitle_manager import get_srt, get_vtt, get_txt, write_file, safe_filename
from modules.utils.youtube_manager import get_ytdata, get_ytaudio
from modules.utils.files_manager import get_media_files, format_gradio_files
from modules.whisper.whisper_parameter import *
from modules.diarize.diarizer import Diarizer
from modules.vad.silero_vad import SileroVAD


class WhisperBase(ABC):
    def __init__(self,
                 model_dir: str,
                 output_dir: str,
                 args: Namespace
                 ):
        self.model = None
        self.current_model_size = None
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.translatable_models = ["large", "large-v1", "large-v2", "large-v3"]
        self.device = self.get_device()
        self.available_compute_types = ["float16", "float32"]
        self.current_compute_type = "float16" if self.device == "cuda" else "float32"
        self.diarizer = Diarizer(
            model_dir=args.diarization_model_dir
        )
        self.vad = SileroVAD()

    @abstractmethod
    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress,
                   *whisper_params,
                   ):
        pass

    @abstractmethod
    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress
                     ):
        pass

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress,
            *whisper_params,
            ) -> Tuple[List[dict], float]:
        """
        Run transcription with conditional pre-processing and post-processing.
        The VAD will be performed to remove noise from the audio input in pre-processing, if enabled.
        The diarization will be performed in post-processing, if enabled.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *whisper_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        segments_result: List[dict]
            list of dicts that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        params = WhisperParameters.as_value(*whisper_params)

        if params.vad_filter:
            vad_options = VadOptions(
                threshold=params.threshold,
                min_speech_duration_ms=params.min_speech_duration_ms,
                max_speech_duration_s=params.max_speech_duration_s,
                min_silence_duration_ms=params.min_silence_duration_ms,
                speech_pad_ms=params.speech_pad_ms
            )
            self.vad.run(
                audio=audio,
                vad_parameters=vad_options,
                progress=progress
            )

        if params.lang == "Automatic Detection":
            params.lang = None
        else:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            params.lang = language_code_dict[params.lang]

        result, elapsed_time = self.transcribe(
            audio,
            progress,
            *astuple(params)
        )

        if params.is_diarize:
            result, elapsed_time_diarization = self.diarizer.run(
                audio=audio,
                use_auth_token=params.hf_token,
                transcribed_result=result,
                device=self.device
            )
            elapsed_time += elapsed_time_diarization
        return result, elapsed_time

    def transcribe_file(self,
                        files: list,
                        input_folder_path: str,
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
        input_folder_path: str
            Input folder path to transcribe from gr.Textbox(). If this is provided, `files` will be ignored and
            this will be used instead.
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the subtitle filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *whisper_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            if input_folder_path:
                files = get_media_files(input_folder_path)
                files = format_gradio_files(files)

            files_info = {}
            for file in files:
                transcribed_segments, time_for_task = self.run(
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
                    file_format=file_format,
                    output_dir=self.output_dir
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
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.run(
                mic_audio,
                progress,
                *whisper_params,
            )
            progress(1, desc="Completed!")

            subtitle, result_file_path = self.generate_and_write_file(
                file_name="Mic",
                transcribed_segments=transcribed_segments,
                add_timestamp=True,
                file_format=file_format,
                output_dir=self.output_dir
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return [result_str, result_file_path]
        except Exception as e:
            print(f"Error transcribing file: {e}")
        finally:
            self.release_cuda_memory()
            self.remove_input_files([mic_audio])

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
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

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

            transcribed_segments, time_for_task = self.run(
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
                file_format=file_format,
                output_dir=self.output_dir
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

    @staticmethod
    def generate_and_write_file(file_name: str,
                                transcribed_segments: list,
                                add_timestamp: bool,
                                file_format: str,
                                output_dir: str
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
        output_dir: str
            Directory path of the output

        Returns
        ----------
        content: str
            Result of the transcription
        output_path: str
            output file path
        """
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        if add_timestamp:
            output_path = os.path.join(output_dir, f"{file_name}-{timestamp}")
        else:
            output_path = os.path.join(output_dir, f"{file_name}")

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

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def release_cuda_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
