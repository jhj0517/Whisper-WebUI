import whisper
import gradio as gr
import time
import os
from typing import BinaryIO, Union, Tuple, List
import numpy as np
from datetime import datetime
import torch

from .base_interface import BaseInterface
from modules.subtitle_manager import get_srt, get_vtt, get_txt, write_file, safe_filename
from modules.youtube_manager import get_ytdata, get_ytaudio
from modules.whisper_parameter import *

DEFAULT_MODEL_SIZE = "large-v3"


class WhisperInference(BaseInterface):
    def __init__(self):
        super().__init__()
        self.current_model_size = None
        self.model = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.translatable_model = ["large", "large-v1", "large-v2", "large-v3"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_compute_types = ["float16", "float32"]
        self.current_compute_type = "float16" if self.device == "cuda" else "float32"
        self.default_beam_size = 1

    def transcribe_file(self,
                        files: list,
                        file_format: str,
                        add_timestamp: bool,
                        progress=gr.Progress(),
                        *whisper_params
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
                progress(0, desc="Loading Audio..")
                audio = whisper.load_audio(file.name)

                result, elapsed_time = self.transcribe(audio,
                                                       progress,
                                                       *whisper_params)
                progress(1, desc="Completed!")

                file_name, file_ext = os.path.splitext(os.path.basename(file.name))
                file_name = safe_filename(file_name)
                subtitle, file_path = self.generate_and_write_file(
                    file_name=file_name,
                    transcribed_segments=result,
                    add_timestamp=add_timestamp,
                    file_format=file_format
                )
                files_info[file_name] = {"subtitle": subtitle, "elapsed_time": elapsed_time, "path": file_path}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f"{info['subtitle']}"
                total_time += info["elapsed_time"]

            result_str = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            result_file_path = [info['path'] for info in files_info.values()]

            return [result_str, result_file_path]
        except Exception as e:
            print(f"Error transcribing file: {str(e)}")
        finally:
            self.release_cuda_memory()
            self.remove_input_files([file.name for file in files])

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: str,
                           add_timestamp: bool,
                           progress=gr.Progress(),
                           *whisper_params) -> list:
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
            audio = whisper.load_audio(get_ytaudio(yt))

            result, elapsed_time = self.transcribe(audio,
                                                   progress,
                                                   *whisper_params)
            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle, result_file_path = self.generate_and_write_file(
                file_name=file_name,
                transcribed_segments=result,
                add_timestamp=add_timestamp,
                file_format=file_format
            )

            result_str = f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return [result_str, result_file_path]
        except Exception as e:
            print(f"Error transcribing youtube video: {str(e)}")
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
                       *whisper_params) -> list:
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
            result, elapsed_time = self.transcribe(
                mic_audio,
                progress,
                *whisper_params,
            )
            progress(1, desc="Completed!")

            subtitle, result_file_path = self.generate_and_write_file(
                file_name="Mic",
                transcribed_segments=result,
                add_timestamp=True,
                file_format=file_format
            )

            result_str = f"Done in {self.format_time(elapsed_time)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return [result_str, result_file_path]
        except Exception as e:
            print(f"Error transcribing mic: {str(e)}")
        finally:
            self.release_cuda_memory()
            self.remove_input_files([mic_audio])

    def transcribe(self,
                   audio: Union[str, np.ndarray, torch.Tensor],
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

        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        segments_result = self.model.transcribe(audio=audio,
                                                language=params.lang,
                                                verbose=False,
                                                beam_size=params.beam_size,
                                                logprob_threshold=params.log_prob_threshold,
                                                no_speech_threshold=params.no_speech_threshold,
                                                task="translate" if params.is_translate and self.current_model_size in self.translatable_model else "transcribe",
                                                fp16=True if params.compute_type == "float16" else False,
                                                best_of=params.best_of,
                                                patience=params.patience,
                                                temperature=params.temperature,
                                                compression_ratio_threshold=params.compression_ratio_threshold,
                                                progress_callback=progress_callback,)["segments"]
        elapsed_time = time.time() - start_time

        return segments_result, elapsed_time

    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress,
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
        self.current_compute_type = compute_type
        self.current_model_size = model_size
        self.model = whisper.load_model(
            name=model_size,
            device=self.device,
            download_root=os.path.join("models", "Whisper")
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
