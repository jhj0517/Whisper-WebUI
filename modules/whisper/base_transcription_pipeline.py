import os
import whisper
import ctranslate2
import gradio as gr
import torchaudio
from abc import ABC, abstractmethod
from typing import BinaryIO, Union, Tuple, List
import numpy as np
from datetime import datetime
from faster_whisper.vad import VadOptions
import gc

from modules.uvr.music_separator import MusicSeparator
from modules.utils.paths import (WHISPER_MODELS_DIR, DIARIZATION_MODELS_DIR, OUTPUT_DIR, DEFAULT_PARAMETERS_CONFIG_PATH,
                                 UVR_MODELS_DIR)
from modules.utils.constants import *
from modules.utils.subtitle_manager import *
from modules.utils.youtube_manager import get_ytdata, get_ytaudio
from modules.utils.files_manager import get_media_files, format_gradio_files, load_yaml, save_yaml, read_file
from modules.whisper.data_classes import *
from modules.diarize.diarizer import Diarizer
from modules.vad.silero_vad import SileroVAD


class BaseTranscriptionPipeline(ABC):
    def __init__(self,
                 model_dir: str = WHISPER_MODELS_DIR,
                 diarization_model_dir: str = DIARIZATION_MODELS_DIR,
                 uvr_model_dir: str = UVR_MODELS_DIR,
                 output_dir: str = OUTPUT_DIR,
                 ):
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.model_dir, exist_ok=True)
        self.diarizer = Diarizer(
            model_dir=diarization_model_dir
        )
        self.vad = SileroVAD()
        self.music_separator = MusicSeparator(
            model_dir=uvr_model_dir,
            output_dir=os.path.join(output_dir, "UVR")
        )

        self.model = None
        self.current_model_size = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.translatable_models = ["large", "large-v1", "large-v2", "large-v3"]
        self.device = self.get_device()
        self.available_compute_types = self.get_available_compute_type()
        self.current_compute_type = self.get_compute_type()

    @abstractmethod
    def transcribe(self,
                   audio: Union[str, BinaryIO, np.ndarray],
                   progress: gr.Progress = gr.Progress(),
                   *whisper_params,
                   ):
        """Inference whisper model to transcribe"""
        pass

    @abstractmethod
    def update_model(self,
                     model_size: str,
                     compute_type: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        """Initialize whisper model"""
        pass

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            progress: gr.Progress = gr.Progress(),
            file_format: str = "SRT",
            add_timestamp: bool = True,
            *pipeline_params,
            ) -> Tuple[List[Segment], float]:
        """
        Run transcription with conditional pre-processing and post-processing.
        The VAD will be performed to remove noise from the audio input in pre-processing, if enabled.
        The diarization will be performed in post-processing, if enabled.
        Due to the integration with gradio, the parameters have to be specified with a `*` wildcard.

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio input. This can be file path or binary type.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        file_format: str
            Subtitle file format between ["SRT", "WebVTT", "txt", "lrc"]
        add_timestamp: bool
            Whether to add a timestamp at the end of the filename.
        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class.
            This must be provided as a List with * wildcard because of the integration with gradio.
            See more info at : https://github.com/gradio-app/gradio/issues/2471

        Returns
        ----------
        segments_result: List[Segment]
            list of Segment that includes start, end timestamps and transcribed text
        elapsed_time: float
            elapsed time for running
        """
        params = TranscriptionPipelineParams.from_list(list(pipeline_params))
        params = self.validate_gradio_values(params)
        bgm_params, vad_params, whisper_params, diarization_params = params.bgm_separation, params.vad, params.whisper, params.diarization

        if bgm_params.is_separate_bgm:
            music, audio, _ = self.music_separator.separate(
                audio=audio,
                model_name=bgm_params.model_size,
                device=bgm_params.device,
                segment_size=bgm_params.segment_size,
                save_file=bgm_params.save_file,
                progress=progress
            )

            if audio.ndim >= 2:
                audio = audio.mean(axis=1)
                if self.music_separator.audio_info is None:
                    origin_sample_rate = 16000
                else:
                    origin_sample_rate = self.music_separator.audio_info.sample_rate
                audio = self.resample_audio(audio=audio, original_sample_rate=origin_sample_rate)

            if bgm_params.enable_offload:
                self.music_separator.offload()

        if vad_params.vad_filter:
            vad_options = VadOptions(
                threshold=vad_params.threshold,
                min_speech_duration_ms=vad_params.min_speech_duration_ms,
                max_speech_duration_s=vad_params.max_speech_duration_s,
                min_silence_duration_ms=vad_params.min_silence_duration_ms,
                speech_pad_ms=vad_params.speech_pad_ms
            )

            vad_processed, speech_chunks = self.vad.run(
                audio=audio,
                vad_parameters=vad_options,
                progress=progress
            )

            if vad_processed.size > 0:
                audio = vad_processed
            else:
                vad_params.vad_filter = False

        result, elapsed_time = self.transcribe(
            audio,
            progress,
            *whisper_params.to_list()
        )

        if vad_params.vad_filter:
            result = self.vad.restore_speech_timestamps(
                segments=result,
                speech_chunks=speech_chunks,
            )

        if diarization_params.is_diarize:
            result, elapsed_time_diarization = self.diarizer.run(
                audio=audio,
                use_auth_token=diarization_params.hf_token,
                transcribed_result=result,
                device=diarization_params.device
            )
            elapsed_time += elapsed_time_diarization

        self.cache_parameters(
            params=params,
            file_format=file_format,
            add_timestamp=add_timestamp
        )
        return result, elapsed_time

    def transcribe_file(self,
                        files: Optional[List] = None,
                        input_folder_path: Optional[str] = None,
                        file_format: str = "SRT",
                        add_timestamp: bool = True,
                        progress=gr.Progress(),
                        *pipeline_params,
                        ) -> Tuple[str, List]:
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
        *pipeline_params: tuple
            Parameters for the transcription pipeline. This will be dealt with "TranscriptionPipelineParams" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            if input_folder_path:
                files = get_media_files(input_folder_path)
            if isinstance(files, str):
                files = [files]
            if files and isinstance(files[0], gr.utils.NamedString):
                files = [file.name for file in files]

            files_info = {}
            for file in files:
                transcribed_segments, time_for_task = self.run(
                    file,
                    progress,
                    file_format,
                    add_timestamp,
                    *pipeline_params,
                )

                file_name, file_ext = os.path.splitext(os.path.basename(file))
                subtitle, file_path = generate_file(
                    output_dir=self.output_dir,
                    output_file_name=file_name,
                    output_format=file_format,
                    result=transcribed_segments,
                    add_timestamp=add_timestamp,
                    **writer_options
                )
                files_info[file_name] = {"subtitle": read_file(file_path), "time_for_task": time_for_task, "path": file_path}

            total_result = ''
            total_time = 0
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
                total_time += info["time_for_task"]

            result_str = f"Done in {self.format_time(total_time)}! Subtitle is in the outputs folder.\n\n{total_result}"
            result_file_path = [info['path'] for info in files_info.values()]

            return result_str, result_file_path

        except Exception as e:
            print(f"Error transcribing file: {e}")
            raise
        finally:
            self.release_cuda_memory()

    def transcribe_mic(self,
                       mic_audio: str,
                       file_format: str = "SRT",
                       add_timestamp: bool = True,
                       progress=gr.Progress(),
                       *pipeline_params,
                       ) -> Tuple[str, str]:
        """
        Write subtitle file from microphone

        Parameters
        ----------
        mic_audio: str
            Audio file path from gr.Microphone()
        file_format: str
            Subtitle File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio..")
            transcribed_segments, time_for_task = self.run(
                mic_audio,
                progress,
                file_format,
                add_timestamp,
                *pipeline_params,
            )
            progress(1, desc="Completed!")

            file_name = "Mic"
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
            return result_str, file_path
        except Exception as e:
            print(f"Error transcribing mic: {e}")
            raise
        finally:
            self.release_cuda_memory()

    def transcribe_youtube(self,
                           youtube_link: str,
                           file_format: str = "SRT",
                           add_timestamp: bool = True,
                           progress=gr.Progress(),
                           *pipeline_params,
                           ) -> Tuple[str, str]:
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
        *pipeline_params: tuple
            Parameters related with whisper. This will be dealt with "WhisperParameters" data class

        Returns
        ----------
        result_str:
            Result of transcription to return to gr.Textbox()
        result_file_path:
            Output file path to return to gr.Files()
        """
        try:
            params = TranscriptionPipelineParams.from_list(list(pipeline_params))
            writer_options = {
                "highlight_words": True if params.whisper.word_timestamps else False
            }

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtube_link)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.run(
                audio,
                progress,
                file_format,
                add_timestamp,
                *pipeline_params,
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle, file_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_format,
                result=transcribed_segments,
                add_timestamp=add_timestamp,
                **writer_options
            )

            result_str = f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"

            if os.path.exists(audio):
                os.remove(audio)

            return result_str, file_path

        except Exception as e:
            print(f"Error transcribing youtube: {e}")
            raise
        finally:
            self.release_cuda_memory()

    def get_compute_type(self):
        if "float16" in self.available_compute_types:
            return "float16"
        if "float32" in self.available_compute_types:
            return "float32"
        else:
            return self.available_compute_types[0]

    def get_available_compute_type(self):
        if self.device == "cuda":
            return list(ctranslate2.get_supported_compute_types("cuda"))
        else:
            return list(ctranslate2.get_supported_compute_types("cpu"))

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            self.release_cuda_memory()
        gc.collect()

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
            if not BaseTranscriptionPipeline.is_sparse_api_supported():
                # Device `SparseMPS` is not supported for now. See : https://github.com/pytorch/pytorch/issues/87886
                return "cpu"
            return "mps"
        else:
            return "cpu"

    @staticmethod
    def is_sparse_api_supported():
        if not torch.backends.mps.is_available():
            return False

        try:
            device = torch.device("mps")
            sparse_tensor = torch.sparse_coo_tensor(
                indices=torch.tensor([[0, 1], [2, 3]]),
                values=torch.tensor([1, 2]),
                size=(4, 4),
                device=device
            )
            return True
        except RuntimeError:
            return False

    @staticmethod
    def release_cuda_memory():
        """Release memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()

    @staticmethod
    def remove_input_files(file_paths: List[str]):
        """Remove gradio cached files"""
        if not file_paths:
            return

        for file_path in file_paths:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)

    @staticmethod
    def validate_gradio_values(params: TranscriptionPipelineParams):
        """
        Validate gradio specific values that can't be displayed as None in the UI.
        Related issue : https://github.com/gradio-app/gradio/issues/8723
        """
        if params.whisper.lang is None:
            pass
        elif params.whisper.lang == AUTOMATIC_DETECTION:
            params.whisper.lang = None
        else:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            params.whisper.lang = language_code_dict[params.whisper.lang]

        if params.whisper.initial_prompt == GRADIO_NONE_STR:
            params.whisper.initial_prompt = None
        if params.whisper.prefix == GRADIO_NONE_STR:
            params.whisper.prefix = None
        if params.whisper.hotwords == GRADIO_NONE_STR:
            params.whisper.hotwords = None
        if params.whisper.max_new_tokens == GRADIO_NONE_NUMBER_MIN:
            params.whisper.max_new_tokens = None
        if params.whisper.hallucination_silence_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.hallucination_silence_threshold = None
        if params.whisper.language_detection_threshold == GRADIO_NONE_NUMBER_MIN:
            params.whisper.language_detection_threshold = None
        if params.vad.max_speech_duration_s == GRADIO_NONE_NUMBER_MAX:
            params.vad.max_speech_duration_s = float('inf')
        return params

    @staticmethod
    def cache_parameters(
        params: TranscriptionPipelineParams,
        file_format: str = "SRT",
        add_timestamp: bool = True
    ):
        """Cache parameters to the yaml file"""
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        param_to_cache = params.to_dict()

        cached_yaml = {**cached_params, **param_to_cache}
        cached_yaml["whisper"]["add_timestamp"] = add_timestamp
        cached_yaml["whisper"]["file_format"] = file_format

        supress_token = cached_yaml["whisper"].get("suppress_tokens", None)
        if supress_token and isinstance(supress_token, list):
            cached_yaml["whisper"]["suppress_tokens"] = str(supress_token)

        if cached_yaml["whisper"].get("lang", None) is None:
            cached_yaml["whisper"]["lang"] = AUTOMATIC_DETECTION.unwrap()
        else:
            language_dict = whisper.tokenizer.LANGUAGES
            cached_yaml["whisper"]["lang"] = language_dict[cached_yaml["whisper"]["lang"]]

        if cached_yaml["vad"].get("max_speech_duration_s", float('inf')) == float('inf'):
            cached_yaml["vad"]["max_speech_duration_s"] = GRADIO_NONE_NUMBER_MAX

        if cached_yaml is not None and cached_yaml:
            save_yaml(cached_yaml, DEFAULT_PARAMETERS_CONFIG_PATH)

    @staticmethod
    def resample_audio(audio: Union[str, np.ndarray],
                       new_sample_rate: int = 16000,
                       original_sample_rate: Optional[int] = None,) -> np.ndarray:
        """Resamples audio to 16k sample rate, standard on Whisper model"""
        if isinstance(audio, str):
            audio, original_sample_rate = torchaudio.load(audio)
        else:
            if original_sample_rate is None:
                raise ValueError("original_sample_rate must be provided when audio is numpy array.")
            audio = torch.from_numpy(audio)
        resampler = torchaudio.transforms.Resample(orig_freq=original_sample_rate, new_freq=new_sample_rate)
        resampled_audio = resampler(audio).numpy()
        return resampled_audio
