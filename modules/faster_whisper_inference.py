import os

import tqdm
import time
import numpy as np
from typing import BinaryIO, Union, Tuple
from datetime import datetime, timedelta

import faster_whisper
import ctranslate2
import whisper
import torch
import gradio as gr

from .base_interface import BaseInterface
from modules.subtitle_manager import get_srt, get_vtt, get_txt, write_file, safe_filename
from modules.youtube_manager import get_ytdata, get_ytaudio


class FasterWhisperInference(BaseInterface):
    def __init__(self):
        super().__init__()
        self.current_model_size = None
        self.model = None
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))
        self.translatable_models = ["large", "large-v1", "large-v2"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.available_compute_types = ctranslate2.get_supported_compute_types("cuda") if self.device == "cuda" else ctranslate2.get_supported_compute_types("cpu")
        self.current_compute_type = "float16" if self.device == "cuda" else "float32"
        self.default_beam_size = 1

    def transcribe_file(self,
                        fileobjs: list,
                        model_size: str,
                        lang: str,
                        file_format: str,
                        istranslate: bool,
                        add_timestamp: bool,
                        beam_size: int,
                        log_prob_threshold: float,
                        no_speech_threshold: float,
                        compute_type: str,
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
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        String to return to gr.Textbox()
        """
        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)

            files_info = {}
            for fileobj in fileobjs:
                transcribed_segments, time_for_task = self.transcribe(
                    audio=fileobj.name,
                    lang=lang,
                    istranslate=istranslate,
                    beam_size=beam_size,
                    log_prob_threshold=log_prob_threshold,
                    no_speech_threshold=no_speech_threshold,
                    progress=progress
                )

                file_name, file_ext = os.path.splitext(os.path.basename(fileobj.orig_name))
                file_name = safe_filename(file_name)
                subtitle = self.generate_and_write_file(
                    file_name=file_name,
                    transcribed_segments=transcribed_segments,
                    add_timestamp=add_timestamp,
                    file_format=file_format
                )
                print(f"{subtitle}")
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
                           file_format: str,
                           istranslate: bool,
                           add_timestamp: bool,
                           beam_size: int,
                           log_prob_threshold: float,
                           no_speech_threshold: float,
                           compute_type: str,
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
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
        compute_type: str
            compute type from gr.Dropdown().
            see more info : https://opennmt.net/CTranslate2/quantization.html
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        String to return to gr.Textbox()
        """
        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)

            progress(0, desc="Loading Audio from Youtube..")
            yt = get_ytdata(youtubelink)
            audio = get_ytaudio(yt)

            transcribed_segments, time_for_task = self.transcribe(
                audio=audio,
                lang=lang,
                istranslate=istranslate,
                beam_size=beam_size,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                progress=progress
            )

            progress(1, desc="Completed!")

            file_name = safe_filename(yt.title)
            subtitle = self.generate_and_write_file(
                file_name=file_name,
                transcribed_segments=transcribed_segments,
                add_timestamp=add_timestamp,
                file_format=file_format
            )
            return f"Done in {self.format_time(time_for_task)}! Subtitle file is in the outputs folder.\n\n{subtitle}"
        except Exception as e:
            return f"Error: {str(e)}"
        finally:
            try:
                if 'yt' not in locals():
                    yt = get_ytdata(youtubelink)
                    file_path = get_ytaudio(yt)
                else:
                    file_path = get_ytaudio(yt)

                self.release_cuda_memory()
                self.remove_input_files([file_path])
            except Exception as cleanup_error:
                pass

    def transcribe_mic(self,
                       micaudio: str,
                       model_size: str,
                       lang: str,
                       file_format: str,
                       istranslate: bool,
                       beam_size: int,
                       log_prob_threshold: float,
                       no_speech_threshold: float,
                       compute_type: str,
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
        file_format: str
            File format to write from gr.Dropdown(). Supported format: [SRT, WebVTT, txt]
        istranslate: bool
            Boolean value from gr.Checkbox() that determines whether to translate to English.
            It's Whisper's feature to translate speech from another language directly into English end-to-end.
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
        compute_type: str
            compute type from gr.Dropdown().
            see more info : https://opennmt.net/CTranslate2/quantization.html
            consider the segment as silent.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        String to return to gr.Textbox()
        """
        try:
            self.update_model_if_needed(model_size=model_size, compute_type=compute_type, progress=progress)

            progress(0, desc="Loading Audio..")

            transcribed_segments, time_for_task = self.transcribe(
                audio=micaudio,
                lang=lang,
                istranslate=istranslate,
                beam_size=beam_size,
                log_prob_threshold=log_prob_threshold,
                no_speech_threshold=no_speech_threshold,
                progress=progress
            )
            progress(1, desc="Completed!")

            subtitle = self.generate_and_write_file(
                file_name="Mic",
                transcribed_segments=transcribed_segments,
                add_timestamp=True,
                file_format=file_format
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
                   beam_size: int,
                   log_prob_threshold: float,
                   no_speech_threshold: float,
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
        beam_size: int
            Int value from gr.Number() that is used for decoding option.
        log_prob_threshold: float
            float value from gr.Number(). If the average log probability over sampled tokens is
            below this value, treat as failed.
        no_speech_threshold: float
            float value from gr.Number(). If the no_speech probability is higher than this value AND
            the average log probability over sampled tokens is below `log_prob_threshold`,
            consider the segment as silent.
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

        if lang == "Automatic Detection":
            lang = None
        else:
            language_code_dict = {value: key for key, value in whisper.tokenizer.LANGUAGES.items()}
            lang = language_code_dict[lang]
        segments, info = self.model.transcribe(
            audio=audio,
            language=lang,
            task="translate" if istranslate and self.current_model_size in self.translatable_models else "transcribe",
            beam_size=beam_size,
            log_prob_threshold=log_prob_threshold,
            no_speech_threshold=no_speech_threshold,
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

    def update_model_if_needed(self,
                               model_size: str,
                               compute_type: str,
                               progress: gr.Progress
                               ):
        """
        Initialize model if it doesn't match with current model setting
        """
        if model_size != self.current_model_size or self.model is None or self.current_compute_type != compute_type:
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
        This method writes subtitle file and returns str to gr.Textbox
        """
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        if add_timestamp:
            output_path = os.path.join("outputs", f"{file_name}-{timestamp}")
        else:
            output_path = os.path.join("outputs", f"{file_name}")

        if file_format == "SRT":
            content = get_srt(transcribed_segments)
            write_file(content, f"{output_path}.srt")

        elif file_format == "WebVTT":
            content = get_vtt(transcribed_segments)
            write_file(content, f"{output_path}.vtt")

        elif file_format == "txt":
            content = get_txt(transcribed_segments)
            write_file(content, f"{output_path}.txt")
        return content

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
