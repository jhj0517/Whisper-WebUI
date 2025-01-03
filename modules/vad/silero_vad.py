# Adapted from https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py

from faster_whisper.vad import VadOptions, get_vad_model
import numpy as np
from typing import BinaryIO, Union, List, Optional, Tuple
import warnings
import bisect
import faster_whisper
from faster_whisper.transcribe import SpeechTimestampsMap
import gradio as gr

from modules.whisper.data_classes import *


class SileroVAD:
    def __init__(self):
        self.sampling_rate = 16000
        self.window_size_samples = 512
        self.model = None

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            vad_parameters: VadOptions,
            progress: gr.Progress = gr.Progress()
            ) -> Tuple[np.ndarray, List[dict]]:
        """
        Run VAD

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        vad_parameters:
            Options for VAD processing.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        np.ndarray
            Pre-processed audio with VAD
        List[dict]
            Chunks of speeches to be used to restore the timestamps later
        """

        sampling_rate = self.sampling_rate

        if not isinstance(audio, np.ndarray):
            audio = faster_whisper.decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate
        duration_after_vad = duration

        if vad_parameters is None:
            vad_parameters = VadOptions()
        elif isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)
        speech_chunks = self.get_speech_timestamps(
            audio=audio,
            vad_options=vad_parameters,
            progress=progress
        )

        audio = self.collect_chunks(audio, speech_chunks)
        duration_after_vad = audio.shape[0] / sampling_rate

        return audio, speech_chunks

    def get_speech_timestamps(
        self,
        audio: np.ndarray,
        vad_options: Optional[VadOptions] = None,
        progress: gr.Progress = gr.Progress(),
        **kwargs,
    ) -> List[dict]:
        """This method is used for splitting long audios into speech chunks using silero VAD.

        Args:
          audio: One dimensional float array.
          vad_options: Options for VAD processing.
          kwargs: VAD options passed as keyword arguments for backward compatibility.
          progress: Gradio progress to indicate progress.

        Returns:
          List of dicts containing begin and end samples of each speech chunk.
        """

        if self.model is None:
            self.update_model()

        if vad_options is None:
            vad_options = VadOptions(**kwargs)

        threshold = vad_options.threshold
        neg_threshold = vad_options.neg_threshold
        min_speech_duration_ms = vad_options.min_speech_duration_ms
        max_speech_duration_s = vad_options.max_speech_duration_s
        min_silence_duration_ms = vad_options.min_silence_duration_ms
        window_size_samples = self.window_size_samples
        speech_pad_ms = vad_options.speech_pad_ms
        min_speech_samples = self.sampling_rate * min_speech_duration_ms / 1000
        speech_pad_samples = self.sampling_rate * speech_pad_ms / 1000
        max_speech_samples = (
                self.sampling_rate * max_speech_duration_s
                - window_size_samples
                - 2 * speech_pad_samples
        )
        min_silence_samples = self.sampling_rate * min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = self.sampling_rate * 98 / 1000

        audio_length_samples = len(audio)

        padded_audio = np.pad(
            audio, (0, window_size_samples - audio.shape[0] % window_size_samples)
        )
        speech_probs = self.model(padded_audio.reshape(1, -1)).squeeze(0)

        triggered = False
        speeches = []
        current_speech = {}
        if neg_threshold is None:
            neg_threshold = max(threshold - 0.15, 0.01)

        # to save potential segment end (and tolerate some silence)
        temp_end = 0
        # to save potential segment limits in case of maximum segment size reached
        prev_end = next_start = 0

        for i, speech_prob in enumerate(speech_probs):
            if (speech_prob >= threshold) and temp_end:
                temp_end = 0
                if next_start < prev_end:
                    next_start = window_size_samples * i

            if (speech_prob >= threshold) and not triggered:
                triggered = True
                current_speech["start"] = window_size_samples * i
                continue

            if (
                    triggered
                    and (window_size_samples * i) - current_speech["start"] > max_speech_samples
            ):
                if prev_end:
                    current_speech["end"] = prev_end
                    speeches.append(current_speech)
                    current_speech = {}
                    # previously reached silence (< neg_thres) and is still not speech (< thres)
                    if next_start < prev_end:
                        triggered = False
                    else:
                        current_speech["start"] = next_start
                    prev_end = next_start = temp_end = 0
                else:
                    current_speech["end"] = window_size_samples * i
                    speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

            if (speech_prob < neg_threshold) and triggered:
                if not temp_end:
                    temp_end = window_size_samples * i
                # condition to avoid cutting in very short silence
                if (window_size_samples * i) - temp_end > min_silence_samples_at_max_speech:
                    prev_end = temp_end
                if (window_size_samples * i) - temp_end < min_silence_samples:
                    continue
                else:
                    current_speech["end"] = temp_end
                    if (
                            current_speech["end"] - current_speech["start"]
                    ) > min_speech_samples:
                        speeches.append(current_speech)
                    current_speech = {}
                    prev_end = next_start = temp_end = 0
                    triggered = False
                    continue

        if (
                current_speech
                and (audio_length_samples - current_speech["start"]) > min_speech_samples
        ):
            current_speech["end"] = audio_length_samples
            speeches.append(current_speech)

        for i, speech in enumerate(speeches):
            if i == 0:
                speech["start"] = int(max(0, speech["start"] - speech_pad_samples))
            if i != len(speeches) - 1:
                silence_duration = speeches[i + 1]["start"] - speech["end"]
                if silence_duration < 2 * speech_pad_samples:
                    speech["end"] += int(silence_duration // 2)
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - silence_duration // 2)
                    )
                else:
                    speech["end"] = int(
                        min(audio_length_samples, speech["end"] + speech_pad_samples)
                    )
                    speeches[i + 1]["start"] = int(
                        max(0, speeches[i + 1]["start"] - speech_pad_samples)
                    )
            else:
                speech["end"] = int(
                    min(audio_length_samples, speech["end"] + speech_pad_samples)
                )

        return speeches

    def update_model(self):
        self.model = get_vad_model()

    @staticmethod
    def collect_chunks(audio: np.ndarray, chunks: List[dict]) -> np.ndarray:
        """Collects and concatenates audio chunks."""
        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate([audio[chunk["start"]: chunk["end"]] for chunk in chunks])

    @staticmethod
    def format_timestamp(
        seconds: float,
        always_include_hours: bool = False,
        decimal_marker: str = ".",
    ) -> str:
        assert seconds >= 0, "non-negative timestamp expected"
        milliseconds = round(seconds * 1000.0)

        hours = milliseconds // 3_600_000
        milliseconds -= hours * 3_600_000

        minutes = milliseconds // 60_000
        milliseconds -= minutes * 60_000

        seconds = milliseconds // 1_000
        milliseconds -= seconds * 1_000

        hours_marker = f"{hours:02d}:" if always_include_hours or hours > 0 else ""
        return (
            f"{hours_marker}{minutes:02d}:{seconds:02d}{decimal_marker}{milliseconds:03d}"
        )

    def restore_speech_timestamps(
        self,
        segments: List[Segment],
        speech_chunks: List[dict],
        sampling_rate: Optional[int] = None,
    ) -> List[Segment]:
        if sampling_rate is None:
            sampling_rate = self.sampling_rate

        ts_map = SpeechTimestampsMap(speech_chunks, sampling_rate)

        for segment in segments:
            if segment.words:
                words = []
                for word in segment.words:
                    # Ensure the word start and end times are resolved to the same chunk.
                    middle = (word.start + word.end) / 2
                    chunk_index = ts_map.get_chunk_index(middle)
                    word.start = ts_map.get_original_time(word.start, chunk_index)
                    word.end = ts_map.get_original_time(word.end, chunk_index)
                    words.append(word)

                segment.start = words[0].start
                segment.end = words[-1].end
                segment.words = words

            else:
                segment.start = ts_map.get_original_time(segment.start)
                segment.end = ts_map.get_original_time(segment.end)

        return segments

