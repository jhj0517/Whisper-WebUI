"""
This is VAD implementation from faster-whisper.
Adapted from https://github.com/SYSTRAN/faster-whisper/blob/master/faster_whisper/vad.py

This strategy cutoff non-speech parts of the audio to preprocess the audio before transcribing it.
When it restores the timestamps, it assumes the VAD detected same amount of speeches as transcriber (whisper).
"""

from faster_whisper.vad import VadOptions, get_vad_model
import numpy as np
from typing import BinaryIO, Union, List, Optional, Tuple
import warnings
import bisect
import faster_whisper
from faster_whisper.transcribe import SpeechTimestampsMap
import gradio as gr

from modules.whisper.data_classes import *
from modules.vad.base_silero_vad import BaseSileroVAD


class FasterWhisperSileroVAD(BaseSileroVAD):
    def __init__(self):
        super().__init__()
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

    def update_model(self):
        self.model = get_vad_model()

    @staticmethod
    def collect_chunks(audio: np.ndarray, chunks: List[dict]) -> np.ndarray:
        """Collects and concatenates audio chunks."""
        if not chunks:
            return np.array([], dtype=np.float32)

        return np.concatenate([audio[chunk["start"]: chunk["end"]] for chunk in chunks])

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

