from abc import ABC, abstractmethod
from typing import BinaryIO, Union, List, Optional, Tuple
import numpy as np
import gradio as gr
from faster_whisper.vad import VadOptions

from modules.whisper.data_classes import Segment


class BaseSileroVAD(ABC):
    def __init__(self):
        self.sampling_rate = 16000
        self.window_size_samples = 512
        self.model = None

    @abstractmethod
    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            vad_parameters: dict,
            progress: gr.Progress = gr.Progress()
            ) -> Tuple[np.ndarray, List[dict]]:
        """
        Run VAD

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        vad_parameters: dict
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
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def update_model(self):
        """Update or initialize the VAD model."""
        raise NotImplementedError("Not implemented")

    @abstractmethod
    def restore_speech_timestamps(
        self,
        segments: List[Segment],
        speech_chunks: List[dict],
        sampling_rate: Optional[int] = None,
    ) -> List[Segment]:
        """
        Restore speech timestamps for segments based on speech chunks.

        Parameters
        ----------
        segments: List[Segment]
            List of segments with timestamps
        speech_chunks: List[dict]
            List of speech chunks from VAD
        sampling_rate: Optional[int]
            Sampling rate of the audio

        Returns
        ----------
        List[Segment]
            Segments with restored timestamps
        """
        raise NotImplementedError("Not implemented")

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

    @staticmethod
    def format_timestamp(
        seconds: float,
        always_include_hours: bool = False,
        decimal_marker: str = ".",
    ) -> str:
        """Format timestamp in HH:MM:SS.mmm format."""
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
