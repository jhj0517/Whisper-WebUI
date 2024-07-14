from faster_whisper.vad import VadOptions, get_vad_model
import numpy as np
from typing import BinaryIO, Union, List, Optional, Tuple
import warnings
import faster_whisper
import gradio as gr


class SileroVAD:
    def __init__(self):
        self.sampling_rate = 16000
        self.window_size_samples = 512
        self.model = None

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            vad_parameters: VadOptions,
            silence_non_speech: bool = True,
            progress: gr.Progress = gr.Progress()):
        """
        Run VAD

        Parameters
        ----------
        audio: Union[str, BinaryIO, np.ndarray]
            Audio path or file binary or Audio numpy array
        vad_parameters:
            Options for VAD processing.
        silence_non_speech: bool
            If True, non-speech parts will be silenced instead of being removed.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        audio: np.ndarray
            Pre-processed audio with VAD
        """

        sampling_rate = self.sampling_rate

        if not isinstance(audio, np.ndarray):
            audio = faster_whisper.decode_audio(audio, sampling_rate=sampling_rate)

        duration = audio.shape[0] / sampling_rate

        if vad_parameters is None:
            vad_parameters = VadOptions()
        elif isinstance(vad_parameters, dict):
            vad_parameters = VadOptions(**vad_parameters)

        speech_chunks = self.get_speech_timestamps(
            audio=audio,
            vad_options=vad_parameters,
            progress=progress
        )

        audio, duration_diff = self.collect_chunks(
            audio=audio,
            chunks=speech_chunks,
            silence_non_speech=silence_non_speech
        )

        if silence_non_speech:
            print(
                f"VAD filter silenced {self.format_timestamp(duration_diff)} of audio.",
            )
        else:
            print(
                f"VAD filter removed {self.format_timestamp(duration_diff)} of audio",
            )

        return audio

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
        min_speech_duration_ms = vad_options.min_speech_duration_ms
        max_speech_duration_s = vad_options.max_speech_duration_s
        min_silence_duration_ms = vad_options.min_silence_duration_ms
        window_size_samples = self.window_size_samples
        speech_pad_ms = vad_options.speech_pad_ms
        sampling_rate = 16000
        min_speech_samples = sampling_rate * min_speech_duration_ms / 1000
        speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        max_speech_samples = (
                sampling_rate * max_speech_duration_s
                - window_size_samples
                - 2 * speech_pad_samples
        )
        min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        min_silence_samples_at_max_speech = sampling_rate * 98 / 1000

        audio_length_samples = len(audio)

        state, context = self.model.get_initial_states(batch_size=1)

        speech_probs = []
        for current_start_sample in range(0, audio_length_samples, window_size_samples):
            progress(current_start_sample/audio_length_samples, desc="Detecting speeches only using VAD...")

            chunk = audio[current_start_sample: current_start_sample + window_size_samples]
            if len(chunk) < window_size_samples:
                chunk = np.pad(chunk, (0, int(window_size_samples - len(chunk))))
            speech_prob, state, context = self.model(chunk, state, context, sampling_rate)
            speech_probs.append(speech_prob)

        triggered = False
        speeches = []
        current_speech = {}
        neg_threshold = threshold - 0.15

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

    def collect_chunks(
        self,
        audio: np.ndarray,
        chunks: List[dict],
        silence_non_speech: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Collects and concatenate audio chunks.

        Args:
          audio: One dimensional float array.
          chunks: List of dictionaries containing start and end samples of speech chunks
          silence_non_speech: If True, non-speech parts will be silenced instead of being removed.

        Returns:
          Tuple containing:
            - Processed audio as a numpy array
            - Duration of non-speech (silenced or removed) audio in seconds
        """
        if not chunks:
            return np.array([], dtype=np.float32), 0.0

        total_samples = audio.shape[0]
        speech_samples_count = sum(chunk["end"] - chunk["start"] for chunk in chunks)
        non_speech_samples_count = total_samples - speech_samples_count
        non_speech_duration = non_speech_samples_count / self.sampling_rate

        if not silence_non_speech:
            processed_audio = np.concatenate([audio[chunk["start"]: chunk["end"]] for chunk in chunks])
        else:
            processed_audio = np.zeros_like(audio)
            for chunk in chunks:
                start, end = chunk['start'], chunk['end']
                processed_audio[start:end] = audio[start:end]

        return processed_audio, non_speech_duration

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
