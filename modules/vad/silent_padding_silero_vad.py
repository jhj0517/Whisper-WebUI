from typing import BinaryIO, Union, List, Optional, Tuple
import numpy as np
import gradio as gr
import faster_whisper
from faster_whisper.vad import get_vad_model

from modules.whisper.data_classes import Segment
from modules.vad.base_silero_vad import BaseSileroVAD, VadOptions


class SilentPaddingSileroVAD(BaseSileroVAD):
    def __init__(self):
        super().__init__()
        self.sampling_rate = 16000
        self.window_size_samples = 512
        self.model = None

    def run(self,
            audio: Union[str, BinaryIO, np.ndarray],
            vad_parameters: dict,
            progress: gr.Progress = gr.Progress()
            ) -> Tuple[np.ndarray, List[dict]]:
        """


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

        padded_audio = self.silent_padding_chunks(
            audio,
            speech_chunks,
            padding=vad_parameters.silent_pad_ms,
            sampling_rate=self.sampling_rate
        )

        return padded_audio, speech_chunks

    def update_model(self):
        """Initialize or update your VAD model."""
        self.model = get_vad_model()

    @staticmethod
    def silent_padding_chunks(audio: np.ndarray,
                              chunks: List[dict],
                              padding: int = 1000,
                              sampling_rate: int = 16000) -> np.ndarray:
        """
        Give additional padding to each silent part and concatenates audio chunks.
        It returns longer audio than the original audio because of the silent padding.

        Parameters
        ----------
        audio: np.ndarray
            Original audio that is encoded in numpy array
        chunks: List[dict]
            List of speech chunks from VAD
        padding: int
            Padding to be added to each silent part in milliseconds
        sampling_rate: int
            Sampling rate of the audio

        Returns
        ----------
        np.ndarray
            Audio with silent paddings
        """
        if not chunks:
            return np.array([], dtype=np.float32)

        padded_audio = []
        pad_samples = int(padding * sampling_rate / 1000)

        # Add initial silence from the original audio
        if chunks[0]["start"] > 0:
            padded_audio.append(np.zeros(chunks[0]["start"], dtype=np.float32))

        for i, chunk in enumerate(chunks):
            start, end = chunk["start"], chunk["end"]
            speech = audio[start:end]
            padded_audio.append(speech)

            # Add additional silent padding between speech chunks
            if i < len(chunks) - 1:
                next_start = chunks[i + 1]["start"]
                silence_len = max(next_start - end, 0)
                silence = np.zeros(silence_len + pad_samples, dtype=np.float32)
                padded_audio.append(silence)

        # If last chunk doesn't end at end of audio, pad that too ( For safety from the hallucination )
        if chunks[-1]["end"] < len(audio):
            final_silence = np.zeros(len(audio) - chunks[-1]["end"], dtype=np.float32)
            padded_audio.append(final_silence)

        return np.concatenate(padded_audio)

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
        pass
