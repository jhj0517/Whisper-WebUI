from typing import BinaryIO, Union, List, Optional, Tuple
import numpy as np
import gradio as gr
from modules.whisper.data_classes import Segment

from modules.vad.base_silero_vad import BaseSileroVAD


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
        Run your custom VAD implementation.

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
        pass

    def update_model(self):
        """Initialize or update your VAD model."""
        pass

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