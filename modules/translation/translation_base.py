import os
import torch
import gradio as gr
from abc import ABC, abstractmethod
import gc
from typing import List
from datetime import datetime

import modules.translation.nllb_inference as nllb
from modules.whisper.data_classes import *
from modules.utils.subtitle_manager import *
from modules.utils.files_manager import load_yaml, save_yaml
from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH, NLLB_MODELS_DIR, TRANSLATION_OUTPUT_DIR


class TranslationBase(ABC):
    def __init__(self,
                 model_dir: str = NLLB_MODELS_DIR,
                 output_dir: str = TRANSLATION_OUTPUT_DIR
                 ):
        super().__init__()
        self.model = None
        self.model_dir = model_dir
        self.output_dir = output_dir
        os.makedirs(self.model_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        self.current_model_size = None
        self.device = self.get_device()

    @abstractmethod
    def translate(self,
                  text: str,
                  max_length: int
                  ):
        pass

    @abstractmethod
    def update_model(self,
                     model_size: str,
                     src_lang: str,
                     tgt_lang: str,
                     progress: gr.Progress = gr.Progress()
                     ):
        pass

    def translate_file(self,
                       fileobjs: list,
                       model_size: str,
                       src_lang: str,
                       tgt_lang: str,
                       max_length: int = 200,
                       add_timestamp: bool = True,
                       progress=gr.Progress()) -> list:
        """
        Translate subtitle file from source language to target language

        Parameters
        ----------
        fileobjs: list
            List of files to transcribe from gr.Files()
        model_size: str
            Whisper model size from gr.Dropdown()
        src_lang: str
            Source language of the file to translate from gr.Dropdown()
        tgt_lang: str
            Target language of the file to translate from gr.Dropdown()
        max_length: int
            Max length per line to translate
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.
            I use a forked version of whisper for this. To see more info : https://github.com/jhj0517/jhj0517-whisper/tree/add-progress-callback

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        try:
            if fileobjs and isinstance(fileobjs[0], gr.utils.NamedString):
                fileobjs = [file.name for file in fileobjs]

            self.cache_parameters(model_size=model_size,
                                  src_lang=src_lang,
                                  tgt_lang=tgt_lang,
                                  max_length=max_length,
                                  add_timestamp=add_timestamp)

            self.update_model(model_size=model_size,
                              src_lang=src_lang,
                              tgt_lang=tgt_lang,
                              progress=progress)

            files_info = {}
            for fileobj in fileobjs:
                file_name, file_ext = os.path.splitext(os.path.basename(fileobj))
                writer = get_writer(file_ext, self.output_dir)
                segments = writer.to_segments(fileobj)
                for i, segment in enumerate(segments):
                    progress(i / len(segments), desc="Translating..")
                    translated_text = self.translate(segment.text, max_length=max_length)
                    segment.text = translated_text

                subtitle, file_path = generate_file(
                    output_dir=self.output_dir,
                    output_file_name=file_name,
                    output_format=file_ext,
                    result=segments,
                    add_timestamp=add_timestamp
                )

                files_info[file_name] = {"subtitle": subtitle, "path": file_path}

            total_result = ''
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
            gr_str = f"Done! Subtitle is in the outputs/translation folder.\n\n{total_result}"

            output_file_paths = [item["path"] for key, item in files_info.items()]
            return [gr_str, output_file_paths]

        except Exception as e:
            print(f"Error translating file: {e}")
            raise
        finally:
            self.release_cuda_memory()

    def offload(self):
        """Offload the model and free up the memory"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.device == "cuda":
            self.release_cuda_memory()
        gc.collect()

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

    @staticmethod
    def cache_parameters(model_size: str,
                         src_lang: str,
                         tgt_lang: str,
                         max_length: int,
                         add_timestamp: bool):
        def validate_lang(lang: str):
            if lang in list(nllb.NLLB_AVAILABLE_LANGS.values()):
                flipped = {value: key for key, value in nllb.NLLB_AVAILABLE_LANGS.items()}
                return flipped[lang]
            return lang

        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        cached_params["translation"]["nllb"] = {
            "model_size": model_size,
            "source_lang": validate_lang(src_lang),
            "target_lang": validate_lang(tgt_lang),
            "max_length": max_length,
        }
        cached_params["translation"]["add_timestamp"] = add_timestamp
        save_yaml(cached_params, DEFAULT_PARAMETERS_CONFIG_PATH)
