import os
import torch
try:
    import intel_extension_for_pytorch as ipex
    if torch.xpu.is_available():
        xpu_available = True
except:
    pass
import gradio as gr
from abc import ABC, abstractmethod
from typing import List
from datetime import datetime

from modules.whisper.whisper_parameter import *
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
                if file_ext == ".srt":
                    parsed_dicts = parse_srt(file_path=fileobj)
                    total_progress = len(parsed_dicts)
                    for index, dic in enumerate(parsed_dicts):
                        progress(index / total_progress, desc="Translating..")
                        translated_text = self.translate(dic["sentence"], max_length=max_length)
                        dic["sentence"] = translated_text
                    subtitle = get_serialized_srt(parsed_dicts)

                elif file_ext == ".vtt":
                    parsed_dicts = parse_vtt(file_path=fileobj)
                    total_progress = len(parsed_dicts)
                    for index, dic in enumerate(parsed_dicts):
                        progress(index / total_progress, desc="Translating..")
                        translated_text = self.translate(dic["sentence"], max_length=max_length)
                        dic["sentence"] = translated_text
                    subtitle = get_serialized_vtt(parsed_dicts)

                if add_timestamp:
                    timestamp = datetime.now().strftime("%m%d%H%M%S")
                    file_name += f"-{timestamp}"

                output_path = os.path.join(self.output_dir, f"{file_name}{file_ext}")
                write_file(subtitle, output_path)

                files_info[file_name] = {"subtitle": subtitle, "path": output_path}

            total_result = ''
            for file_name, info in files_info.items():
                total_result += '------------------------------------\n'
                total_result += f'{file_name}\n\n'
                total_result += f'{info["subtitle"]}'
            gr_str = f"Done! Subtitle is in the outputs/translation folder.\n\n{total_result}"

            output_file_paths = [item["path"] for key, item in files_info.items()]
            return [gr_str, output_file_paths]

        except Exception as e:
            print(f"Error: {str(e)}")
        finally:
            self.release_gpu_memory()

    @staticmethod
    def get_device():
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        elif torch.xpu.is_available():
            return "xpu"
        else:
            return "cpu"

    @staticmethod
    def release_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        elif torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.reset_peak_memory_stats()

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
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        cached_params["translation"]["nllb"] = {
            "model_size": model_size,
            "source_lang": src_lang,
            "target_lang": tgt_lang,
            "max_length": max_length,
        }
        cached_params["translation"]["add_timestamp"] = add_timestamp
        save_yaml(cached_params, DEFAULT_PARAMETERS_CONFIG_PATH)
