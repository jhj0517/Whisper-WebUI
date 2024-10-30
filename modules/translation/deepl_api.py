import requests
import time
import os
from datetime import datetime
import gradio as gr

from modules.utils.paths import TRANSLATION_OUTPUT_DIR, DEFAULT_PARAMETERS_CONFIG_PATH
from modules.utils.constants import AUTOMATIC_DETECTION
from modules.utils.subtitle_manager import *
from modules.utils.files_manager import load_yaml, save_yaml

"""
This is written with reference to the DeepL API documentation.
If you want to know the information of the DeepL API, see here: https://www.deepl.com/docs-api/documents
"""

DEEPL_AVAILABLE_TARGET_LANGS = {
    'Bulgarian': 'BG',
    'Czech': 'CS',
    'Danish': 'DA',
    'German': 'DE',
    'Greek': 'EL',
    'English': 'EN',
    'English (British)': 'EN-GB',
    'English (American)': 'EN-US',
    'Spanish': 'ES',
    'Estonian': 'ET',
    'Finnish': 'FI',
    'French': 'FR',
    'Hungarian': 'HU',
    'Indonesian': 'ID',
    'Italian': 'IT',
    'Japanese': 'JA',
    'Korean': 'KO',
    'Lithuanian': 'LT',
    'Latvian': 'LV',
    'Norwegian (Bokmål)': 'NB',
    'Dutch': 'NL',
    'Polish': 'PL',
    'Portuguese': 'PT',
    'Portuguese (Brazilian)': 'PT-BR',
    'Portuguese (all Portuguese varieties excluding Brazilian Portuguese)': 'PT-PT',
    'Romanian': 'RO',
    'Russian': 'RU',
    'Slovak': 'SK',
    'Slovenian': 'SL',
    'Swedish': 'SV',
    'Turkish': 'TR',
    'Ukrainian': 'UK',
    'Chinese (simplified)': 'ZH'
}

DEEPL_AVAILABLE_SOURCE_LANGS = {
    AUTOMATIC_DETECTION: None,
    'Bulgarian': 'BG',
    'Czech': 'CS',
    'Danish': 'DA',
    'German': 'DE',
    'Greek': 'EL',
    'English': 'EN',
    'Spanish': 'ES',
    'Estonian': 'ET',
    'Finnish': 'FI',
    'French': 'FR',
    'Hungarian': 'HU',
    'Indonesian': 'ID',
    'Italian': 'IT',
    'Japanese': 'JA',
    'Korean': 'KO',
    'Lithuanian': 'LT',
    'Latvian': 'LV',
    'Norwegian (Bokmål)': 'NB',
    'Dutch': 'NL',
    'Polish': 'PL',
    'Portuguese (all Portuguese varieties mixed)': 'PT',
    'Romanian': 'RO',
    'Russian': 'RU',
    'Slovak': 'SK',
    'Slovenian': 'SL',
    'Swedish': 'SV',
    'Turkish': 'TR',
    'Ukrainian': 'UK',
    'Chinese': 'ZH'
}


class DeepLAPI:
    def __init__(self,
                 output_dir: str = TRANSLATION_OUTPUT_DIR
                 ):
        self.api_interval = 1
        self.max_text_batch_size = 50
        self.available_target_langs = DEEPL_AVAILABLE_TARGET_LANGS
        self.available_source_langs = DEEPL_AVAILABLE_SOURCE_LANGS
        self.output_dir = output_dir

    def translate_deepl(self,
                        auth_key: str,
                        fileobjs: list,
                        source_lang: str,
                        target_lang: str,
                        is_pro: bool = False,
                        add_timestamp: bool = True,
                        progress=gr.Progress()) -> list:
        """
        Translate subtitle files using DeepL API
        Parameters
        ----------
        auth_key: str
            API Key for DeepL from gr.Textbox()
        fileobjs: list
            List of files to transcribe from gr.Files()
        source_lang: str
            Source language of the file to transcribe from gr.Dropdown()
        target_lang: str
            Target language of the file to transcribe from gr.Dropdown()
        is_pro: str
            Boolean value that is about pro user or not from gr.Checkbox().
        add_timestamp: bool
            Boolean value from gr.Checkbox() that determines whether to add a timestamp at the end of the filename.
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """
        if fileobjs and isinstance(fileobjs[0], gr.utils.NamedString):
            fileobjs = [fileobj.name for fileobj in fileobjs]

        self.cache_parameters(
            api_key=auth_key,
            is_pro=is_pro,
            source_lang=source_lang,
            target_lang=target_lang,
            add_timestamp=add_timestamp
        )

        files_info = {}
        for file_path in fileobjs:
            file_name, file_ext = os.path.splitext(os.path.basename(file_path))
            writer = get_writer(file_ext, self.output_dir)
            segments = writer.to_segments(file_path)

            batch_size = self.max_text_batch_size
            for batch_start in range(0, len(segments), batch_size):
                progress(batch_start / len(segments), desc="Translating..")
                sentences_to_translate = [seg.text for seg in segments[batch_start:batch_start+batch_size]]
                translated_texts = self.request_deepl_translate(auth_key, sentences_to_translate, source_lang,
                                                                target_lang, is_pro)
                for i, translated_text in enumerate(translated_texts):
                    segments[batch_start + i].text = translated_text["text"]

            subtitle, output_path = generate_file(
                output_dir=self.output_dir,
                output_file_name=file_name,
                output_format=file_ext,
                result=segments,
                add_timestamp=add_timestamp
            )

            files_info[file_name] = {"subtitle": subtitle, "path": output_path}

        total_result = ''
        for file_name, info in files_info.items():
            total_result += '------------------------------------\n'
            total_result += f'{file_name}\n\n'
            total_result += f'{info["subtitle"]}'
        gr_str = f"Done! Subtitle is in the outputs/translation folder.\n\n{total_result}"

        output_file_paths = [item["path"] for key, item in files_info.items()]
        return [gr_str, output_file_paths]

    def request_deepl_translate(self,
                                auth_key: str,
                                text: list,
                                source_lang: str,
                                target_lang: str,
                                is_pro: bool = False):
        """Request API response to DeepL server"""
        if source_lang not in list(DEEPL_AVAILABLE_SOURCE_LANGS.keys()):
            raise ValueError(f"Source language {source_lang} is not supported."
                             f"Use one of {list(DEEPL_AVAILABLE_SOURCE_LANGS.keys())}")
        if target_lang not in list(DEEPL_AVAILABLE_TARGET_LANGS.keys()):
            raise ValueError(f"Target language {target_lang} is not supported."
                             f"Use one of {list(DEEPL_AVAILABLE_TARGET_LANGS.keys())}")

        url = 'https://api.deepl.com/v2/translate' if is_pro else 'https://api-free.deepl.com/v2/translate'
        headers = {
            'Authorization': f'DeepL-Auth-Key {auth_key}'
        }
        data = {
            'text': text,
            'source_lang': DEEPL_AVAILABLE_SOURCE_LANGS[source_lang],
            'target_lang': DEEPL_AVAILABLE_TARGET_LANGS[target_lang]
        }
        response = requests.post(url, headers=headers, data=data).json()
        time.sleep(self.api_interval)
        return response["translations"]

    @staticmethod
    def cache_parameters(api_key: str,
                         is_pro: bool,
                         source_lang: str,
                         target_lang: str,
                         add_timestamp: bool):
        cached_params = load_yaml(DEFAULT_PARAMETERS_CONFIG_PATH)
        cached_params["translation"]["deepl"] = {
            "api_key": api_key,
            "is_pro": is_pro,
            "source_lang": source_lang,
            "target_lang": target_lang
        }
        cached_params["translation"]["add_timestamp"] = add_timestamp
        save_yaml(cached_params, DEFAULT_PARAMETERS_CONFIG_PATH)
