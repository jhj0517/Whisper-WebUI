import requests
import time
import os
from datetime import datetime
import gradio as gr

from modules.utils.subtitle_manager import *

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
    'Automatic Detection': None,
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
                 output_dir: str
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
                        is_pro: bool,
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
        progress: gr.Progress
            Indicator to show progress directly in gradio.

        Returns
        ----------
        A List of
        String to return to gr.Textbox()
        Files to return to gr.Files()
        """

        files_info = {}
        for fileobj in fileobjs:
            file_path = fileobj.name
            file_name, file_ext = os.path.splitext(os.path.basename(fileobj.name))

            if file_ext == ".srt":
                parsed_dicts = parse_srt(file_path=file_path)

                batch_size = self.max_text_batch_size
                for batch_start in range(0, len(parsed_dicts), batch_size):
                    batch_end = min(batch_start + batch_size, len(parsed_dicts))
                    sentences_to_translate = [dic["sentence"] for dic in parsed_dicts[batch_start:batch_end]]
                    translated_texts = self.request_deepl_translate(auth_key, sentences_to_translate, source_lang,
                                                                    target_lang, is_pro)
                    for i, translated_text in enumerate(translated_texts):
                        parsed_dicts[batch_start + i]["sentence"] = translated_text["text"]
                    progress(batch_end / len(parsed_dicts), desc="Translating..")

                subtitle = get_serialized_srt(parsed_dicts)
                timestamp = datetime.now().strftime("%m%d%H%M%S")

                file_name = file_name[:-9]
                output_path = os.path.join(self.output_dir, "", f"{file_name}-{timestamp}.srt")
                write_file(subtitle, output_path)

            elif file_ext == ".vtt":
                parsed_dicts = parse_vtt(file_path=file_path)

                batch_size = self.max_text_batch_size
                for batch_start in range(0, len(parsed_dicts), batch_size):
                    batch_end = min(batch_start + batch_size, len(parsed_dicts))
                    sentences_to_translate = [dic["sentence"] for dic in parsed_dicts[batch_start:batch_end]]
                    translated_texts = self.request_deepl_translate(auth_key, sentences_to_translate, source_lang,
                                                                    target_lang, is_pro)
                    for i, translated_text in enumerate(translated_texts):
                        parsed_dicts[batch_start + i]["sentence"] = translated_text["text"]
                    progress(batch_end / len(parsed_dicts), desc="Translating..")

                subtitle = get_serialized_vtt(parsed_dicts)
                timestamp = datetime.now().strftime("%m%d%H%M%S")

                file_name = file_name[:-9]
                output_path = os.path.join(self.output_dir, "", f"{file_name}-{timestamp}.vtt")

                write_file(subtitle, output_path)

            files_info[file_name] = subtitle
        total_result = ''
        for file_name, subtitle in files_info.items():
            total_result += '------------------------------------\n'
            total_result += f'{file_name}\n\n'
            total_result += f'{subtitle}'

        gr_str = f"Done! Subtitle is in the outputs/translation folder.\n\n{total_result}"
        return [gr_str, output_path]

    def request_deepl_translate(self,
                                auth_key: str,
                                text: list,
                                source_lang: str,
                                target_lang: str,
                                is_pro: bool):
        """Request API response to DeepL server"""

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