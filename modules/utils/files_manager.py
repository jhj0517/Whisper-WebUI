import os
import fnmatch
from ruamel.yaml import YAML
from gradio.utils import NamedString

from modules.utils.paths import DEFAULT_PARAMETERS_CONFIG_PATH


def load_yaml(path: str = DEFAULT_PARAMETERS_CONFIG_PATH):
    yaml = YAML(typ="safe")
    yaml.preserve_quotes = True
    with open(path, 'r', encoding='utf-8') as file:
        config = yaml.load(file)
    return config


def save_yaml(data: dict, path: str = DEFAULT_PARAMETERS_CONFIG_PATH):
    yaml = YAML(typ="safe")
    yaml.map_indent = 2
    yaml.sequence_indent = 4
    yaml.sequence_dash_offset = 2
    yaml.preserve_quotes = True
    yaml.default_flow_style = False
    yaml.sort_base_mapping_type_on_output = False

    with open(path, 'w', encoding='utf-8') as file:
        yaml.dump(data, file)
    return path


def get_media_files(folder_path, include_sub_directory=False):
    video_extensions = ['*.mp4', '*.mkv', '*.flv', '*.avi', '*.mov', '*.wmv', '*.webm', '*.m4v', '*.mpeg', '*.mpg',
                        '*.3gp', '*.f4v', '*.ogv', '*.vob', '*.mts', '*.m2ts', '*.divx', '*.mxf', '*.rm', '*.rmvb']
    audio_extensions = ['*.mp3', '*.wav', '*.aac', '*.flac', '*.ogg', '*.m4a']
    media_extensions = video_extensions + audio_extensions

    media_files = []

    if include_sub_directory:
        for root, _, files in os.walk(folder_path):
            for extension in media_extensions:
                media_files.extend(
                    os.path.join(root, file) for file in fnmatch.filter(files, extension)
                    if os.path.exists(os.path.join(root, file))
                )
    else:
        for extension in media_extensions:
            media_files.extend(
                os.path.join(folder_path, file) for file in fnmatch.filter(os.listdir(folder_path), extension)
                if os.path.isfile(os.path.join(folder_path, file)) and os.path.exists(os.path.join(folder_path, file))
            )

    return media_files


def format_gradio_files(files: list):
    if not files:
        return files

    gradio_files = []
    for file in files:
        gradio_files.append(NamedString(file))
    return gradio_files


def is_video(file_path):
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg', '.3gp']
    extension = os.path.splitext(file_path)[1].lower()
    return extension in video_extensions


def read_file(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        subtitle_content = f.read()
    return subtitle_content
