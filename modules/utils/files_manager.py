import os
import fnmatch

from gradio.utils import NamedString


def get_media_files(folder_path, include_sub_directory=False):
    video_extensions = ['*.mp4', '*.mkv', '*.flv', '*.avi', '*.mov', '*.wmv']
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

