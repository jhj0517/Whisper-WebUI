import whisper
from modules.subtitle_manager import get_srt, get_vtt, write_srt, write_vtt, safe_filename
from modules.youtube_manager import get_ytdata, get_ytaudio
import gradio as gr
import os
from datetime import datetime

DEFAULT_MODEL_SIZE = "large-v2"


class WhisperInference:
    def __init__(self):
        print("\nInitializing Model..\n")
        self.current_model_size = DEFAULT_MODEL_SIZE
        self.model = whisper.load_model(name=DEFAULT_MODEL_SIZE, download_root="models/Whisper")
        self.available_models = whisper.available_models()
        self.available_langs = sorted(list(whisper.tokenizer.LANGUAGES.values()))

    def transcribe_file(self, fileobjs
                        , model_size, lang, subformat, istranslate,
                        progress=gr.Progress()):

        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        if model_size != self.current_model_size:
            progress(0, desc="Initializing Model..")
            self.current_model_size = model_size
            self.model = whisper.load_model(name=model_size, download_root="models/Whisper")

        if lang == "Automatic Detection":
            lang = None

        progress(0, desc="Loading Audio..")

        files_info = {}
        for fileobj in fileobjs:

            audio = whisper.load_audio(fileobj.name)

            translatable_model = ["large", "large-v1", "large-v2"]
            if istranslate and self.current_model_size in translatable_model:
                result = self.model.transcribe(audio=audio, language=lang, verbose=False, task="translate",
                                               progress_callback=progress_callback)
            else:
                result = self.model.transcribe(audio=audio, language=lang, verbose=False,
                                               progress_callback=progress_callback)

            progress(1, desc="Completed!")

            file_name, file_ext = os.path.splitext(os.path.basename(fileobj.orig_name))
            file_name = file_name[:-9]
            file_name = safe_filename(file_name)
            timestamp = datetime.now().strftime("%m%d%H%M%S")
            output_path = f"outputs/{file_name}-{timestamp}"

            if subformat == "SRT":
                subtitle = get_srt(result["segments"])
                write_srt(subtitle, f"{output_path}.srt")
            elif subformat == "WebVTT":
                subtitle = get_vtt(result["segments"])
                write_vtt(subtitle, f"{output_path}.vtt")

            files_info[file_name] = subtitle

        total_result = ''
        for file_name, subtitle in files_info.items():
            total_result += '------------------------------------\n'
            total_result += f'{file_name}\n\n'
            total_result += f'{subtitle}'

        return f"Done! Subtitle is in the outputs folder.\n\n{total_result}"

    def transcribe_youtube(self, youtubelink
                           , model_size, lang, subformat, istranslate,
                           progress=gr.Progress()):

        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        if model_size != self.current_model_size:
            progress(0, desc="Initializing Model..")
            self.current_model_size = model_size
            self.model = whisper.load_model(name=model_size, download_root="models/Whisper")

        if lang == "Automatic Detection":
            lang = None

        progress(0, desc="Loading Audio from Youtube..")
        yt = get_ytdata(youtubelink)
        audio = whisper.load_audio(get_ytaudio(yt))

        translatable_model = ["large", "large-v1", "large-v2"]
        if istranslate and self.current_model_size in translatable_model:
            result = self.model.transcribe(audio=audio, language=lang, verbose=False, task="translate",
                                           progress_callback=progress_callback)
        else:
            result = self.model.transcribe(audio=audio, language=lang, verbose=False,
                                           progress_callback=progress_callback)

        progress(1, desc="Completed!")

        file_name = safe_filename(yt.title)
        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_path = f"outputs/{file_name}-{timestamp}"

        if subformat == "SRT":
            subtitle = get_srt(result["segments"])
            write_srt(subtitle, f"{output_path}.srt")
        elif subformat == "WebVTT":
            subtitle = get_vtt(result["segments"])
            write_vtt(subtitle, f"{output_path}.vtt")

        return f"Done! Subtitle file is in the outputs folder.\n\n{subtitle}"

    def transcribe_mic(self, micaudio
                       , model_size, lang, subformat, istranslate,
                       progress=gr.Progress()):

        def progress_callback(progress_value):
            progress(progress_value, desc="Transcribing..")

        if model_size != self.current_model_size:
            progress(0, desc="Initializing Model..")
            self.current_model_size = model_size
            self.model = whisper.load_model(name=model_size, download_root="models/Whisper")

        if lang == "Automatic Detection":
            lang = None

        progress(0, desc="Loading Audio..")

        translatable_model = ["large", "large-v1", "large-v2"]
        if istranslate and self.current_model_size in translatable_model:
            result = self.model.transcribe(audio=micaudio, language=lang, verbose=False, task="translate",
                                           progress_callback=progress_callback)
        else:
            result = self.model.transcribe(audio=micaudio, language=lang, verbose=False,
                                           progress_callback=progress_callback)

        progress(1, desc="Completed!")

        timestamp = datetime.now().strftime("%m%d%H%M%S")
        output_path = f"outputs/Mic-{timestamp}"

        if subformat == "SRT":
            subtitle = get_srt(result["segments"])
            write_srt(subtitle, f"{output_path}.srt")
        elif subformat == "WebVTT":
            subtitle = get_vtt(result["segments"])
            write_vtt(subtitle, f"{output_path}.vtt")

        return f"Done! Subtitle file is in the outputs folder.\n\n{subtitle}"
