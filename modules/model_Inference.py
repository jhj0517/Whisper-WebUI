import whisper
from modules.subtitle_manager import get_srt,get_vtt,write_srt,write_vtt,safe_filename
from modules.youtube_manager import get_ytdata,get_ytaudio
import gradio as gr
import os
from datetime import datetime

class ModelInference():
    def __init__(self):
        print("\nInitializing Model..\n")
        self.default_model = "large-v2"
        self.model = whisper.load_model(self.default_model)

    def transcribe_file(self,fileobj
                        ,model,lang,subformat,istranslate,
                        progress=gr.Progress()):
        
        def progress_callback(progress_value):
            progress(progress_value,desc="Transcribing..")
        
        if model != self.default_model or model==None:
            progress(0,desc="Initializing Model..")
            self.model = whisper.load_model(model)

        if lang == "Automatic Detection" :
            lang = None    

        progress(0,desc="Loading Audio..")    
        audio = whisper.load_audio(fileobj.name)

        if istranslate == True:
            result = self.model.transcribe(audio=audio,language=lang,verbose=False,task="translate",progress_callback=progress_callback)
        else : 
            result = self.model.transcribe(audio=audio,language=lang,verbose=False,progress_callback=progress_callback)

        progress(1,desc="Completed!")

        file_name, file_ext = os.path.splitext(os.path.basename(fileobj.orig_name))
        file_name = file_name[:-9]
        file_name = safe_filename(file_name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"outputs/{file_name}-{timestamp}"

        if subformat == "SRT":
            subtitle = get_srt(result["segments"])
            write_srt(subtitle,f"{output_path}.srt")
        elif subformat == "WebVTT":
            subtitle = get_vtt(result["segments"])
            write_vtt(subtitle,f"{output_path}.vtt")    

        return f"Done! Subtitle is in the outputs folder.\n\n{subtitle}"
    
    def transcribe_youtube(self,youtubelink
                        ,model,lang,subformat,istranslate,
                        progress=gr.Progress()):
        
        def progress_callback(progress_value):
            progress(progress_value,desc="Transcribing..")

        if model != self.default_model or model==None:
            progress(0,desc="Initializing Model..")
            self.model = whisper.load_model(model)

        if lang == "Automatic Detection" :
            lang = None    

        progress(0,desc="Loading Audio from Youtube..")    
        yt = get_ytdata(youtubelink)
        audio = whisper.load_audio(get_ytaudio(yt))

        if istranslate == True:
            result = self.model.transcribe(audio=audio,language=lang,verbose=False,task="translate",progress_callback=progress_callback)
        else : 
            result = self.model.transcribe(audio=audio,language=lang,verbose=False,progress_callback=progress_callback)

        progress(1,desc="Completed!")

        file_name = safe_filename(yt.title)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"outputs/{file_name}-{timestamp}"

        if subformat == "SRT":
            subtitle = get_srt(result["segments"])
            write_srt(subtitle,f"{output_path}.srt")
        elif subformat == "WebVTT":
            subtitle = get_vtt(result["segments"])
            write_vtt(subtitle,f"{output_path}.vtt")   

        return f"Done! Subtitle file is in the outputs folder.\n\n{subtitle}"
    
    def transcribe_mic(self,micaudio
                    ,model,lang,subformat,istranslate,
                    progress=gr.Progress()):

        def progress_callback(progress_value):
            progress(progress_value,desc="Transcribing..")
        
        if model != self.default_model or model==None:
            progress(0,desc="Initializing Model..")
            self.model = whisper.load_model(model)

        if lang == "Automatic Detection" :
            lang = None    

        progress(0,desc="Loading Audio..")    

        if istranslate == True:
            result = self.model.transcribe(audio=micaudio,language=lang,verbose=False,task="translate",progress_callback=progress_callback)
        else : 
            result = self.model.transcribe(audio=micaudio,language=lang,verbose=False,progress_callback=progress_callback)

        progress(1,desc="Completed!")

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_path = f"outputs/Mic-{timestamp}"

        if subformat == "SRT":
            subtitle = get_srt(result["segments"])
            write_srt(subtitle,f"{output_path}.srt")
        elif subformat == "WebVTT":
            subtitle = get_vtt(result["segments"])
            write_vtt(subtitle,f"{output_path}.vtt")   
            
        return f"Done! Subtitle file is in the outputs folder.\n\n{subtitle}"
    
    