import re

def timeformat_srt(time):
    hours = time//3600
    minutes = (time - hours*3600)//60
    seconds = time - hours*3600 - minutes*60
    milliseconds = (time - int(time))*1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d},{int(milliseconds):03d}"

def timeformat_vtt(time):
    hours = time//3600
    minutes = (time - hours*3600)//60
    seconds = time - hours*3600 - minutes*60
    milliseconds = (time - int(time))*1000
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}.{int(milliseconds):03d}"

def write_srt(subtitle,output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(subtitle)

def write_vtt(subtitle,output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(subtitle)        

def get_srt(segments):
    output = ""
    for i, segment in enumerate(segments):
        output += f"{i+1}\n"
        output += f"{timeformat_srt(segment['start'])} --> {timeformat_srt(segment['end'])}\n"
        output += f"{segment['text']}\n\n"        
    return output    

def get_vtt(segments):
    output = "WebVTT\n\n"
    for i, segment in enumerate(segments):
        output += f"{i+1}\n"
        output += f"{timeformat_vtt(segment['start'])} --> {timeformat_vtt(segment['end'])}\n"
        output += f"{segment['text']}\n\n"        
    return output

def safe_filename(name):
    INVALID_FILENAME_CHARS = r'[<>:"/\\|?*\x00-\x1f]'
    return re.sub(INVALID_FILENAME_CHARS, '_', name)