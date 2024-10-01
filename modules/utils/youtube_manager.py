from pytubefix import YouTube
import subprocess
import os


def get_ytdata(link):
    return YouTube(link)


def get_ytmetas(link):
    yt = YouTube(link)
    return yt.thumbnail_url, yt.title, yt.description


def get_ytaudio(ytdata: YouTube):
    # Somehow the audio is corrupted so need to convert to valid audio file.
    # Fix for : https://github.com/jhj0517/Whisper-WebUI/issues/304

    audio_path = ytdata.streams.get_audio_only().download(filename=os.path.join("modules", "yt_tmp.wav"))
    temp_audio_path = os.path.join("modules", "yt_tmp_fixed.wav")

    try:
        subprocess.run([
            'ffmpeg', '-y',
            '-i', audio_path,
            temp_audio_path
        ], check=True)

        os.replace(temp_audio_path, audio_path)
        return audio_path
    except subprocess.CalledProcessError as e:
        print(f"Error during ffmpeg conversion: {e}")
        return None
