from pytube import YouTube
import os


def get_ytdata(link):
    return YouTube(link)


def get_ytmetas(link):
    yt = YouTube(link)
    return yt.thumbnail_url, yt.title, yt.description


def get_ytaudio(ytdata: YouTube):
    return ytdata.streams.get_audio_only().download(filename=os.path.join("modules", "yt_tmp.wav"))
