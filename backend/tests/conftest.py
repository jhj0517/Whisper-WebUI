import torchaudio


if not hasattr(torchaudio, "AudioMetaData"):
    try:
        from torchaudio._backend.common import AudioMetaData  # type: ignore[attr-defined]
    except Exception:
        class AudioMetaData:  # pragma: no cover
            pass

    torchaudio.AudioMetaData = AudioMetaData  # type: ignore[attr-defined]


if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]
else:
    try:
        _backends = list(torchaudio.list_audio_backends())  # type: ignore[attr-defined]
    except Exception:
        _backends = []
    if not _backends:
        torchaudio.list_audio_backends = lambda: ["soundfile"]  # type: ignore[attr-defined]

if not hasattr(torchaudio, "set_audio_backend"):
    torchaudio.set_audio_backend = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]

if not hasattr(torchaudio, "get_audio_backend"):
    torchaudio.get_audio_backend = lambda: None  # type: ignore[attr-defined]
