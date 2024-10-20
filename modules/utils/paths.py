import os

WEBUI_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODELS_DIR = os.path.join(WEBUI_DIR, "models")
WHISPER_MODELS_DIR = os.path.join(MODELS_DIR, "Whisper")
FASTER_WHISPER_MODELS_DIR = os.path.join(WHISPER_MODELS_DIR, "faster-whisper")
INSANELY_FAST_WHISPER_MODELS_DIR = os.path.join(WHISPER_MODELS_DIR, "insanely-fast-whisper")
NLLB_MODELS_DIR = os.path.join(MODELS_DIR, "NLLB")
DIARIZATION_MODELS_DIR = os.path.join(MODELS_DIR, "Diarization")
UVR_MODELS_DIR = os.path.join(MODELS_DIR, "UVR", "MDX_Net_Models")
CONFIGS_DIR = os.path.join(WEBUI_DIR, "configs")
DEFAULT_PARAMETERS_CONFIG_PATH = os.path.join(CONFIGS_DIR, "default_parameters.yaml")
I18N_YAML_PATH = os.path.join(CONFIGS_DIR, "translation.yaml")
OUTPUT_DIR = os.path.join(WEBUI_DIR, "outputs")
TRANSLATION_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "translations")
UVR_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "UVR")
UVR_INSTRUMENTAL_OUTPUT_DIR = os.path.join(UVR_OUTPUT_DIR, "instrumental")
UVR_VOCALS_OUTPUT_DIR = os.path.join(UVR_OUTPUT_DIR, "vocals")

for dir_path in [MODELS_DIR,
                 WHISPER_MODELS_DIR,
                 FASTER_WHISPER_MODELS_DIR,
                 INSANELY_FAST_WHISPER_MODELS_DIR,
                 NLLB_MODELS_DIR,
                 DIARIZATION_MODELS_DIR,
                 UVR_MODELS_DIR,
                 CONFIGS_DIR,
                 OUTPUT_DIR,
                 TRANSLATION_OUTPUT_DIR,
                 UVR_INSTRUMENTAL_OUTPUT_DIR,
                 UVR_VOCALS_OUTPUT_DIR]:
    os.makedirs(dir_path, exist_ok=True)
