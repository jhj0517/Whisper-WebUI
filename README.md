# Whisper-WebUI
A Gradio-based browser interface for [Whisper](https://github.com/openai/whisper). You can use it as an Easy Subtitle Generator!

![Whisper WebUI](https://github.com/jhj0517/Whsiper-WebUI/blob/master/screenshot.png)

## Notebook
If you wish to try this on Colab, you can do it in [here](https://colab.research.google.com/github/jhj0517/Whisper-WebUI/blob/master/notebook/whisper-webui.ipynb)!

# Feature
- Generate subtitles from various sources, including :
  - Files
  - Youtube
  - Microphone
- Currently supported subtitle formats : 
  - SRT
  - WebVTT
  - txt ( only text file without timeline )
- Speech to Text Translation 
  - From other languages to English. ( This is Whisper's end-to-end speech-to-text translation feature )
- Text to Text Translation
  - Translate subtitle files using Facebook NLLB models
  - Translate subtitle files using DeepL API

# Installation and Running
### Prerequisite
To run this WebUI, you need to have `git`, `python` version 3.8 ~ 3.10, `FFmpeg` and `CUDA` version above 12.0 (only if you use NVIDIA GPU) 

Please follow the links below to install the necessary software:
- git : [https://git-scm.com/downloads](https://git-scm.com/downloads)
- python : [https://www.python.org/downloads/](https://www.python.org/downloads/) **( If your python version is too new, torch will not install properly.)**
- FFmpeg :  [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- CUDA : [https://developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)

After installing FFmpeg, **make sure to add the `FFmpeg/bin` folder to your system PATH!**

### Automatic Installation

1. Download `Whisper-WebUI.zip` with the file corresponding to your OS from [v1.0.0](https://github.com/jhj0517/Whisper-WebUI/releases/tag/v1.0.0) and extract its contents. 
2. Run `install.bat` or `install.sh` to install dependencies. (This will create a `venv` directory and install dependencies there.)
3. Start WebUI with `start-webui.bat` or `start-webui.sh`
4. To update the WebUI, run `update.bat` or `update.sh`

And you can also run the project with command line arguments if you like by running `start-webui.bat`, see [wiki](https://github.com/jhj0517/Whisper-WebUI/wiki/Command-Line-Arguments) for a guide to arguments.

- ## Or Run with Docker 

1. Build the image

```sh
docker build -t whisper-webui:latest . 
```

2. Run the container with commands

- For bash :
```sh
docker run --gpus all -d \
-v /path/to/models:/Whisper-WebUI/models \
-v /path/to/outputs:/Whisper-WebUI/outputs \
-p 7860:7860 \
-it \
whisper-webui:latest --server_name 0.0.0.0 --server_port 7860
```
- For PowerShell:
```shell
docker run --gpus all -d `
-v /path/to/models:/Whisper-WebUI/models `
-v /path/to/outputs:/Whisper-WebUI/outputs `
-p 7860:7860 `
-it `
whisper-webui:latest --server_name 0.0.0.0 --server_port 7860
```

# VRAM Usages
This project is integrated with [faster-whisper](https://github.com/guillaumekln/faster-whisper) by default for better VRAM usage and transcription speed.

According to faster-whisper, the efficiency of the optimized whisper model is as follows: 
| Implementation    | Precision | Beam size | Time  | Max. GPU memory | Max. CPU memory |
|-------------------|-----------|-----------|-------|-----------------|-----------------|
| openai/whisper    | fp16      | 5         | 4m30s | 11325MB         | 9439MB          |
| faster-whisper    | fp16      | 5         | 54s   | 4755MB          | 3244MB          |

If you want to use the original Open AI whisper implementation instead of optimized whisper, you can set the command line argument `--disable_faster_whisper` to `True`. See the [wiki](https://github.com/jhj0517/Whisper-WebUI/wiki/Command-Line-Arguments) for more information.

## Available models
This is Whisper's original VRAM usage table for models.

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |


`.en` models are for English only, and the cool thing is that you can use the `Translate to English` option from the "large" models!

## TODO🗓

- [x] Add DeepL API translation
- [x] Add NLLB Model translation
- [x] Integrate with faster-whisper
- [ ] Integrate with insanely-fast-whisper
- [ ] Integrate with whisperX


