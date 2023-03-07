# Whsiper-WebUI
A Gradio-based browser interface for Whisper. You can use it as an Easy Subtitle Generator!

![Whisper WebUI](https://github.com/jhj0517/Whsiper-WebUI/blob/master/screenshot.png)

# Feature
- Generate subtitles from various sources, including :
  - Files
  - Youtube
  - Microphone
- Currently supported subtitle formats : 
  - SRT
  - WebVTT
- Speech to Text Translation
  - From other languages to English.

# Installation and Running
## Prerequisite
To run Whisper, you need to have `python` version 3.8 to 3.10 and `FFmpeg`.

Please follow the links below to install the necessary software:
- python : [https://www.python.org/downloads/](https://www.python.org/downloads/)
- FFmpeg :  [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)

After installing FFmpeg, **make sure to add the `FFmpeg/bin` folder to your system PATH!**

## Automatic Installation
If you have satisfied the prerequisites listed above, you are now ready to start Whisper-WebUI.

1. Run `Install.bat` from Windows Explorer as a regular, non-administrator user.
2. After installation, run the `start-webui.bat`. (It will automatically download the model if it is not already installed.)
3. Open your web browser and go to `http://localhost:7860`

# Available models

The WebUI uses the Open AI Whisper model

|  Size  | Parameters | English-only model | Multilingual model | Required VRAM | Relative speed |
|:------:|:----------:|:------------------:|:------------------:|:-------------:|:--------------:|
|  tiny  |    39 M    |     `tiny.en`      |       `tiny`       |     ~1 GB     |      ~32x      |
|  base  |    74 M    |     `base.en`      |       `base`       |     ~1 GB     |      ~16x      |
| small  |   244 M    |     `small.en`     |      `small`       |     ~2 GB     |      ~6x       |
| medium |   769 M    |    `medium.en`     |      `medium`      |     ~5 GB     |      ~2x       |
| large  |   1550 M   |        N/A         |      `large`       |    ~10 GB     |       1x       |


`.en` models are for English only, and the cool thing is that you can use the `Translate to English` option from the "large" models!

## Support

If you found this project useful, kindly consider supporting it.

<a href="https://www.buymeacoffee.com/jhj0517" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" alt="Buy Me A Coffee" height="41" width="174"></a>


