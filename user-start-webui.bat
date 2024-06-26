@echo off
:: This batch file is for launching with command line args
:: See the wiki for a guide to command line arguments: https://github.com/jhj0517/Whisper-WebUI/wiki/Command-Line-Arguments
:: Set the values here to whatever you want. See the wiki above for how to set this.
set SERVER_NAME=
set SERVER_PORT=
set USERNAME=
set PASSWORD=
set SHARE=
set THEME=
set API_OPEN=
set WHISPER_TYPE=
set WHISPER_MODEL_DIR=
set FASTER_WHISPER_MODEL_DIR=
set INSANELY_FAST_WHISPER_MODEL_DIR=
set DIARIZATION_MODEL_DIR=


if not "%SERVER_NAME%"=="" (
    set SERVER_NAME_ARG=--server_name %SERVER_NAME%
)
if not "%SERVER_PORT%"=="" (
    set SERVER_PORT_ARG=--server_port %SERVER_PORT%
)
if not "%USERNAME%"=="" (
    set USERNAME_ARG=--username %USERNAME%
)
if not "%PASSWORD%"=="" (
    set PASSWORD_ARG=--password %PASSWORD%
)
if /I "%SHARE%"=="true" (
    set SHARE_ARG=--share
)
if not "%THEME%"=="" (
    set THEME_ARG=--theme %THEME%
)
if /I "%DISABLE_FASTER_WHISPER%"=="true" (
    set DISABLE_FASTER_WHISPER_ARG=--disable_faster_whisper
)
if /I "%API_OPEN%"=="true" (
    set API_OPEN=--api_open
)
if not "%WHISPER_TYPE%"=="" (
    set WHISPER_TYPE_ARG=--whisper_type %WHISPER_TYPE%
)
if not "%WHISPER_MODEL_DIR%"=="" (
    set WHISPER_MODEL_DIR_ARG=--whisper_model_dir "%WHISPER_MODEL_DIR%"
)
if not "%FASTER_WHISPER_MODEL_DIR%"=="" (
    set FASTER_WHISPER_MODEL_DIR_ARG=--faster_whisper_model_dir "%FASTER_WHISPER_MODEL_DIR%"
)
if not "%INSANELY_FAST_WHISPER_MODEL_DIR%"=="" (
    set INSANELY_FAST_WHISPER_MODEL_DIR_ARG=--insanely_fast_whisper_model_dir "%INSANELY_FAST_WHISPER_MODEL_DIR%"
)
if not "%DIARIZATION_MODEL_DIR%"=="" (
    set DIARIZATION_MODEL_DIR_ARG=--diarization_model_dir "%DIARIZATION_MODEL_DIR%"
)

:: Call the original .bat script with cli arguments
start-webui.bat %SERVER_NAME_ARG% %SERVER_PORT_ARG% %USERNAME_ARG% %PASSWORD_ARG% %SHARE_ARG% %THEME_ARG% %API_OPEN% %WHISPER_TYPE_ARG% %WHISPER_MODEL_DIR_ARG% %FASTER_WHISPER_MODEL_DIR_ARG% %INSANELY_FAST_WHISPER_MODEL_DIR_ARG% %DIARIZATION_MODEL_DIR_ARG%
pause