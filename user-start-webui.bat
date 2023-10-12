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
set DISABLE_FASTER_WHISPER=true




:: Set args accordingly
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

:: Call the original .bat script with optional arguments
start-webui.bat %SERVER_NAME_ARG% %SERVER_PORT_ARG% %USERNAME_ARG% %PASSWORD_ARG% %SHARE_ARG% %THEME_ARG% %DISABLE_FASTER_WHISPER_ARG%
pause