:: This batch file is for launching with command line args
@echo off

:: Set values
set SERVER_NAME=
set SERVER_PORT=
set USERNAME=
set PASSWORD=
set SHARE=

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

:: Call the original .bat script with optional arguments
start-webui.bat %SERVER_NAME_ARG% %SERVER_PORT_ARG% %USERNAME_ARG% %PASSWORD_ARG% %SHARE_ARG%
pause