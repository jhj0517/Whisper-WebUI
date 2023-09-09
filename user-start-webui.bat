@echo off

:: Set default values
set SERVER_NAME=
set SERVER_PORT=
set USERNAME=
set PASSWORD=

:: Uncomment and set the values for the optional arguments

:: set SERVER_NAME=0.0.0.0
:: set SERVER_PORT=36540
:: set USERNAME=your_username
:: set PASSWORD=your_password

:: Check if the arguments are uncommented and set them accordingly
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

:: Call the original .bat script with optional arguments
start-webui.bat %SERVER_NAME_ARG% %SERVER_PORT_ARG% %USERNAME_ARG% %PASSWORD_ARG%
pause
