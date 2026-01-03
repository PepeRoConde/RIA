@echo off
set SOAR_HOME=%~dp0bin
set PATH=%SOAR_HOME%;%PATH%

cmd /c "%~dp0setup.bat"

start "" javaw.exe -Djava.library.path="%SOAR_HOME%" -jar "%~dp0bin\SoarJavaDebugger.jar" %*