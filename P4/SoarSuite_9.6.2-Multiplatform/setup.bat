@REM Set up Soar for Windows, but only if not done already

@echo off
set SOAR_HOME=%~dp0bin

if not exist "%SOAR_HOME%\pkgIndex.tcl" (
    echo "First time initialization of Soar for Windows..."
    copy "%SOAR_HOME%\win_x86-64\swt.jar" "%SOAR_HOME%\java\swt.jar"
    copy "%SOAR_HOME%\win_x86-64\*.*" "%SOAR_HOME%\"
    @REM rmdir /S /Q "%SOAR_HOME%\mac_x86-64"
    @REM rmdir /S /Q "%SOAR_HOME%\mac_ARM64"
    @REM rmdir /S /Q "%SOAR_HOME%\linux_x86-64"
    @REM rmdir /S /Q "%SOAR_HOME%\win_x86-64"
    @REM del /F /Q "%~dp0*.sh"
)
