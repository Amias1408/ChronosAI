@echo off
setlocal
set "RELEASE_EXE=%~dp0src\Chronos.Cli\bin\Release\net10.0\chronos.exe"
set "DEBUG_EXE=%~dp0src\Chronos.Cli\bin\Debug\net10.0\chronos.exe"
if exist "%RELEASE_EXE%" (
    "%RELEASE_EXE%" %*
) else if exist "%DEBUG_EXE%" (
    "%DEBUG_EXE%" %*
) else (
    echo No build found. Run: dotnet build -c Release
    exit /b 1
)
