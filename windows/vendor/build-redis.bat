@echo off
REM ============================================================================
REM Build/download the vendored redis-server.exe for Windows.
REM
REM Uses the tporadowski/redis fork (Redis 5.0 for Windows) which supports
REM variadic HSET — required by the Python redis 7.x client. The Microsoft
REM Archive Redis 3.2 is too old (HSET only accepts 3 arguments).
REM ============================================================================
setlocal enabledelayedexpansion

set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"

set "DEST=windows\vendor\redis\%ARCH%"
set "REDIS_VERSION=5.0.14.1"
set "ZIP=Redis-x64-%REDIS_VERSION%.zip"
set "URL=https://github.com/tporadowski/redis/releases/download/v%REDIS_VERSION%/%ZIP%"

if not exist "%DEST%" mkdir "%DEST%"

if exist "%DEST%\redis-server.exe" (
    for %%A in ("%DEST%\redis-server.exe") do if %%~zA gtr 100000 (
        echo redis-server.exe already exists at %DEST%\redis-server.exe -- skipping
        exit /b 0
    )
)

echo Downloading Redis %REDIS_VERSION% for Windows...
curl -fsSL -o "%TEMP%\%ZIP%" "%URL%"
if errorlevel 1 (
    echo ::error::Failed to download Redis for Windows from %URL%
    exit /b 1
)

echo Extracting redis-server.exe...
powershell -Command "Expand-Archive -Path '%TEMP%\%ZIP%' -DestinationPath '%TEMP%\redis-extract' -Force"
if errorlevel 1 (
    echo ::error::Failed to extract Redis zip
    exit /b 1
)

for /r "%TEMP%\redis-extract" %%f in (redis-server.exe) do (
    copy /y "%%f" "%DEST%\redis-server.exe" >nul
    goto :found
)

echo ::error::redis-server.exe not found in extracted zip
exit /b 1

:found
echo ==^> redis-server.exe placed at %DEST%\redis-server.exe
del "%TEMP%\%ZIP%" 2>nul
rmdir /s /q "%TEMP%\redis-extract" 2>nul
