@echo off
REM ============================================================================
REM Build/download the vendored redis-server.exe for Windows.
REM
REM Redis does not officially support Windows. We use the Microsoft Archive
REM Redis for Windows (https://github.com/microsoftarchive/redis).
REM
REM This script downloads the MS Open Tech Redis release for Windows.
REM If the download fails, check the URL and update the version as needed.
REM ============================================================================
setlocal enabledelayedexpansion

set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"

set "DEST=windows\vendor\redis\%ARCH%"
set "REDIS_VERSION=3.2.100"

if not exist "%DEST%" mkdir "%DEST%"

if exist "%DEST%\redis-server.exe" (
    echo redis-server.exe already exists at %DEST%\redis-server.exe -- skipping
    exit /b 0
)

REM Microsoft Archive Redis for Windows (MSI-based):
REM We download the MSI, extract redis-server.exe from it, and discard the rest.
set "MSI=Redis-x64-%REDIS_VERSION%.msi"
set "URL=https://github.com/microsoftarchive/redis/releases/download/win-%REDIS_VERSION%/%MSI%"

echo Downloading Redis %REDIS_VERSION% for Windows...
curl -fsSL -o "%TEMP%\%MSI%" "%URL%"
if errorlevel 1 (
    echo ::error::Failed to download Redis for Windows from %URL%
    echo ::notice::Download the MSI manually from https://github.com/microsoftarchive/redis/releases
    echo ::notice::and extract redis-server.exe to %DEST%
    exit /b 1
)

REM Extract redis-server.exe from the MSI using msiexec admin install
echo Extracting redis-server.exe...
msiexec /a "%TEMP%\%MSI%" /qb TARGETDIR="%TEMP%\redis-extract"
if errorlevel 1 (
    echo ::error::Failed to extract MSI
    exit /b 1
)

REM Find and copy redis-server.exe
for /r "%TEMP%\redis-extract" %%f in (redis-server.exe) do (
    copy /y "%%f" "%DEST%\redis-server.exe"
    goto :found
)

echo ::error::redis-server.exe not found in extracted MSI
exit /b 1

:found
echo ==^> redis-server.exe placed at %DEST%\redis-server.exe
del "%TEMP%\%MSI%" 2>nul
rmdir /s /q "%TEMP%\redis-extract" 2>nul
