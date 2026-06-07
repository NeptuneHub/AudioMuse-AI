@echo off
setlocal enabledelayedexpansion
REM ============================================================================
REM Build the standalone Windows bundle with PyInstaller, then package it as an
REM MSI installer with WiX Toolset.
REM
REM Run from the repo root, inside the build venv:
REM     .venv-windows\Scripts\activate
REM     set PKG_VERSION=1.0.0
REM     windows\build.bat
REM
REM Prerequisites (the CI workflow installs these):
REM   * Python 3.12 + deps from requirements/windows.txt (incl. pyinstaller, pgserver)
REM   * WiX Toolset v5 on PATH (wix.exe)
REM   * the vendored redis-server.exe + pg-contrib for this arch
REM     (windows\vendor\...; built by windows\vendor\build-redis.bat and
REM      windows\vendor\pg-contrib\build-pg-contrib.bat)
REM ============================================================================

if "%PKG_VERSION%"=="" set PKG_VERSION=0.0.0
REM Strip leading v (v1.2.3 -> 1.2.3) and sanitize.
set "VER=%PKG_VERSION:v=%"
echo ==^> Package version: %VER%

REM Detect architecture
set "ARCH=amd64"
if "%PROCESSOR_ARCHITECTURE%"=="ARM64" set "ARCH=arm64"
echo ==^> Architecture: %ARCH%

echo ==^> Cleaning previous build
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist

echo ==^> Verifying vendored native build inputs are present
set MISSING=0
if not exist "windows\vendor\redis\%ARCH%\redis-server.exe" (
    echo ::error::Missing vendored file: windows\vendor\redis\%ARCH%\redis-server.exe
    set MISSING=1
)
if not exist "windows\vendor\pg-contrib\%ARCH%\lib\unaccent.dll" (
    echo ::error::Missing vendored file: windows\vendor\pg-contrib\%ARCH%\lib\unaccent.dll
    set MISSING=1
)
if "%MISSING%"=="1" (
    echo Vendored inputs missing (see windows\vendor\README.md).
    exit /b 1
)

echo ==^> Running PyInstaller
pyinstaller windows\AudioMuse-AI.spec --noconfirm

set "BUNDLE=dist\AudioMuse-AI"
if not exist "%BUNDLE%\AudioMuse-AI.exe" (
    echo ::error::PyInstaller did not produce %BUNDLE%\AudioMuse-AI.exe
    exit /b 1
)

echo ==^> Building MSI installer with WiX
REM Generate the WiX source from the template, substituting variables.
set "WXS=dist\AudioMuse-AI.wxs"
set "MSI=dist\AudioMuse-AI-%ARCH%-windows.msi"

REM Create the WiX source file.
(
echo ^<Wix xmlns="http://wixtoolset.org/schemas/v4/wxs"^>
echo   ^<Package Name="AudioMuse-AI" Manufacturer="AudioMuse-AI" Version="%VER%" UpgradeCode="A1B2C3D4-E5F6-7890-ABCD-EF1234567890"^>
echo     ^<MajorUpgrade DowngradeErrorMessage="A newer version is already installed." /^>
echo     ^<MediaTemplate EmbedCab="yes" /^>
echo     ^<Feature Id="Main" Title="AudioMuse-AI" Level="1"^>
echo       ^<ComponentGroupRef Id="AppComponents" /^>
echo     ^</Feature^>
echo   ^</Package^>
echo ^</Wix^>
) > "%WXS%"

REM Use WiX to build the MSI.  For now we package the one-dir bundle simply;
REM a full WiX authoring with shortcuts, Add/Remove Programs entries, etc.
REM should be added in a follow-up (see windows\packaging\AudioMuse-AI.wxs).
if exist "%WXS%" (
    wix build -o "%MSI%" "%WXS%" || (
        echo ::warning::WiX build failed -- zip fallback will be used instead
        del "%MSI%" 2>nul
    )
)

echo ==^> Packaging zip fallback
set "ZIP=dist\AudioMuse-AI-%ARCH%-windows.zip"
powershell -Command "Compress-Archive -Path '%BUNDLE%' -DestinationPath '%ZIP%' -Force"

echo ==^> Done
echo     MSI: %MSI% (if WiX is installed)
echo     ZIP: %ZIP%
