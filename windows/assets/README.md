# Windows assets directory

Place the application icon here (``AudioMuse-AI.ico``) for the PyInstaller build.

Generate the .ico from the project's main icon:
1. Use ``screenshot/audiomuseai.png`` as the source
2. Convert to .ico with multiple resolutions (16, 32, 48, 256)
3. Place as ``windows/assets/AudioMuse-AI.ico``

The macOS build's ``make_icns.sh`` can serve as reference for the conversion.
