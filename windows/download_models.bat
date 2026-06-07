@echo off
setlocal enabledelayedexpansion
REM Download all AudioMuse-AI ML models from GitHub releases
REM Total: ~2.4 GB compressed, ~5 GB extracted

set MODEL_REL=v5.0.0-model
set DCLAP_REL=v1
set BASE=https://github.com/NeptuneHub/AudioMuse-AI/releases/download/%MODEL_REL%
set DCLAP=https://github.com/NeptuneHub/AudioMuse-AI-DCLAP/releases/download/%DCLAP_REL%

if not exist model mkdir model

echo ============================================================
echo Downloading AudioMuse-AI models (~2.4 GB total)
echo ============================================================

echo.
echo [1/9] musicnn_embedding.onnx (~5 MB)
curl -fsSL -o model\musicnn_embedding.onnx %BASE%/musicnn_embedding.onnx
if errorlevel 1 exit /b 1

echo [2/9] musicnn_prediction.onnx (~19 MB)
curl -fsSL -o model\musicnn_prediction.onnx %BASE%/musicnn_prediction.onnx
if errorlevel 1 exit /b 1

echo [3/9] clap_text_model.onnx (~478 MB)
curl -fsSL -o model\clap_text_model.onnx %BASE%/clap_text_model.onnx
if errorlevel 1 exit /b 1

echo [4/9] model_epoch_36.onnx (~1 MB)
curl -fsSL -o model\model_epoch_36.onnx %DCLAP%/model_epoch_36.onnx
if errorlevel 1 exit /b 1

echo [5/9] model_epoch_36.onnx.data (~20 MB)
curl -fsSL -o model\model_epoch_36.onnx.data %DCLAP%/model_epoch_36.onnx.data
if errorlevel 1 exit /b 1

echo [6/9] huggingface_models.tar.gz (~985 MB)
curl -fsSL -o model\huggingface_models.tar.gz %BASE%/huggingface_models.tar.gz
if errorlevel 1 exit /b 1

echo [7/9] lyrics_model_whisper.tar.gz (~570 MB)
curl -fsSL -o model\lyrics_model_whisper.tar.gz %BASE%/lyrics_model_whisper.tar.gz
if errorlevel 1 exit /b 1

echo [8/9] lyrics_model_silero_vad.tar.gz (~2 MB)
curl -fsSL -o model\lyrics_model_silero_vad.tar.gz %BASE%/lyrics_model_silero_vad.tar.gz
if errorlevel 1 exit /b 1

echo [9/9] lyrics_model_gte_vnni.tar.gz (~325 MB)
curl -fsSL -o model\lyrics_model_gte_vnni.tar.gz %BASE%/lyrics_model_gte_vnni.tar.gz
if errorlevel 1 exit /b 1

echo.
echo ============================================================
echo All downloads complete. Now extracting tarballs...
echo ============================================================

echo Extracting huggingface models...
mkdir model\huggingface 2>nul
tar -xzf model\huggingface_models.tar.gz -C model\huggingface
del model\huggingface_models.tar.gz

echo Trimming HF cache to roberta-base tokenizer only...
if exist "model\huggingface\hub\models--bert-base-uncased" rmdir /s /q "model\huggingface\hub\models--bert-base-uncased"
if exist "model\huggingface\hub\models--facebook--bart-base" rmdir /s /q "model\huggingface\hub\models--facebook--bart-base"

echo Extracting whisper models...
tar -xzf model\lyrics_model_whisper.tar.gz -C model
del model\lyrics_model_whisper.tar.gz

echo Extracting silero VAD model...
tar -xzf model\lyrics_model_silero_vad.tar.gz -C model
del model\lyrics_model_silero_vad.tar.gz

echo Extracting GTE model...
tar -xzf model\lyrics_model_gte_vnni.tar.gz -C model
del model\lyrics_model_gte_vnni.tar.gz

echo.
echo ============================================================
echo ALL DONE - Models are ready in .\model\
echo ============================================================
dir /s model\*.onnx 2>nul
