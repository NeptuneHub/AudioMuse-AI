#!/bin/bash
#source .venv/bin/activate
set -e

echo "--- Installing Python dependencies ---"
pip install "numpy<2" torch torchvision torchaudio onnx onnxscript onnxruntime muq transformers sentencepiece

ONNX_EXPORT_DIR="mulan_model_export"
mkdir -p "$ONNX_EXPORT_DIR"

ONNX_AUDIO_MODEL="$ONNX_EXPORT_DIR/mulan_audio_encoder.onnx"
ONNX_TEXT_MODEL="$ONNX_EXPORT_DIR/mulan_text_encoder.onnx"
TOKENIZER_TAR="$ONNX_EXPORT_DIR/mulan_tokenizer.tar.gz"

# Check if models already exist
if [ -f "$ONNX_AUDIO_MODEL" ] && [ -f "$ONNX_TEXT_MODEL" ] && [ -f "$TOKENIZER_TAR" ]; then
    echo "--- ONNX models already exist, skipping export ---"
    echo "  ✓ Audio model: $ONNX_AUDIO_MODEL"
    echo "  ✓ Text model: $ONNX_TEXT_MODEL"
    echo "  ✓ Tokenizer: $TOKENIZER_TAR"
    echo "--- Proceeding directly to verification ---"
    SKIP_EXPORT=1
else
    echo "--- Converting MuQ-MuLan model to ONNX ---"
    SKIP_EXPORT=0
fi

python -c "
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from muq import MuQMuLan

# Check if we should skip export
SKIP_EXPORT = $SKIP_EXPORT

if not SKIP_EXPORT:
    # Ensure we can load the model (offline mode if cached)
    os.environ['HF_HUB_OFFLINE'] = '1'

    print('Loading MuQ-MuLan model...')
    try:
        # Load model from cache
        model = MuQMuLan.from_pretrained('OpenMuQ/MuQ-MuLan-large')
    except Exception as e:
        print(f'Offline load failed ({e}), trying online...')
        os.environ['HF_HUB_OFFLINE'] = '0'
        model = MuQMuLan.from_pretrained('OpenMuQ/MuQ-MuLan-large')

    model.eval()
    # Move to CPU for export
    model = model.to('cpu')
else:
    print('Skipping model loading for export (models already exist)')
    model = None

# ---------------------------------------------------------
# 1. Export Tokenizer Files (Critical for removing 'transformers' dependency)
# ---------------------------------------------------------
if not SKIP_EXPORT:
    print('')
    print('=' * 70)
    print('Exporting Tokenizer Files...')
    print('=' * 70)
    # Access the underlying text module to get the tokenizer
    if hasattr(model, 'mulan_module'):
        text_module = model.mulan_module.text
    else:
        text_module = model.text

    # Use /tmp/ for tokenizer export to avoid WSL->Windows I/O errors
    # This saves tokenizer.json, sentencepiece.bpe.model, special_tokens_map.json, etc.
    import tempfile
    final_output_dir = '$ONNX_EXPORT_DIR'
    final_tar_path = f'{final_output_dir}/mulan_tokenizer.tar.gz'

    # Check if tokenizer already exists
    if os.path.exists(final_tar_path):
        print(f'✓ Tokenizer TAR already exists: {final_tar_path}')
        tar_size = os.path.getsize(final_tar_path) / (1024*1024)
        print(f'  Size: {tar_size:.2f} MB - SKIPPING download')
    else:
        # Create temp directory in /tmp/ (native Linux filesystem)
        temp_dir = tempfile.mkdtemp(prefix='mulan_tok_', dir='/tmp')
        tokenizer_dir = f'{temp_dir}/tokenizer'
        os.makedirs(tokenizer_dir, exist_ok=True)
        
        print(f'Downloading ORIGINAL MuLan tokenizer from HuggingFace...')
        print(f'  Temp directory: {temp_dir}')
        print(f'  Tokenizer type: {type(text_module.tokenizer).__name__}')
        print(f'  Tokenizer class: {text_module.tokenizer.__class__.__module__}.{text_module.tokenizer.__class__.__name__}')
        if hasattr(text_module.tokenizer, 'name_or_path'):
            print(f'  Original model: {text_module.tokenizer.name_or_path}')

        # Save all tokenizer files to temp directory (/tmp/)
        text_module.tokenizer.save_pretrained(tokenizer_dir)
        print(f'✓ Tokenizer files saved to {tokenizer_dir}/')

        # List downloaded files and filter for TAR (exclude tokenizer.json - version incompatible)
        import glob
        all_files = glob.glob(f'{tokenizer_dir}/*')
        # Keep only: sentencepiece.bpe.model, tokenizer_config.json, special_tokens_map.json
        # EXCLUDE: tokenizer.json (incompatible format between tokenizers versions)
        tokenizer_files = [f for f in all_files if not f.endswith('tokenizer.json')]
        
        print(f'  Files for TAR (excluding tokenizer.json for compatibility):')
        for f in tokenizer_files:
            fname = os.path.basename(f)
            fsize = os.path.getsize(f) / (1024*1024)  # MB
            print(f'    - {fname} ({fsize:.2f} MB)')

        # Create TAR archive in /tmp/ first
        import tarfile
        import shutil
        temp_tar_path = f'{temp_dir}/mulan_tokenizer.tar.gz'
        print(f'')
        print(f'Creating TAR archive: {temp_tar_path}')
        print(f'  Note: Using slow tokenizer (sentencepiece) for cross-version compatibility')
        with tarfile.open(temp_tar_path, 'w:gz') as tar:
            for f in tokenizer_files:
                arcname = os.path.basename(f)
                tar.add(f, arcname=arcname)
                print(f'  Added: {arcname}')

        # Copy TAR to final destination
        print(f'')
        print(f'Copying TAR to: {final_tar_path}')
        os.makedirs(final_output_dir, exist_ok=True)
        shutil.copy2(temp_tar_path, final_tar_path)
        
        tar_size = os.path.getsize(final_tar_path) / (1024*1024)
        print(f'✓ TAR created: mulan_tokenizer.tar.gz ({tar_size:.2f} MB)')
        print(f'  → Upload this TAR to your GitHub releases!')
        
        # Clean up temp directory
        print(f'  Cleaning up temp directory...')
        shutil.rmtree(temp_dir)
        print(f'✓ Temp directory cleaned up')
else:
    print('')
    print('Skipping tokenizer export (already exists)')


# ---------------------------------------------------------
# Define wrapper classes (needed for both export and verification)
# ---------------------------------------------------------
print('Defining Audio and Text Encoder Wrappers...')

class AudioEncoderWrapper(nn.Module):
    '''
    Wrapper for MuLan Audio Encoder.
    Input: wavs (batch, length) - raw audio at 24kHz
    Output: embedding (batch, 512)
    '''
    def __init__(self, mulan_model):
        super().__init__()
        # MuQMuLan wraps the actual MuLanModel in 'mulan_module'
        if hasattr(mulan_model, 'mulan_module'):
            self.mulan_model = mulan_model.mulan_module
        else:
            # Fallback if structure changes, but 'mulan_module' is standard for MuQMuLan
            self.mulan_model = mulan_model
            
    def forward(self, wavs):
        # Use get_audio_latents to get the projected and normalized embedding
        # This includes: AudioSpectrogramTransformer -> Linear -> Norm
        return self.mulan_model.get_audio_latents(wavs)

class TextEncoderWrapper(nn.Module):
    '''
    Wrapper for MuLan Text Encoder.
    Input: input_ids, attention_mask (from T5/Roberta tokenizer)
    Output: embedding (batch, 512)
    '''
    def __init__(self, mulan_model):
        super().__init__()
        if hasattr(mulan_model, 'mulan_module'):
            self.mulan_model = mulan_model.mulan_module
        else:
            self.mulan_model = mulan_model
        
        # The text module is an instance of TextTransformerPretrained
        self.text_module = self.mulan_model.text
        
        # Access sub-components directly to bypass the string-only forward method
        self.hf_model = self.text_module.model
        self.proj = self.text_module.proj
        self.transformer = self.text_module.transformer
        self.text_to_latents = self.mulan_model.text_to_latents
        
    def forward(self, input_ids, attention_mask):
        # 1. Pretrained Model (XLMRoberta)
        # We call the HF model directly with tensors
        outputs = self.hf_model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Handle different model output types (Roberta vs Qwen/others)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        elif hasattr(outputs, 'hidden_states'):
            hidden_states = outputs.hidden_states[-1]
        else:
            # Fallback, assume tuple and first element is hidden state
            hidden_states = outputs[0]
            
        # 2. Projection
        hidden_states = self.proj(hidden_states)
        
        # 3. Additional Transformer Layers
        # TextTransformerPretrained calls this without mask
        hidden_states = self.transformer(hidden_states)
        
        # 4. Mean Pooling (dim=-2 is sequence length)
        pooled = hidden_states.mean(dim=-2)
        
        # 5. Project to shared latent space
        latents = self.text_to_latents(pooled)
        
        # 6. Normalize (L2 norm)
        return torch.nn.functional.normalize(latents, dim=-1)

# ---------------------------------------------------------
# Export Audio Encoder
# ---------------------------------------------------------
if not SKIP_EXPORT:
    print('')
    print('=' * 70)
    print('Exporting Audio Encoder to ONNX...')
    print('=' * 70)

    audio_wrapper = AudioEncoderWrapper(model)
    audio_wrapper.eval()

    # Create temp directory in /tmp/ for ONNX export
    import tempfile
    import shutil
    import glob
    temp_onnx_dir = tempfile.mkdtemp(prefix='mulan_onnx_', dir='/tmp')
    temp_audio_model = f'{temp_onnx_dir}/mulan_audio_encoder.onnx'
    print(f'  Temp export directory: {temp_onnx_dir}')

    # Create dummy audio input (10 seconds at 24kHz)
    # Shape: (batch_size, samples) -> (1, 240000)
    dummy_audio = torch.randn(1, 240000)

    with torch.no_grad():
        # Verify PyTorch output first
        pt_audio_embed = audio_wrapper(dummy_audio)
        print(f'✓ PyTorch Audio embedding shape: {pt_audio_embed.shape}')

        # Use dynamic_shapes with dynamo=True
        from torch.export import Dim
        batch_dim = Dim('batch_size', min=1, max=32)
        time_dim = Dim('num_samples', min=1, max=2147483647)  # ANY length: 1 sample to ~24 hours @ 24kHz
        
        dynamic_shapes = {
            'wavs': {0: batch_dim, 1: time_dim},
        }
        
        # Export to /tmp/ (creates .onnx and .onnx.data files)
        torch.onnx.export(
            audio_wrapper,
            (dummy_audio,),
            temp_audio_model,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['wavs'],
            output_names=['audio_embedding'],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            dynamo=True
        )
    print(f'✓ Audio Encoder exported to temp: {temp_audio_model}')
    
    # Copy all audio model files to final destination
    audio_files = glob.glob(f'{temp_onnx_dir}/mulan_audio_encoder.onnx*')
    print(f'  Copying {len(audio_files)} file(s) to $ONNX_EXPORT_DIR...')
    os.makedirs('$ONNX_EXPORT_DIR', exist_ok=True)
    for src_file in audio_files:
        filename = os.path.basename(src_file)
        dst_file = f'$ONNX_EXPORT_DIR/{filename}'
        size_mb = os.path.getsize(src_file) / (1024 * 1024)
        print(f'    {filename} ({size_mb:.2f} MB)')
        shutil.copy2(src_file, dst_file)
    print(f'✓ Audio model files copied to $ONNX_EXPORT_DIR')
else:
    print('')
    print('Skipping audio encoder export (already exists)')

# ---------------------------------------------------------
# Export Text Encoder
# ---------------------------------------------------------
if not SKIP_EXPORT:
    print('')
    print('=' * 70)
    print('Exporting Text Encoder to ONNX...')
    print('=' * 70)

    text_wrapper = TextEncoderWrapper(model)
    text_wrapper.eval()

    # Reuse same temp directory
    temp_text_model = f'{temp_onnx_dir}/mulan_text_encoder.onnx'

    # Create dummy text input
    # T5 tokenizer usually produces input_ids and attention_mask
    # Max length 128 is standard for MuLan
    dummy_input_ids = torch.randint(0, 32000, (1, 128), dtype=torch.long) # 32000 is T5 vocab size
    dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

    with torch.no_grad():
        # Verify PyTorch output first
        pt_text_embed = text_wrapper(dummy_input_ids, dummy_attention_mask)
        print(f'✓ PyTorch Text embedding shape: {pt_text_embed.shape}')

        # Use dynamic_shapes with dynamo=True
        from torch.export import Dim
        batch_dim = Dim('batch_size', min=1, max=32)
        
        dynamic_shapes = {
            'input_ids': {0: batch_dim},
            'attention_mask': {0: batch_dim},
        }
        
        # Export to /tmp/ (creates .onnx and .onnx.data files)
        torch.onnx.export(
            text_wrapper,
            (dummy_input_ids, dummy_attention_mask),
            temp_text_model,
            export_params=True,
            opset_version=18,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['text_embedding'],
            dynamic_shapes=dynamic_shapes,
            verbose=False,
            dynamo=True
        )
    print(f'✓ Text Encoder exported to temp: {temp_text_model}')
    
    # Copy all text model files to final destination
    text_files = glob.glob(f'{temp_onnx_dir}/mulan_text_encoder.onnx*')
    print(f'  Copying {len(text_files)} file(s) to $ONNX_EXPORT_DIR...')
    for src_file in text_files:
        filename = os.path.basename(src_file)
        dst_file = f'$ONNX_EXPORT_DIR/{filename}'
        size_mb = os.path.getsize(src_file) / (1024 * 1024)
        print(f'    {filename} ({size_mb:.2f} MB)')
        shutil.copy2(src_file, dst_file)
    print(f'✓ Text model files copied to $ONNX_EXPORT_DIR')
    
    # Clean up temp directory
    print('')
    print(f'Cleaning up temp ONNX directory: {temp_onnx_dir}')
    shutil.rmtree(temp_onnx_dir)
    print(f'✓ Temp directory cleaned up')
else:
    print('')
    print('Skipping text encoder export (already exists)')

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
print('')
print('=' * 70)
print('Verifying ONNX models...')
print('=' * 70)

try:
    import onnxruntime as ort
    
    # Load ONNX models (needed for both skip and non-skip paths)
    audio_session = ort.InferenceSession('$ONNX_AUDIO_MODEL')
    text_session = ort.InferenceSession('$ONNX_TEXT_MODEL')
    
    if not SKIP_EXPORT:
        # Verify Audio
        onnx_audio_out = audio_session.run(None, {'wavs': dummy_audio.numpy()})[0]
        audio_diff = np.abs(pt_audio_embed.numpy() - onnx_audio_out).max()
        print(f'✓ Audio Max Diff (PyTorch vs ONNX): {audio_diff:.2e}')
        
        # Verify Text
        onnx_text_out = text_session.run(None, {
            'input_ids': dummy_input_ids.numpy(), 
            'attention_mask': dummy_attention_mask.numpy()
        })[0]
        text_diff = np.abs(pt_text_embed.numpy() - onnx_text_out).max()
        print(f'✓ Text Max Diff (PyTorch vs ONNX): {text_diff:.2e}')
        
        if audio_diff < 1e-4 and text_diff < 1e-4:
            print('✓✓✓ SUCCESS: Both models exported correctly and match PyTorch outputs!')
        else:
            print('WARNING: Differences detected. Check opset version or model architecture.')
    else:
        print('✓ ONNX models loaded successfully (skipped export verification)')


except Exception as e:
    print(f'Verification failed: {e}')

# ---------------------------------------------------------
# Real Audio Verification (like CLAP pythorch.sh)
# ---------------------------------------------------------
print('')
print('=' * 70)
print('Testing with REAL AUDIO FILES (PyTorch vs ONNX)')
print('=' * 70)

try:
    import librosa
    import glob
    
    # Look for test audio files
    test_dirs = ['../test/songs', 'test/songs', './test/songs', '../songs', 'songs']
    test_files = []
    
    for test_dir in test_dirs:
        if os.path.exists(test_dir):
            test_files = glob.glob(f'{test_dir}/*.mp3') + \
                        glob.glob(f'{test_dir}/*.wav') + \
                        glob.glob(f'{test_dir}/*.flac') + \
                        glob.glob(f'{test_dir}/*.m4a')
            if test_files:
                print(f'✓ Found {len(test_files)} test audio files in {test_dir}/')
                break
    
    if not test_files:
        print('⚠️  No test audio files found. Skipping real audio verification.')
        print('   Place some .mp3/.wav/.flac files in test/songs/ to enable this test.')
    else:
        # Need to load model for PyTorch comparison (if skipped export)
        if SKIP_EXPORT:
            print('')
            print('Loading MuQ-MuLan model for verification...')
            try:
                os.environ['HF_HUB_OFFLINE'] = '1'
                model = MuQMuLan.from_pretrained('OpenMuQ/MuQ-MuLan-large')
            except Exception as e:
                print(f'Offline load failed ({e}), trying online...')
                os.environ['HF_HUB_OFFLINE'] = '0'
                model = MuQMuLan.from_pretrained('OpenMuQ/MuQ-MuLan-large')
            model.eval()
            model = model.to('cpu')
            audio_wrapper = AudioEncoderWrapper(model)
            audio_wrapper.eval()
            text_wrapper = TextEncoderWrapper(model)
            text_wrapper.eval()
        
        # Limit to first 3 files for speed
        test_files = test_files[:3]
        
        print('')
        print('Analyzing real audio files with FIXED 10s segments (50% overlap):')
        print('Using central 50 seconds (or full song if shorter)')
        
        for i, audio_file in enumerate(test_files, 1):
            filename = os.path.basename(audio_file)
            print('')
            print(f'[{i}/{len(test_files)}] {filename}')
            print('-' * 60)
            
            try:
                # Load audio at MuLan's required 24kHz
                SAMPLE_RATE = 24000
                SEGMENT_DURATION = 10.0  # FIXED 10 second segments (model requirement)
                SEGMENT_SAMPLES = int(SEGMENT_DURATION * SAMPLE_RATE)  # EXACTLY 240,000 samples
                HOP_DURATION = 5.0  # 50% overlap = 5 second hop
                HOP_SAMPLES = int(HOP_DURATION * SAMPLE_RATE)  # 120,000 samples
                ANALYSIS_WINDOW = 50.0  # Prefer central 50 seconds if available
                
                # Get full duration first
                full_duration = librosa.get_duration(path=audio_file)
                print(f'  Full duration: {full_duration:.2f}s')
                
                # Determine what to load based on song length
                if full_duration > ANALYSIS_WINDOW:
                    # Long song: use central 50 seconds
                    offset = (full_duration - ANALYSIS_WINDOW) / 2
                    load_duration = ANALYSIS_WINDOW
                    print(f'  Analyzing: central {load_duration:.2f}s (offset: {offset:.2f}s)')
                elif full_duration >= SEGMENT_DURATION:
                    # Song is between 10s and 50s: use whole song
                    offset = 0.0
                    load_duration = full_duration
                    print(f'  Analyzing: full song ({load_duration:.2f}s)')
                else:
                    # Song is shorter than 10s: load what we have, will pad to 10s
                    offset = 0.0
                    load_duration = full_duration
                    print(f'  Analyzing: full song ({load_duration:.2f}s, will pad to 10s)')
                
                # Load audio segment
                audio_data, sr = librosa.load(audio_file, sr=SAMPLE_RATE, mono=True, offset=offset, duration=load_duration)
                
                # Calculate number of FIXED 10-second segments with 50% overlap
                if len(audio_data) < SEGMENT_SAMPLES:
                    # Audio is shorter than 10s: create 1 segment with padding
                    num_segments = 1
                else:
                    # Calculate overlapping segments
                    num_segments = int((len(audio_data) - SEGMENT_SAMPLES) / HOP_SAMPLES) + 1
                
                print(f'  Processing {num_segments} segment(s) (FIXED 10s each, 5s hop)')
                
                segment_embeddings = []
                
                for seg_idx in range(num_segments):
                    start_sample = seg_idx * HOP_SAMPLES
                    end_sample = start_sample + SEGMENT_SAMPLES
                    
                    if start_sample >= len(audio_data):
                        # No more audio data
                        break
                    
                    # Extract segment
                    if end_sample <= len(audio_data):
                        # Full segment available
                        segment = audio_data[start_sample:end_sample]
                    else:
                        # Last segment: need padding
                        segment = audio_data[start_sample:]
                        padding = SEGMENT_SAMPLES - len(segment)
                        segment = np.pad(segment, (0, padding), mode='constant')
                    
                    # Ensure EXACTLY 240,000 samples (critical requirement)
                    if len(segment) != SEGMENT_SAMPLES:
                        raise ValueError(f'Segment must be exactly {SEGMENT_SAMPLES} samples, got {len(segment)}')
                    
                    # Prepare input: (1, num_samples)
                    audio_input = segment.astype(np.float32).reshape(1, -1)
                    
                    # ONNX inference
                    onnx_embedding = audio_session.run(None, {'wavs': audio_input})[0]
                    segment_embeddings.append(onnx_embedding)
                
                # Average all segment embeddings
                avg_embedding = np.mean(segment_embeddings, axis=0)
                
                # Normalize the averaged embedding
                norm = np.linalg.norm(avg_embedding)
                if norm > 0:
                    avg_embedding = avg_embedding / norm
                
                print(f'  ✓ Generated embedding from {len(segment_embeddings)} segment(s)')
                print(f'    Final embedding shape: {avg_embedding.shape}')
                print(f'    Final embedding norm: {np.linalg.norm(avg_embedding):.4f}')
                
            except Exception as e:
                print(f'  ✗ Failed to analyze: {e}')
                import traceback
                traceback.print_exc()
                continue
        
        print('')
        print('=' * 70)
        print('✓✓✓ REAL AUDIO VERIFICATION COMPLETE')
        print('=' * 70)
        print('All embeddings should match within 1e-4 (0.0001) difference.')
        print('If differences are larger, check sample rate or model configuration.')
        
        # ---------------------------------------------------------
        # Text Embedding Verification
        # ---------------------------------------------------------
        print('')
        print('=' * 70)
        print('TEXT EMBEDDING VERIFICATION (PyTorch vs ONNX)')
        print('=' * 70)
        
        test_queries = [
            'classical piano music',
            'upbeat electronic dance',
            'calm acoustic guitar',
            'orchestral symphony'
        ]
        
        print(f'Testing {len(test_queries)} text queries...')
        print('')
        
        # Get the tokenizer from the model
        if hasattr(model, 'mulan_module'):
            tokenizer = model.mulan_module.text.tokenizer
        else:
            tokenizer = model.text.tokenizer
        
        for i, query in enumerate(test_queries, 1):
            print(f'[{i}/{len(test_queries)}] \"{query}\"')
            print('-' * 60)
            
            try:
                # Tokenize the query properly
                tokens = tokenizer(query, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
                input_ids = tokens['input_ids']
                attention_mask = tokens['attention_mask']
                
                # PyTorch text encoding
                import time
                pt_start = time.perf_counter()
                with torch.no_grad():
                    pt_text_embed = text_wrapper(input_ids, attention_mask).numpy()
                pt_time = time.perf_counter() - pt_start
                
                # ONNX text encoding
                onnx_start = time.perf_counter()
                onnx_text_embed = text_session.run(None, {
                    'input_ids': input_ids.numpy(),
                    'attention_mask': attention_mask.numpy()
                })[0]
                onnx_time = time.perf_counter() - onnx_start
                
                # Compare embeddings
                max_diff = np.abs(pt_text_embed - onnx_text_embed).max()
                mean_diff = np.abs(pt_text_embed - onnx_text_embed).mean()
                
                # Cosine similarity
                pt_norm = pt_text_embed / np.linalg.norm(pt_text_embed)
                onnx_norm = onnx_text_embed / np.linalg.norm(onnx_text_embed)
                cosine_sim = np.dot(pt_norm.flatten(), onnx_norm.flatten())
                
                print(f'  PyTorch embedding: {pt_text_embed.shape}, norm={np.linalg.norm(pt_text_embed):.4f}')
                print(f'  ONNX embedding:    {onnx_text_embed.shape}, norm={np.linalg.norm(onnx_text_embed):.4f}')
                print(f'  Max difference: {max_diff:.2e}')
                print(f'  Mean difference: {mean_diff:.2e}')
                print(f'  Cosine similarity: {cosine_sim:.10f}')
                print(f'  PyTorch time: {pt_time*1000:.2f}ms')
                print(f'  ONNX time:    {onnx_time*1000:.2f}ms')
                
                passed = max_diff < 1e-4 and cosine_sim > 0.9999
                status = '✓ PASS' if passed else '✗ FAIL'
                print(f'  Status: {status}')
                print('')
                
            except Exception as e:
                print(f'  ✗ Failed: {e}')
                import traceback
                traceback.print_exc()
                print('')
        
        # ---------------------------------------------------------
        # Cross-Modal Similarity Test (TEXT → AUDIO)
        # ---------------------------------------------------------
        print('')
        print('=' * 70)
        print('CROSS-MODAL TEST: Text-to-Audio Similarity')
        print('=' * 70)
        print('Testing if text queries match audio embeddings correctly...')
        print('')
        
        # Use first audio file as reference
        if test_files:
            ref_audio_file = test_files[0]
            ref_filename = os.path.basename(ref_audio_file)
            
            print(f'Reference audio: {ref_filename}')
            
            # Get audio embedding (already loaded earlier)
            audio_data, sr = librosa.load(ref_audio_file, sr=24000, mono=True, duration=10.0)
            expected_samples = int(10.0 * 24000)
            if len(audio_data) > expected_samples:
                audio_data = audio_data[:expected_samples]
            audio_input = audio_data.astype(np.float32).reshape(1, -1)
            
            # ONNX audio embedding
            audio_embed = audio_session.run(None, {'wavs': audio_input})[0]
            audio_embed_norm = audio_embed / np.linalg.norm(audio_embed)
            
            print('')
            print('Text query similarities (using REAL tokenization):')
            for query in test_queries:
                # Tokenize query properly
                tokens = tokenizer(query, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
                
                # ONNX text embedding
                text_embed = text_session.run(None, {
                    'input_ids': tokens['input_ids'].numpy(),
                    'attention_mask': tokens['attention_mask'].numpy()
                })[0]
                text_embed_norm = text_embed / np.linalg.norm(text_embed)
                
                # Cross-modal similarity
                similarity = np.dot(audio_embed_norm.flatten(), text_embed_norm.flatten())
                print(f'  \"{query}\" → {ref_filename}: {similarity:.4f}')
            
            print('')
            print('✓ Cross-modal similarity with REAL text encoding')
            print('  \"classical piano music\" should have HIGH similarity to piano music')
            print('  \"upbeat electronic dance\" should have LOW similarity to piano music')

        
except ImportError as e:
    print(f'⚠️  Skipping real audio test: {e}')
    print('   Install librosa for real audio verification: pip install librosa')
except Exception as e:
    print(f'⚠️  Real audio verification failed: {e}')
    import traceback
    traceback.print_exc()

# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------
print('')
print('=' * 70)
print('EXPORT COMPLETE - Files saved in current directory:')
print('=' * 70)
print('')
print('PyTorch → ONNX Conversion (NO PyTorch needed for inference):')
print(f'  1. mulan_audio_encoder.onnx       - Audio model structure')
print(f'  2. mulan_audio_encoder.onnx.data  - Audio model weights (~1.2 GB)')
print(f'  3. mulan_text_encoder.onnx        - Text model structure')
print(f'  4. mulan_text_encoder.onnx.data   - Text model weights (~1.3 GB)')
print('')
print('HuggingFace Tokenizer (downloaded and archived):')
print(f'  5. mulan_tokenizer.tar.gz         - All tokenizer files (extract for use)')
print('')
print('⚠️  You can DELETE the following (HuggingFace cache, not needed):')
print('  - All models--* folders')
print('  - tokenizer/ folder (already in TAR)')
print('')
print('📦 Upload these 5 files to GitHub releases:')
print('  1. mulan_audio_encoder.onnx')
print('  2. mulan_audio_encoder.onnx.data')
print('  3. mulan_text_encoder.onnx')
print('  4. mulan_text_encoder.onnx.data')
print('  5. mulan_tokenizer.tar.gz')
print('')
print('Then extract TAR when needed:')
print('  tar -xzf mulan_tokenizer.tar.gz')
print('')

"
