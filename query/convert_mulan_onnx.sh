#!/bin/bash
set -e

echo "--- Installing Python dependencies ---"
pip install "numpy<2" torch torchvision torchaudio onnx onnxscript onnxruntime muq transformers sentencepiece

ONNX_EXPORT_DIR="mulan_model_export"
mkdir -p "$ONNX_EXPORT_DIR"

ONNX_AUDIO_MODEL="$ONNX_EXPORT_DIR/mulan_audio_encoder.onnx"
ONNX_TEXT_MODEL="$ONNX_EXPORT_DIR/mulan_text_encoder.onnx"

echo "--- Converting MuQ-MuLan model to ONNX ---"
python -c "
import torch
import torch.nn as nn
import numpy as np
import os
import sys
from muq import MuQMuLan

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

# ---------------------------------------------------------
# 1. Export Tokenizer Files (Critical for removing 'transformers' dependency)
# ---------------------------------------------------------
print('')
print('=' * 70)
print('Exporting Tokenizer Files...')
print('=' * 70)
# Access the underlying text module to get the tokenizer
if hasattr(model, 'mulan_module'):
    text_module = model.mulan_module.text
else:
    text_module = model.text

# Save tokenizer to the export directory
# This saves tokenizer.json, special_tokens_map.json, etc.
output_dir = '$ONNX_EXPORT_DIR'
print(f'Saving tokenizer to {output_dir}...')
text_module.tokenizer.save_pretrained(output_dir)
print('✓ Tokenizer files saved. You can now use the lightweight ''tokenizers'' library instead of ''transformers''.')


print('Creating Audio Encoder Wrapper...')
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

print('Creating Text Encoder Wrapper...')
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
print('')
print('=' * 70)
print('Exporting Audio Encoder to ONNX...')
print('=' * 70)

audio_wrapper = AudioEncoderWrapper(model)
audio_wrapper.eval()

# Create dummy audio input (10 seconds at 24kHz)
# Shape: (batch_size, samples) -> (1, 240000)
dummy_audio = torch.randn(1, 240000)

with torch.no_grad():
    # Verify PyTorch output first
    pt_audio_embed = audio_wrapper(dummy_audio)
    print(f'✓ PyTorch Audio embedding shape: {pt_audio_embed.shape}')

    torch.onnx.export(
        audio_wrapper,
        (dummy_audio,),
        '$ONNX_AUDIO_MODEL',
        export_params=True,
        opset_version=18, # Updated to 18 to fix version conversion error
        do_constant_folding=True,
        input_names=['wavs'],
        output_names=['audio_embedding'],
        dynamic_axes={
            'wavs': {0: 'batch_size', 1: 'samples'},
            'audio_embedding': {0: 'batch_size'}
        },
        dynamo=True # Enable dynamo to handle complex STFT output
    )
print(f'✓ Audio Encoder exported to: $ONNX_AUDIO_MODEL')

# ---------------------------------------------------------
# Export Text Encoder
# ---------------------------------------------------------
print('')
print('=' * 70)
print('Exporting Text Encoder to ONNX...')
print('=' * 70)

text_wrapper = TextEncoderWrapper(model)
text_wrapper.eval()

# Create dummy text input
# T5 tokenizer usually produces input_ids and attention_mask
# Max length 128 is standard for MuLan
dummy_input_ids = torch.randint(0, 32000, (1, 128), dtype=torch.long) # 32000 is T5 vocab size
dummy_attention_mask = torch.ones(1, 128, dtype=torch.long)

with torch.no_grad():
    # Verify PyTorch output first
    pt_text_embed = text_wrapper(dummy_input_ids, dummy_attention_mask)
    print(f'✓ PyTorch Text embedding shape: {pt_text_embed.shape}')

    torch.onnx.export(
        text_wrapper,
        (dummy_input_ids, dummy_attention_mask),
        '$ONNX_TEXT_MODEL',
        export_params=True,
        opset_version=18, # Updated to 18
        do_constant_folding=True,
        input_names=['input_ids', 'attention_mask'],
        output_names=['text_embedding'],
        dynamic_axes={
            'input_ids': {0: 'batch_size', 1: 'sequence_length'},
            'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
            'text_embedding': {0: 'batch_size'}
        },
        dynamo=False # Disable dynamo for text encoder (standard export usually works for transformers)
    )
print(f'✓ Text Encoder exported to: $ONNX_TEXT_MODEL')

# ---------------------------------------------------------
# Verification
# ---------------------------------------------------------
print('')
print('=' * 70)
print('Verifying ONNX models...')
print('=' * 70)

try:
    import onnxruntime as ort
    
    # Verify Audio
    audio_session = ort.InferenceSession('$ONNX_AUDIO_MODEL')
    onnx_audio_out = audio_session.run(None, {'wavs': dummy_audio.numpy()})[0]
    audio_diff = np.abs(pt_audio_embed.numpy() - onnx_audio_out).max()
    print(f'✓ Audio Max Diff (PyTorch vs ONNX): {audio_diff:.2e}')
    
    # Verify Text
    text_session = ort.InferenceSession('$ONNX_TEXT_MODEL')
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

except Exception as e:
    print(f'Verification failed: {e}')

"
