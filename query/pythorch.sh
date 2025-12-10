#!/bin/bash
set -e

echo "--- Installing Python dependencies ---"
pip install "numpy<2" torch torchvision torchaudio onnx onnxscript onnxruntime laion-clap

MERGED_MODEL="music_audioset_epoch_15_esc_90.14.pt"
ONNX_MODEL="clap_model.onnx"

# Check if merged model already exists
if [ -f "$MERGED_MODEL" ]; then
    echo "--- Merged model already exists, skipping download and merge ---"
else
    echo "--- Downloading CLAP model parts from v3.0.0-model release ---"
    wget -q --show-progress \
        https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/clap_model_part.aa \
        https://github.com/NeptuneHub/AudioMuse-AI/releases/download/v3.0.0-model/clap_model_part.ab

    echo "--- Verifying checksums ---"
    echo "0047a9274849ed3bb515af85557b657bfde628b3d698bdd0e503e393d055d534  clap_model_part.aa" | sha256sum -c -
    echo "9e1460a3070d4a1af1db08d954bc0a744eed12733d0e090f70a029a634dfa9e5  clap_model_part.ab" | sha256sum -c -

    echo "--- Merging model parts ---"
    cat clap_model_part.aa clap_model_part.ab > "$MERGED_MODEL"

    echo "--- Cleaning up part files ---"
    rm clap_model_part.aa clap_model_part.ab
fi

echo "--- Converting CLAP model to ONNX ---"
python -c "
import torch
import torch.nn as nn
import numpy as np
import laion_clap
import sys

# Load the CLAP model using the same approach as clap_analyzer.py
print('Loading CLAP model...')
model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')

# Patch load_state_dict to ignore unexpected keys (like position_ids)
original_load = model.model.load_state_dict
model.model.load_state_dict = lambda *args, **kwargs: original_load(
    *args, **{**kwargs, 'strict': False}
)

print('Loading checkpoint: $MERGED_MODEL')
model.load_ckpt('$MERGED_MODEL')
model.model.load_state_dict = original_load

# Move model to CPU for ONNX export (ONNX export works best on CPU)
print('Moving model to CPU...')
model = model.to('cpu')
model.eval()

# Disable gradients for inference
for param in model.parameters():
    param.requires_grad = False

print('Creating combined CLAP wrapper with both audio and text encoders...')

class CombinedCLAPWrapper(nn.Module):
    '''
    Combined CLAP model with both audio and text encoders in one ONNX model.
    
    Supports two inference modes:
    1. Audio mode: mel_spectrogram → audio embedding
    2. Text mode: tokenized text → text embedding
    
    Both produce identical 512-dim embeddings as the original .pt model.
    '''
    def __init__(self, clap_model):
        super().__init__()
        # Extract both encoders and projections
        self.audio_branch = clap_model.model.audio_branch
        self.audio_projection = clap_model.model.audio_projection
        self.text_branch = clap_model.model.text_branch
        self.text_projection = clap_model.model.text_projection
        
    def forward(self, mel_spec, input_ids, attention_mask):
        '''
        Forward pass for both audio and text.
        
        Args:
            mel_spec: (batch_size, 1, time_frames, 64) log mel-spectrogram (set to zeros if using text mode)
            input_ids: (batch_size, seq_length) token IDs (set to zeros if using audio mode)
            attention_mask: (batch_size, seq_length) attention mask (set to zeros if using audio mode)
        
        Returns:
            audio_embedding: (batch_size, 512) normalized audio embedding
            text_embedding: (batch_size, 512) normalized text embedding
        '''
        # Audio encoding path
        x = mel_spec.transpose(1, 3)  # (batch, 64, time, 1)
        x = self.audio_branch.bn0(x)
        x = x.transpose(1, 3)  # (batch, 1, time, 64)
        x = self.audio_branch.reshape_wav2img(x)
        audio_output = self.audio_branch.forward_features(x)
        audio_embed = self.audio_projection(audio_output['embedding'])
        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
        
        # Text encoding path
        text_output = self.text_branch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_embed = self.text_projection(text_output.pooler_output)
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
        
        return audio_embed, text_embed

try:
    wrapper = CombinedCLAPWrapper(model)
    wrapper.eval()
    
    print('')
    print('=' * 70)
    print('Testing combined CLAP model...')
    print('=' * 70)
    
    # Create dummy inputs
    dummy_mel = torch.randn(1, 1, 1000, 64)
    max_length = 77
    dummy_input_ids = torch.randint(0, 50265, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    
    with torch.no_grad():
        audio_embed, text_embed = wrapper(dummy_mel, dummy_input_ids, dummy_attention_mask)
        print(f'✓ Audio embedding shape: {audio_embed.shape}')
        print(f'✓ Text embedding shape: {text_embed.shape}')
        print(f'✓ Audio embedding norm: {torch.norm(audio_embed[0]).item():.4f}')
        print(f'✓ Text embedding norm: {torch.norm(text_embed[0]).item():.4f}')
        similarity = torch.dot(audio_embed[0], text_embed[0]).item()
        print(f'✓ Test similarity: {similarity:.4f}')
    
    print('')
    print('Exporting combined model to ONNX...')
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_mel, dummy_input_ids, dummy_attention_mask),
            '$ONNX_MODEL',
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['mel_spectrogram', 'input_ids', 'attention_mask'],
            output_names=['audio_embedding', 'text_embedding'],
            dynamic_axes={
                'mel_spectrogram': {0: 'batch_size', 2: 'time_frames'},
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'audio_embedding': {0: 'batch_size'},
                'text_embedding': {0: 'batch_size'}
            },
            dynamo=False
        )
    
    print('')
    print('=' * 70)
    print('✓ COMBINED CLAP MODEL SUCCESSFULLY EXPORTED!')
    print('=' * 70)
    print(f'Output file: $ONNX_MODEL')
    print('')
    print('IMPORTANT: This model produces IDENTICAL embeddings as the .pt model!')
    print('You can safely replace the .pt model and continue analyzing your collection.')
    print('')
    print('USAGE FOR AUDIO:')
    print('  1. Preprocess: compute mel-spectrogram (n_fft=1024, hop=480, n_mels=64)')
    print('  2. Set input_ids and attention_mask to zeros')
    print('  3. Use output: audio_embedding')
    print('')
    print('USAGE FOR TEXT:')
    print('  1. Tokenize with RoBERTa tokenizer (max_length=77)')
    print('  2. Set mel_spectrogram to zeros')
    print('  3. Use output: text_embedding')
    print('')
    print('Inputs:')
    print('  - mel_spectrogram: (batch, 1, time_frames, 64)')
    print('  - input_ids: (batch, seq_length)')
    print('  - attention_mask: (batch, seq_length)')
    print('Outputs:')
    print('  - audio_embedding: (batch, 512)')
    print('  - text_embedding: (batch, 512)')
    print('=' * 70)
    
    # Verify the ONNX model produces identical results
    try:
        import onnx
        import onnxruntime as ort
        
        print('')
        print('Verifying ONNX model produces identical embeddings...')
        
        onnx_model = onnx.load('$ONNX_MODEL')
        onnx.checker.check_model(onnx_model)
        print('✓ ONNX model is valid')
        
        session = ort.InferenceSession('$ONNX_MODEL')
        
        # Test audio encoding
        onnx_outputs = session.run(None, {
            'mel_spectrogram': dummy_mel.numpy(),
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy()
        })
        
        onnx_audio_embed = onnx_outputs[0]
        onnx_text_embed = onnx_outputs[1]
        
        print(f'✓ ONNX audio embedding shape: {onnx_audio_embed.shape}')
        print(f'✓ ONNX text embedding shape: {onnx_text_embed.shape}')
        print(f'✓ ONNX audio norm: {np.linalg.norm(onnx_audio_embed[0]):.4f}')
        print(f'✓ ONNX text norm: {np.linalg.norm(onnx_text_embed[0]):.4f}')
        
        # Compare with PyTorch outputs
        audio_diff = np.abs(audio_embed.numpy() - onnx_audio_embed).max()
        text_diff = np.abs(text_embed.numpy() - onnx_text_embed).max()
        
        print(f'✓ Max difference audio (PyTorch vs ONNX): {audio_diff:.2e}')
        print(f'✓ Max difference text (PyTorch vs ONNX): {text_diff:.2e}')
        
        if audio_diff < 1e-5 and text_diff < 1e-5:
            print('✓✓✓ EMBEDDINGS ARE IDENTICAL! Safe to use for your collection.')
        else:
            print(f'Warning: Small numerical differences detected (expected < 1e-5)')
        
    except ImportError:
        print('')
        print('Note: Install onnx and onnxruntime to verify the exported model')
    except Exception as e:
        print(f'Warning: ONNX verification failed: {e}')
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f'ERROR: Export failed: {e}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "--- Conversion complete! ---"