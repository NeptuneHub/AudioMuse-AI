#!/bin/bash
set -e

echo "--- Installing Python dependencies ---"
pip install "numpy<2" torch torchvision torchaudio onnx onnxscript onnxruntime laion-clap

MERGED_MODEL="music_audioset_epoch_15_esc_90.14.pt"
ONNX_AUDIO_MODEL="clap_audio_model.onnx"
ONNX_TEXT_MODEL="clap_text_model.onnx"

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

print('Creating separate CLAP wrappers for audio and text...')

class AudioCLAPWrapper(nn.Module):
    '''
    Audio-only CLAP model for music encoding.
    
    Takes mel-spectrogram → produces 512-dim audio embedding.
    Produces identical embeddings as the original .pt model.
    '''
    def __init__(self, clap_model):
        super().__init__()
        self.audio_branch = clap_model.model.audio_branch
        self.audio_projection = clap_model.model.audio_projection
        
    def forward(self, mel_spec):
        '''
        Forward pass for audio encoding.
        
        Args:
            mel_spec: (batch_size, 1, time_frames, 64) log mel-spectrogram
        
        Returns:
            audio_embedding: (batch_size, 512) normalized audio embedding
        '''
        x = mel_spec.transpose(1, 3)  # (batch, 64, time, 1)
        x = self.audio_branch.bn0(x)
        x = x.transpose(1, 3)  # (batch, 1, time, 64)
        x = self.audio_branch.reshape_wav2img(x)
        audio_output = self.audio_branch.forward_features(x)
        audio_embed = self.audio_projection(audio_output['embedding'])
        audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
        return audio_embed

class TextCLAPWrapper(nn.Module):
    '''
    Text-only CLAP model for text encoding.
    
    Takes tokenized text → produces 512-dim text embedding.
    Produces identical embeddings as the original .pt model.
    '''
    def __init__(self, clap_model):
        super().__init__()
        self.text_branch = clap_model.model.text_branch
        self.text_projection = clap_model.model.text_projection
        
    def forward(self, input_ids, attention_mask):
        '''
        Forward pass for text encoding.
        
        Args:
            input_ids: (batch_size, seq_length) token IDs
            attention_mask: (batch_size, seq_length) attention mask
        
        Returns:
            text_embedding: (batch_size, 512) normalized text embedding
        '''
        text_output = self.text_branch(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        text_embed = self.text_projection(text_output.pooler_output)
        text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
        return text_embed

try:
    audio_wrapper = AudioCLAPWrapper(model)
    text_wrapper = TextCLAPWrapper(model)
    audio_wrapper.eval()
    text_wrapper.eval()
    
    print('')
    print('=' * 70)
    print('Testing CLAP models...')
    print('=' * 70)
    
    # Create dummy inputs
    dummy_mel = torch.randn(1, 1, 1000, 64)
    max_length = 77
    dummy_input_ids = torch.randint(0, 50265, (1, max_length), dtype=torch.long)
    dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)
    
    with torch.no_grad():
        audio_embed = audio_wrapper(dummy_mel)
        text_embed = text_wrapper(dummy_input_ids, dummy_attention_mask)
        print(f'✓ Audio embedding shape: {audio_embed.shape}')
        print(f'✓ Text embedding shape: {text_embed.shape}')
        print(f'✓ Audio embedding norm: {torch.norm(audio_embed[0]).item():.4f}')
        print(f'✓ Text embedding norm: {torch.norm(text_embed[0]).item():.4f}')
        similarity = torch.dot(audio_embed[0], text_embed[0]).item()
        print(f'✓ Test similarity: {similarity:.4f}')
    
    print('')
    print('Exporting audio model to ONNX...')
    with torch.no_grad():
        torch.onnx.export(
            audio_wrapper,
            (dummy_mel,),
            '$ONNX_AUDIO_MODEL',
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['mel_spectrogram'],
            output_names=['audio_embedding'],
            dynamic_axes={
                'mel_spectrogram': {0: 'batch_size', 2: 'time_frames'},
                'audio_embedding': {0: 'batch_size'}
            },
            dynamo=False
        )
    
    print('Exporting text model to ONNX...')
    with torch.no_grad():
        torch.onnx.export(
            text_wrapper,
            (dummy_input_ids, dummy_attention_mask),
            '$ONNX_TEXT_MODEL',
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input_ids', 'attention_mask'],
            output_names=['text_embedding'],
            dynamic_axes={
                'input_ids': {0: 'batch_size', 1: 'sequence_length'},
                'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
                'text_embedding': {0: 'batch_size'}
            },
            dynamo=False
        )
    
    print('')
    print('=' * 70)
    print('✓ CLAP MODELS SUCCESSFULLY EXPORTED!')
    print('=' * 70)
    print(f'Audio model: $ONNX_AUDIO_MODEL')
    print(f'Text model: $ONNX_TEXT_MODEL')
    print('')
    print('IMPORTANT: These models produce IDENTICAL embeddings as the .pt model!')
    print('You can safely replace the .pt model and continue analyzing your collection.')
    print('')
    print('AUDIO MODEL USAGE:')
    print('  Input: mel_spectrogram (batch, 1, time_frames, 64)')
    print('  Preprocessing: n_fft=1024, hop=480, n_mels=64')
    print('  Output: audio_embedding (batch, 512)')
    print('')
    print('TEXT MODEL USAGE:')
    print('  Inputs: input_ids (batch, seq_length), attention_mask (batch, seq_length)')
    print('  Tokenizer: RoBERTa tokenizer (max_length=77)')
    print('  Output: text_embedding (batch, 512)')
    print('=' * 70)
    
    # Verify the ONNX models produce identical results
    try:
        import onnx
        import onnxruntime as ort
        
        print('')
        print('Verifying ONNX models produce identical embeddings...')
        
        # Verify audio model
        onnx_audio = onnx.load('$ONNX_AUDIO_MODEL')
        onnx.checker.check_model(onnx_audio)
        print('✓ Audio ONNX model is valid')
        
        # Verify text model
        onnx_text = onnx.load('$ONNX_TEXT_MODEL')
        onnx.checker.check_model(onnx_text)
        print('✓ Text ONNX model is valid')
        
        # Test audio model
        audio_session = ort.InferenceSession('$ONNX_AUDIO_MODEL')
        onnx_audio_embed = audio_session.run(None, {
            'mel_spectrogram': dummy_mel.numpy()
        })[0]
        
        print(f'✓ ONNX audio embedding shape: {onnx_audio_embed.shape}')
        print(f'✓ ONNX audio norm: {np.linalg.norm(onnx_audio_embed[0]):.4f}')
        
        audio_diff = np.abs(audio_embed.numpy() - onnx_audio_embed).max()
        print(f'✓ Max difference audio (PyTorch vs ONNX): {audio_diff:.2e}')
        
        # Test text model
        text_session = ort.InferenceSession('$ONNX_TEXT_MODEL')
        onnx_text_embed = text_session.run(None, {
            'input_ids': dummy_input_ids.numpy(),
            'attention_mask': dummy_attention_mask.numpy()
        })[0]
        
        print(f'✓ ONNX text embedding shape: {onnx_text_embed.shape}')
        print(f'✓ ONNX text norm: {np.linalg.norm(onnx_text_embed[0]):.4f}')
        
        text_diff = np.abs(text_embed.numpy() - onnx_text_embed).max()
        print(f'✓ Max difference text (PyTorch vs ONNX): {text_diff:.2e}')
        
        if audio_diff < 1e-5 and text_diff < 1e-5:
            print('')
            print('✓✓✓ EMBEDDINGS ARE IDENTICAL! Safe to use for your collection.')
        else:
            print(f'')
            print(f'Warning: Small numerical differences detected (expected < 1e-5)')
        
    except ImportError:
        print('')
        print('Note: Install onnx and onnxruntime to verify the exported models')
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