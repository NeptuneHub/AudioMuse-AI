#!/usr/bin/env python3

# source .venv/bin/activate
# pip install -r requirements.txt
"""
CLAMP3 Audio Search Demo
Standalone script for analyzing audio files and searching with natural language queries.
Uses CLAMP3 model (SAAS version) for audio-text retrieval.
"""

print("Starting CLAMP3 demo script...")
print("Importing dependencies...")

import os
import sys
import numpy as np
from pathlib import Path
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel, BertConfig, Wav2Vec2FeatureExtractor
from tqdm import tqdm
import librosa
import warnings
warnings.filterwarnings('ignore')

print("✓ All imports successful!")


# =============================================================================
# CLAMP3 Configuration (matching the SAAS model)
# =============================================================================

TEXT_MODEL_NAME = "FacebookAI/xlm-roberta-base"
CLAMP3_HIDDEN_SIZE = 768
AUDIO_HIDDEN_SIZE = 768
AUDIO_NUM_LAYERS = 12
MAX_AUDIO_LENGTH = 128
MAX_TEXT_LENGTH = 128


# =============================================================================
# Simplified CLaMP3 Model (inference only, no M3/symbolic encoder needed)
# =============================================================================

class CLaMP3AudioEncoder(nn.Module):
    """Simplified CLAMP3 model for audio-text retrieval."""
    
    def __init__(self, weights_path):
        super().__init__()
        
        # Load text model (XLM-RoBERTa)
        print("Loading text encoder (XLM-RoBERTa)...")
        print("  (This may take a few minutes on first run - downloading from HuggingFace)")
        self.text_model = AutoModel.from_pretrained(TEXT_MODEL_NAME)
        self.text_proj = nn.Linear(self.text_model.config.hidden_size, CLAMP3_HIDDEN_SIZE)
        
        # Load audio model (BERT-style transformer for MERT features)
        print("Loading audio encoder...")
        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=AUDIO_HIDDEN_SIZE,
            num_hidden_layers=AUDIO_NUM_LAYERS,
            num_attention_heads=AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=MAX_AUDIO_LENGTH
        )
        from transformers import BertModel
        self.audio_model = BertModel(audio_config)
        self.audio_proj = nn.Linear(audio_config.hidden_size, CLAMP3_HIDDEN_SIZE)
        
        # Load checkpoint
        print(f"Loading CLAMP3 weights from: {weights_path}")
        checkpoint = torch.load(weights_path, map_location='cpu')
        
        # Load state dict
        state_dict = checkpoint.get('model', checkpoint)
        
        # Filter and load only audio and text components
        text_model_dict = {k.replace('text_model.', ''): v for k, v in state_dict.items() if k.startswith('text_model.')}
        text_proj_dict = {k.replace('text_proj.', ''): v for k, v in state_dict.items() if k.startswith('text_proj.')}
        audio_model_dict = {k.replace('audio_model.', ''): v for k, v in state_dict.items() if k.startswith('audio_model.')}
        audio_proj_dict = {k.replace('audio_proj.', ''): v for k, v in state_dict.items() if k.startswith('audio_proj.')}
        
        print(f"  - Text model params: {len(text_model_dict)}")
        print(f"  - Text proj params: {len(text_proj_dict)}")
        print(f"  - Audio model params: {len(audio_model_dict)}")
        print(f"  - Audio proj params: {len(audio_proj_dict)}")
        
        if len(audio_model_dict) == 0 or len(text_model_dict) == 0:
            print("  ⚠️  WARNING: Model weights appear to be missing or incorrectly formatted!")
            print("  Available keys in checkpoint:", list(state_dict.keys())[:10])
        
        self.text_model.load_state_dict(text_model_dict, strict=False)
        self.text_proj.load_state_dict(text_proj_dict)
        self.audio_model.load_state_dict(audio_model_dict)
        self.audio_proj.load_state_dict(audio_proj_dict)
        
        self.eval()
        print("✓ CLAMP3 model loaded successfully")
    
    def avg_pooling(self, features, masks):
        """Average pooling with mask."""
        masks = masks.unsqueeze(-1)  # (batch, seq, 1)
        features = features * masks
        return features.sum(dim=1) / masks.sum(dim=1)
    
    def get_text_embedding(self, text_inputs, text_masks):
        """Get text embedding from text."""
        with torch.no_grad():
            text_features = self.text_model(text_inputs, attention_mask=text_masks)['last_hidden_state']
            text_features = self.avg_pooling(text_features, text_masks)
            text_features = self.text_proj(text_features)
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        return text_features
    
    def get_audio_embedding(self, audio_inputs, audio_masks):
        """Get audio embedding from MERT features."""
        with torch.no_grad():
            audio_features = self.audio_model(inputs_embeds=audio_inputs, attention_mask=audio_masks)['last_hidden_state']
            audio_features = self.avg_pooling(audio_features, audio_masks)
            audio_features = self.audio_proj(audio_features)
            # Normalize
            audio_features = audio_features / audio_features.norm(dim=-1, keepdim=True)
        return audio_features


# =============================================================================
# MERT Feature Extractor (with on-the-fly extraction)
# =============================================================================

class MERTFeatureExtractor:
    """MERT feature extractor for audio preprocessing."""
    
    def __init__(self, model_name='m-a-p/MERT-v1-95M', device='cpu'):
        """Initialize MERT model for feature extraction."""
        self.device = torch.device(device)
        self.target_sr = 24000
        self.window_size = 5  # 5-second windows
        
        print(f"Loading MERT model: {model_name}")
        print("  (This may take a few minutes on first run - downloading ~400MB)")
        from transformers import AutoModel as HFAutoModel
        self.model = HFAutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load MERT processor (Wav2Vec2 feature extractor) for proper preprocessing
        self.processor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=self.target_sr,
            padding_value=0.0,
            return_attention_mask=True,
            do_normalize=True,
        )
        print("✓ MERT model loaded")
    
    def load_audio(self, audio_path):
        """Load audio file and resample to target sample rate."""
        try:
            # Use librosa for MP3 compatibility (no torchcodec required)
            waveform, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Convert to torch tensor
            waveform = torch.from_numpy(waveform).float()
            
            return waveform
        except Exception as e:
            print(f"Error loading audio {audio_path}: {e}")
            return None
    
    def extract_features(self, audio_path):
        """Extract MERT features from audio file.
        
        Following CLAMP3 official preprocessing:
        - Wav2Vec2FeatureExtractor for normalization (CRITICAL!)
        - 5-second non-overlapping chunks
        - All layers, average across time dimension, then average layers
        - Results in one feature vector per 5-second chunk
        """
        # Load audio
        waveform = self.load_audio(audio_path)
        if waveform is None:
            return None
        
        # CRITICAL: Process through Wav2Vec2FeatureExtractor for proper normalization
        # This is what the official code does in hf_pretrains.py
        processed_wav = self.processor(waveform.numpy(), return_tensors="pt", 
                                       sampling_rate=self.target_sr).input_values[0]
        processed_wav = processed_wav.to(self.device)
        
        # 5-second chunks, no overlap
        chunk_size = self.target_sr * self.window_size
        
        all_features = []
        
        with torch.no_grad():
            for start in range(0, processed_wav.shape[-1], chunk_size):
                end = min(start + chunk_size, processed_wav.shape[-1])
                chunk = processed_wav[start:end]
                
                # Skip chunks shorter than 1 second
                if len(chunk) < self.target_sr * 1:
                    break
                
                # Pad last chunk if needed
                if len(chunk) < chunk_size:
                    chunk = torch.nn.functional.pad(chunk, (0, chunk_size - len(chunk)))
                
                # Add batch dimension
                chunk = chunk.unsqueeze(0)
                
                # Get all hidden states from all layers
                outputs = self.model(chunk, output_hidden_states=True)
                hidden_states = outputs.hidden_states  # Tuple of (batch, seq_len, hidden) per layer
                
                # Stack all layers: (num_layers, batch, seq_len, hidden)
                hidden_states = torch.stack(hidden_states)
                
                # Average across TIME dimension first (as in official code)
                # Result: (num_layers, batch, hidden)
                features = hidden_states.mean(dim=2)
                
                # Average across layers
                # Result: (batch, hidden)
                features = features.mean(dim=0)
                
                all_features.append(features.cpu())
        
        # Stack all chunks: (num_chunks, hidden_dim)
        if len(all_features) == 0:
            return None
            
        all_features = torch.cat(all_features, dim=0)
        
        # Keep temporal sequence - each row is one 5-second chunk
        return all_features.numpy()


def extract_mert_features_simple(audio_path, mert_extractor=None):
    """
    Load pre-extracted MERT features (.npy files) or extract on-the-fly.
    """
    npy_path = Path(audio_path).with_suffix('.npy')
    
    # Try to load pre-extracted features first
    if npy_path.exists():
        features = np.load(npy_path)
        features = torch.tensor(features).float()
        # Reshape to (time, feature_dim)
        if features.ndim == 3:
            features = features.squeeze(0)
        return features
    
    # Extract features on-the-fly if MERT extractor is provided
    if mert_extractor is not None:
        features = mert_extractor.extract_features(audio_path)
        if features is not None:
            # Optionally save for future use
            npy_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(npy_path, features)
            return torch.tensor(features).float()
    
    return None


# =============================================================================
# CLAMP3 Search System
# =============================================================================

class CLAMP3Searcher:
    """CLAMP3-based audio search system."""
    
    def __init__(self, model_path, device='cpu', extract_mert_on_fly=True):
        """Initialize CLAMP3 model."""
        self.device = torch.device(device)
        self.model = CLaMP3AudioEncoder(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_NAME)
        
        # Initialize MERT extractor if needed
        self.mert_extractor = None
        if extract_mert_on_fly:
            print("\nInitializing MERT feature extractor...")
            self.mert_extractor = MERTFeatureExtractor(device=device)
        
        # Cache for audio embeddings
        self.audio_embeddings = []
        self.audio_files = []
    
    def get_audio_embedding(self, audio_features):
        """Get CLAMP3 embedding for audio (from MERT features)."""
        if audio_features is None:
            return None
        
        # Add zero vectors at start and end (as in training)
        zero_vec = torch.zeros((1, audio_features.size(-1)))
        audio_features = torch.cat((zero_vec, audio_features, zero_vec), 0)
        
        # Store actual length before padding
        actual_length = audio_features.size(0)
        
        # Truncate or pad to MAX_AUDIO_LENGTH
        if actual_length > MAX_AUDIO_LENGTH:
            audio_features = audio_features[:MAX_AUDIO_LENGTH]
            actual_length = MAX_AUDIO_LENGTH
        else:
            pad_len = MAX_AUDIO_LENGTH - actual_length
            pad = torch.zeros((pad_len, audio_features.size(-1)))
            audio_features = torch.cat((audio_features, pad), 0)
        
        # Create mask based on actual content length
        mask = torch.zeros(MAX_AUDIO_LENGTH)
        mask[:actual_length] = 1
        
        # Get embedding
        audio_features = audio_features.unsqueeze(0).to(self.device)
        mask = mask.unsqueeze(0).to(self.device)
        
        embedding = self.model.get_audio_embedding(audio_features, mask)
        return embedding.cpu().numpy().flatten()
    
    def get_text_embedding(self, query_text):
        """Get CLAMP3 embedding for text query."""
        # Tokenize
        inputs = self.tokenizer(query_text, return_tensors='pt', max_length=MAX_TEXT_LENGTH, 
                               truncation=True, padding='max_length')
        
        text_inputs = inputs['input_ids'].to(self.device)
        text_masks = inputs['attention_mask'].to(self.device)
        
        # Get embedding
        embedding = self.model.get_text_embedding(text_inputs, text_masks)
        return embedding.cpu().numpy().flatten()
    
    def analyze_folder(self, folder_path):
        """Analyze all audio files in a folder."""
        folder = Path(folder_path)
        
        print(f"\nAnalyzing audio files in: {folder}")
        print("=" * 70)
        
        # Look for both pre-extracted .npy files and raw audio files
        npy_files = list(folder.glob('*.npy'))
        audio_exts = ['.mp3', '.wav', '.flac', '.ogg', '.m4a']
        raw_audio = []
        for ext in audio_exts:
            raw_audio.extend(folder.glob(f'*{ext}'))
        
        # Prioritize .npy files if they exist
        if npy_files:
            print(f"Found {len(npy_files)} pre-extracted MERT .npy file(s)")
            audio_files = npy_files
            process_as_npy = True
        elif raw_audio:
            if self.mert_extractor is None:
                print(f"⚠️  Found {len(raw_audio)} audio file(s) but MERT extractor not initialized")
                print(f"    Re-run with extract_mert_on_fly=True to enable on-the-fly extraction")
                return
            print(f"Found {len(raw_audio)} audio file(s) - will extract MERT features on-the-fly")
            audio_files = raw_audio
            process_as_npy = False
        else:
            print(f"No audio files found in {folder}")
            return
        
        print(f"Processing {len(audio_files)} file(s)...")
        print()
        
        # Analyze each file
        for audio_file in tqdm(sorted(audio_files), desc="Extracting embeddings"):
            if process_as_npy:
                # Load pre-extracted features
                features = np.load(audio_file)
                features = torch.tensor(features).float()
                if features.ndim == 3:
                    features = features.squeeze(0)
            else:
                # Extract MERT features on-the-fly
                features = extract_mert_features_simple(audio_file, self.mert_extractor)
            
            if features is not None:
                embedding = self.get_audio_embedding(features)
                if embedding is not None:
                    self.audio_embeddings.append(embedding)
                    self.audio_files.append(audio_file)
        
        print()
        print(f"✓ Analyzed {len(self.audio_embeddings)} audio files successfully")
    
    def search(self, query_text, top_k=5):
        """Search audio files using a text query."""
        if not self.audio_embeddings:
            print("No audio files analyzed yet!")
            return []
        
        print(f"\n{'=' * 70}")
        print(f"Query: \"{query_text}\"")
        print(f"{'=' * 70}")
        
        # Get text embedding
        text_embedding = self.get_text_embedding(query_text)
        
        # Compute similarities with all audio files
        embeddings_matrix = np.vstack(self.audio_embeddings)
        similarities = embeddings_matrix @ text_embedding
        
        # Show score statistics for better understanding
        print(f"Score stats: min={similarities.min():.4f}, max={similarities.max():.4f}, "
              f"mean={similarities.mean():.4f}, std={similarities.std():.4f}")
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Print results
        results = []
        for rank, idx in enumerate(top_indices, 1):
            similarity = similarities[idx]
            audio_file = self.audio_files[idx]
            
            # Calculate normalized score (0-100 scale for easier interpretation)
            score_range = similarities.max() - similarities.min()
            normalized_score = ((similarity - similarities.min()) / score_range * 100) if score_range > 0 else 50
            
            result = {
                'rank': rank,
                'file': audio_file.name,
                'path': str(audio_file),
                'similarity': float(similarity),
                'normalized_score': float(normalized_score)
            }
            results.append(result)
            
            print(f"{rank}. {audio_file.stem}")
            print(f"   Similarity: {similarity:.4f} | Normalized: {normalized_score:.1f}/100")
        
        return results


def main():
    """Main function."""
    # Paths
    script_dir = Path(__file__).parent
    model_path = script_dir.parent.parent / "CLAMP3" / "weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth"
    
    # Check for MERT features folder
    mert_folder = script_dir.parent.parent / "test" / "songs_mert"
    audio_folder = script_dir.parent.parent / "test" / "songs"
    
    # Check if model exists
    if not model_path.exists():
        print(f"ERROR: Model not found at {model_path}")
        print(f"\nPlease download the CLAMP3 SAAS model from:")
        print(f"https://huggingface.co/sander-wood/clamp3/blob/main/weights_clamp3_saas_h_size_768_t_model_FacebookAI_xlm-roberta-base_t_length_128_a_size_768_a_layers_12_a_length_128_s_size_768_s_layers_12_p_size_64_p_length_512.pth")
        sys.exit(1)
    
    print("=" * 70)
    print("CLAMP3 AUDIO SEARCH DEMO")
    print("=" * 70)
    print()
    print("This demo uses the CLAMP3 SAAS model for audio-text retrieval.")
    print()
    print("Requirements:")
    print("  1. CLAMP3 SAAS weights (already found)")
    print("  2. MERT-preprocessed audio features (.npy files)")
    print()
    print("=" * 70)
    print()
    
    # Detect CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    print()
    
    # Initialize searcher (with on-the-fly MERT extraction enabled)
    searcher = CLAMP3Searcher(str(model_path), device=device, extract_mert_on_fly=True)
    
    # Check if MERT features exist, otherwise process raw audio
    if mert_folder.exists():
        print(f"✓ Found MERT features folder: {mert_folder}")
        searcher.analyze_folder(mert_folder)
    else:
        print(f"⚠️  MERT features folder not found: {mert_folder}")
        print(f"    Will extract MERT features on-the-fly from audio files in: {audio_folder}")
        
        if audio_folder.exists():
            searcher.analyze_folder(audio_folder)
        else:
            print(f"⚠️  Audio folder not found: {audio_folder}")
            sys.exit(1)
    
    # Search queries
    queries = [
        "Calm piano song",
        "Calm song",
        "Piano songs",
        "Energetic songs",
        "Relaxed Song",
        "happy electronic song",
        "sad electronic song"
    ]
    
    if len(searcher.audio_embeddings) > 0:
        print("\n" + "=" * 70)
        print("SEARCHING WITH TEXT QUERIES")
        print("=" * 70)
        
        all_results = {}
        for query in queries:
            results = searcher.search(query, top_k=3)
            all_results[query] = results
            print()  # Add spacing between queries
        
        # Summary
        print("\n" + "=" * 70)
        print("SEARCH COMPLETE")
        print("=" * 70)
        print(f"Analyzed: {len(searcher.audio_files)} audio files")
        print(f"Queries: {len(queries)}")
        print()


if __name__ == "__main__":
    main()
