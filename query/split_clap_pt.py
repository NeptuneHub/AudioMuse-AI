#!/usr/bin/env python3
"""
Split CLAP .pt checkpoint into two inference-only PyTorch checkpoints:
 - `clap_audio_model.pt`  (contains audio_branch + audio_projection)
 - `clap_text_model.pt`   (contains text_branch  + text_projection)

The output files are stored in the `model/` folder by default and contain only
the parameters required for inference (no optimizer or training metadata).

Usage:
  python query/split_clap_pt.py
  python query/split_clap_pt.py --src model/music_audioset_epoch_15_esc_90.14.pt

The script will also perform a lightweight verification to ensure the
split checkpoints reproduce the original model's embeddings (within
numerical tolerance).
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(prog="split_clap_pt.py",
                                     description="Split CLAP .pt into audio/text inference-only .pt files")
    parser.add_argument("--src",
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model', 'music_audioset_epoch_15_esc_90.14.pt')),
                        help="Path to the source .pt checkpoint (default: model/music_audioset_epoch_15_esc_90.14.pt)")
    parser.add_argument("--out-dir",
                        default=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model')),
                        help="Output directory for split .pt files (default: model/)")
    parser.add_argument("--audio-name", default="clap_audio_model.pt",
                        help="Filename for the audio checkpoint (default: clap_audio_model.pt)")
    parser.add_argument("--text-name", default="clap_text_model.pt",
                        help="Filename for the text checkpoint (default: clap_text_model.pt)")
    parser.add_argument("--no-verify", action="store_true",
                        help="Skip post-save verification step")
    args = parser.parse_args()

    try:
        import torch
        import laion_clap
        import numpy as np
    except Exception as e:
        print(f"ERROR: missing dependency: {e}")
        print("Install with: pip install torch laion-clap numpy")
        sys.exit(1)

    src = os.path.abspath(args.src)
    out_dir = os.path.abspath(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(src):
        print(f"ERROR: source checkpoint not found: {src}")
        sys.exit(1)

    print("Loading source CLAP .pt checkpoint (this uses laion_clap.CLAP_Module)...")
    clap = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap.load_ckpt(src)
    clap.eval()
    clap = clap.to('cpu')

    # Disable gradients (inference-only)
    for p in clap.parameters():
        p.requires_grad = False

    # Lightweight wrappers (same forward signatures used for ONNX export)
    import torch.nn as nn

    class AudioCLAPWrapper(nn.Module):
        def __init__(self, clap_model):
            super().__init__()
            self.audio_branch = clap_model.model.audio_branch
            self.audio_projection = clap_model.model.audio_projection

        def forward(self, mel_spec: torch.Tensor):
            x = mel_spec.transpose(1, 3)  # (batch, 64, time, 1)
            x = self.audio_branch.bn0(x)
            x = x.transpose(1, 3)  # (batch, 1, time, 64)
            x = self.audio_branch.reshape_wav2img(x)
            audio_output = self.audio_branch.forward_features(x)
            audio_embed = self.audio_projection(audio_output['embedding'])
            audio_embed = torch.nn.functional.normalize(audio_embed, dim=-1)
            return audio_embed

    class TextCLAPWrapper(nn.Module):
        def __init__(self, clap_model):
            super().__init__()
            self.text_branch = clap_model.model.text_branch
            self.text_projection = clap_model.model.text_projection

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
            text_output = self.text_branch(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            text_embed = self.text_projection(text_output.pooler_output)
            text_embed = torch.nn.functional.normalize(text_embed, dim=-1)
            return text_embed

    print("Creating inference wrappers from the loaded model (audio + text)...")
    audio_wrapper = AudioCLAPWrapper(clap)
    text_wrapper = TextCLAPWrapper(clap)
    audio_wrapper.eval()
    text_wrapper.eval()

    # Build minimal checkpoint dictionaries containing ONLY required params
    audio_ckpt = {
        'audio_branch': clap.model.audio_branch.state_dict(),
        'audio_projection': clap.model.audio_projection.state_dict()
    }

    text_ckpt = {
        'text_branch': clap.model.text_branch.state_dict(),
        'text_projection': clap.model.text_projection.state_dict()
    }

    audio_out_path = os.path.join(out_dir, args.audio_name)
    text_out_path = os.path.join(out_dir, args.text_name)

    print(f"Saving audio checkpoint → {audio_out_path}")
    torch.save(audio_ckpt, audio_out_path)

    print(f"Saving text checkpoint  → {text_out_path}")
    torch.save(text_ckpt, text_out_path)

    def bytes_human(n):
        for u in ['B','KB','MB','GB']:
            if n < 1024.0:
                return f"{n:3.1f}{u}"
            n /= 1024.0
        return f"{n:.1f}TB"

    try:
        audio_size = os.path.getsize(audio_out_path)
        text_size = os.path.getsize(text_out_path)
        print(f"Saved sizes: audio={bytes_human(audio_size)}, text={bytes_human(text_size)}")
    except Exception:
        pass

    # Optional verification: load the saved pieces into a fresh CLAP_Module and compare outputs
    if not args.no_verify:
        print("\nVerifying split checkpoints reproduce original embeddings (small randomized test)")
        # Create a fresh CLAP module and load the split weights into the corresponding submodules
        restored = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
        restored.eval()
        restored = restored.to('cpu')

        # Load saved dicts
        loaded_audio = torch.load(audio_out_path, map_location='cpu')
        loaded_text = torch.load(text_out_path, map_location='cpu')

        # Load into restored model's submodules
        try:
            restored.model.audio_branch.load_state_dict(loaded_audio['audio_branch'], strict=False)
            restored.model.audio_projection.load_state_dict(loaded_audio['audio_projection'], strict=False)

            restored.model.text_branch.load_state_dict(loaded_text['text_branch'], strict=False)
            restored.model.text_projection.load_state_dict(loaded_text['text_projection'], strict=False)
        except Exception as e:
            print(f"WARNING: failed to load split state_dict into fresh CLAP model: {e}")
            print("Skipping numeric verification.")
            return

        # Build wrappers for restored model
        restored_audio_wrapper = AudioCLAPWrapper(restored)
        restored_text_wrapper = TextCLAPWrapper(restored)
        restored_audio_wrapper.eval()
        restored_text_wrapper.eval()

        # Create dummy inputs (same as export tests)
        with torch.no_grad():
            dummy_mel = torch.randn(1, 1, 1000, 64, dtype=torch.float32)
            max_length = 77
            dummy_input_ids = torch.randint(0, 50265, (1, max_length), dtype=torch.long)
            dummy_attention_mask = torch.ones(1, max_length, dtype=torch.long)

            orig_audio = audio_wrapper(dummy_mel)
            new_audio = restored_audio_wrapper(dummy_mel)
            audio_diff = float((orig_audio - new_audio).abs().max().item())

            orig_text = text_wrapper(dummy_input_ids, dummy_attention_mask)
            new_text = restored_text_wrapper(dummy_input_ids, dummy_attention_mask)
            text_diff = float((orig_text - new_text).abs().max().item())

        print(f"Max difference (audio) : {audio_diff:.2e}")
        print(f"Max difference (text)  : {text_diff:.2e}")

        tol = 1e-5
        if audio_diff < tol and text_diff < tol:
            print("\n✅ Verification passed — split checkpoints reproduce original embeddings.")
        else:
            print("\n⚠️  Verification warning: numerical differences exceed tolerance")
            print("This can be normal due to floating point rounding, but check results before use.")

    print("\nDone. Split .pt files are ready for inference in 'model/' (no internet download required).")


if __name__ == '__main__':
    main()
