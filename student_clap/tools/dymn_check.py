#!/usr/bin/env python3
"""Diagnostic script for DyMN dynamic convolution behavior.

Checks whether DynamicConv attention and aggregated weights vary across samples
and whether param scales look reasonable.

Usage:
  source student_clap/venv/bin/activate
  python student_clap/tools/dymn_check.py --n 3 --items 3

"""
import argparse
import yaml
import sys
from pathlib import Path
import numpy as np
import torch
import logging

# ensure repo root on path
repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from student_clap.models.efficientat.dymn import DyMN, DynamicConv
from student_clap.data.mel_cache import MelSpectrogramCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dymn_check")


def attach_hooks(model, stats):
    """Attach forward hooks to DynamicConv modules to capture attention & agg stats"""
    for name, module in model.named_modules():
        if isinstance(module, DynamicConv):
            def make_hook(mod_name, mod):
                def hook(mod, inputs, output):
                    # inputs: (context, temperature)
                    context = inputs[0]
                    with torch.no_grad():
                        attn_raw = mod.residuals[0](context)  # (B, k)
                        attn = torch.softmax(attn_raw / 1.0, dim=1)
                        # compute agg same as forward
                        attn_unsq = attn.unsqueeze(1).unsqueeze(3)  # (B,1,k,1)
                        agg = (mod.weight * attn_unsq).sum(dim=2).squeeze(1)  # (B, flat)
                        stats.append({
                            'name': mod_name,
                            'attn': attn.cpu().numpy(),
                            'agg_mean': agg.mean().cpu().numpy(),
                            'agg_std': agg.std().cpu().numpy(),
                            'weight_norm': mod.weight.data.norm().cpu().item()
                        })
                return hook
            module.register_forward_hook(make_hook(name, module))


def print_stats(stats):
    # summarize per-module
    from collections import defaultdict
    byname = defaultdict(list)
    for s in stats:
        byname[s['name']].append(s)

    for name, items in byname.items():
        attn_vals = np.concatenate([i['attn'] for i in items], axis=0)
        agg_means = np.array([i['agg_mean'] for i in items])
        agg_stds = np.array([i['agg_std'] for i in items])
        weight_norms = np.array([i['weight_norm'] for i in items])

        print(f"Module: {name}")
        print(f"  attn shape: {attn_vals.shape}, attn mean: {attn_vals.mean():.6f}, std: {attn_vals.std():.6f}, min: {attn_vals.min():.6f}, max: {attn_vals.max():.6f}")
        # Check if attn is varying across samples by column std
        col_std = attn_vals.std(axis=0)
        print(f"  attn per-k std: {col_std}")
        print(f"  agg_mean mean: {agg_means.mean():.6f}, agg_mean std: {agg_means.std():.6f}")
        print(f"  agg_std mean: {agg_stds.mean():.6f}, agg_std std: {agg_stds.std():.6f}")
        print(f"  weight_norm: mean={weight_norms.mean():.6f}, std={weight_norms.std():.6f}\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mel-cache', default=None)
    p.add_argument('--n', type=int, default=5, help='number of segments/samples to test')
    p.add_argument('--items', type=int, default=2, help='number of cached items to sample')
    p.add_argument('--device', default='cpu')
    args = p.parse_args()

    # Load config to find cache path
    cfg_path = Path('student_clap/config.yaml')
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        mel_cache_path = args.mel_cache or cfg['paths'].get('mel_cache')
    else:
        mel_cache_path = args.mel_cache or '/Volumes/audiomuse/student_clap_cache/mel_spectrograms.db'

    print(f"Using mel cache: {mel_cache_path}")
    cache = MelSpectrogramCache(mel_cache_path)
    ids = cache.get_cached_item_ids()
    if not ids:
        print('No cached items found; abort')
        return

    # Build DyMN standalone for diagnostics
    device = torch.device(args.device)
    # Try to pick parameters consistent with model defaults
    dy = DyMN()
    dy.to(device)
    dy.eval()

    stats = []
    attach_hooks(dy, stats)

    # For a few items, extract segments and run through dy
    import random
    sampled = random.sample(ids, min(args.items, len(ids)))
    for item in sampled:
        mel_and_len = cache.get_with_audio_length(item)
        if mel_and_len is None:
            continue
        full_mel, audio_length = mel_and_len
        segments = cache.extract_overlapped_segments(full_mel, audio_length)
        # pick up to n segments
        sel = segments[:args.n]
        # convert to tensor (num, 1, n_mels, time)
        inp = torch.from_numpy(sel).float().to(device)
        print(f"Running {inp.shape[0]} segments for item {item} (song seconds approx {audio_length/48000:.1f})")
        with torch.no_grad():
            # DyMN forward expects (batch, 1, F, T)
            out = dy(inp)

    print_stats(stats)


if __name__ == '__main__':
    main()
