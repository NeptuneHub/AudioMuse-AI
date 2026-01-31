#!/usr/bin/env python3
"""Collect per-module activation statistics (min/max/mean/std, NaN/Inf counts)
for a given cached item id (or sample first failing one).
"""
import yaml
import sys
from pathlib import Path
import numpy as np
import torch
import logging

repo_root = Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from student_clap.models.efficientat.dymn import DyMN
from student_clap.data.mel_cache import MelSpectrogramCache

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dymn_stats")


def make_stat_hook(name, stats):
    def hook(mod, inp, out):
        try:
            tensors = []
            for x in inp:
                if isinstance(x, torch.Tensor):
                    tensors.append(x.detach().cpu())
            if isinstance(out, torch.Tensor):
                tensors.append(out.detach().cpu())
            elif isinstance(out, (list, tuple)):
                for x in out:
                    if isinstance(x, torch.Tensor):
                        tensors.append(x.detach().cpu())

            # compute aggregated stats
            if not tensors:
                return
            vals = torch.cat([t.view(-1) for t in tensors])
            nan = torch.isnan(vals).sum().item()
            inf = torch.isinf(vals).sum().item()
            m = float(vals.mean().item())
            s = float(vals.std().item())
            mn = float(vals.min().item())
            mx = float(vals.max().item())
            stats[name] = dict(nan=nan, inf=inf, mean=m, std=s, min=mn, max=mx)
        except Exception as e:
            stats[name] = dict(error=str(e))
    return hook


def main():
    cfg_path = Path('student_clap/config.yaml')
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text())
        mel_cache_path = cfg['paths'].get('mel_cache')
    else:
        mel_cache_path = '/Volumes/audiomuse/student_clap_cache/mel_spectrograms.db'

    cache = MelSpectrogramCache(mel_cache_path)
    ids = cache.get_cached_item_ids()
    if not ids:
        print('No cached items found; abort')
        return

    # pick first failing item via quick scan
    from student_clap.tools.dymn_nan_trace import nan_found
    # simply reuse: pick same sample above
    import random
    sampled = random.sample(ids, min(20, len(ids)))

    dy = DyMN()
    dy.eval()

    stats = {}
    for name, mod in dy.named_modules():
        mod.register_forward_hook(make_stat_hook(name, stats))

    failed_item = None
    for item in sampled:
        mel_and_len = cache.get_with_audio_length(item)
        if mel_and_len is None:
            continue
        full_mel, audio_length = mel_and_len
        segments = cache.extract_overlapped_segments(full_mel, audio_length)
        if segments.shape[0] < 1:
            continue
        inp = torch.from_numpy(segments[:4]).float()
        print(f"Testing item {item} with {inp.shape[0]} segments")
        try:
            with torch.no_grad():
                _ = dy(inp)
        except Exception as e:
            print(f"Forward raised: {e}")
        # check stats for NaN
        any_nan = any((v.get('nan',0) > 0) or (v.get('inf',0) > 0) for v in stats.values())
        if any_nan:
            failed_item = item
            break
    if failed_item is None:
        print('No NaNs found in scanned items')
        return

    # Print stats for modules around where NaN appears (sorted by layer name)
    keys = sorted(stats.keys())
    for k in keys:
        v = stats[k]
        if 'error' in v:
            print(f"{k}: ERROR {v['error']}")
            continue
        print(f"{k}: nan={v['nan']}, inf={v['inf']}, mean={v['mean']:.6f}, std={v['std']:.6f}, min={v['min']:.6f}, max={v['max']:.6f}")

if __name__ == '__main__':
    main()
