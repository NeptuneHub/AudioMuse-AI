#!/usr/bin/env python3
"""Find first module producing NaN/Inf during DyMN forward.
"""
import yaml
import sys
import argparse
import random
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
logger = logging.getLogger("dymn_nan_trace")

nan_found = False


def make_hook(name):
    def hook(mod, inp, out):
        global nan_found
        if nan_found:
            return
        # check inputs
        try:
            for x in inp:
                if isinstance(x, torch.Tensor):
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        print(f"NaN detected in input of {name}")
                        nan_found = True
                        return
            # check output
            if isinstance(out, torch.Tensor):
                if torch.isnan(out).any() or torch.isinf(out).any():
                    print(f"NaN detected in output of {name}")
                    nan_found = True
                    return
            elif isinstance(out, (list, tuple)):
                for x in out:
                    if isinstance(x, torch.Tensor) and (torch.isnan(x).any() or torch.isinf(x).any()):
                        print(f"NaN detected in output tuple of {name}")
                        nan_found = True
                        return
        except Exception as e:
            print(f"Error checking {name}: {e}")
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

    dy = DyMN()
    dy.eval()

    # attach hooks to all modules
    for name, mod in dy.named_modules():
        mod.register_forward_hook(make_hook(name))

    # pick a sample likely to reproduce NaN (we saw one earlier): iterate items until NaN observed
    import random
    p = argparse.ArgumentParser()
    p.add_argument('--samples', type=int, default=100, help='Number of unique cached items to test')
    args = p.parse_args()

    sample_count = min(args.samples, len(ids))
    sampled_ids = random.sample(ids, sample_count)

    for item in sampled_ids:
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
        if nan_found:
            print('NaN detected for item:', item)
            print('Stopping after NaN detected')
            sys.exit(1)

    # If we reach here, no NaNs were found in sampled items
    print('Zero NaN')
    sys.exit(0)

    if not nan_found:
        print('No NaNs detected in sampled items')

if __name__ == '__main__':
    main()
