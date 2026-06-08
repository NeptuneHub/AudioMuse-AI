"""Assemble (and verify) ./model for the standalone CI builds.

One script for all three runners (Windows/macOS/Linux), replacing the three
near-identical "assemble ./model" blocks that were PowerShell on Windows and bash
on macOS/Linux. The ``gh`` CLI is present on every runner, so the same Python
runs everywhere.

The ~5 GB of models are NOT in git. This downloads them from the same GitHub
releases the Dockerfile uses, INCLUDING the HuggingFace cache, then trims the HF
cache to just the roberta-base tokenizer the app actually loads (the app's only
runtime HF dependency is ``AutoTokenizer.from_pretrained("roberta-base")``;
bert-base-uncased and bart-base are never loaded, and a tokenizer needs no model
weights), keeping release assets under GitHub's 2 GB per-file limit.

Usage (from the repo root):
    python scripts/standalone/assemble_model.py            # download + trim
    python scripts/standalone/assemble_model.py --verify   # check completeness

Reads ``MODEL_RELEASE``, ``DCLAP_RELEASE``, ``GITHUB_REPOSITORY`` (and ``GH_TOKEN``,
which ``gh`` consumes) from the environment, kept in the workflow ``env:`` so the
release tags are bumped in lockstep with the Dockerfile.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

MODEL = Path("model")
DCLAP_REPO = "NeptuneHub/AudioMuse-AI-DCLAP"
_TEN_MB = 10 * 1024 * 1024

REQUIRED = [
    "model/musicnn_embedding.onnx",
    "model/musicnn_prediction.onnx",
    "model/clap_text_model.onnx",
    "model/model_epoch_36.onnx",
    "model/model_epoch_36.onnx.data",
    "model/huggingface/hub/models--roberta-base/snapshots",
    "model/silero_vad.onnx",
    "model/gte-multilingual-base-int8.onnx",
    "model/whisper-small-onnx/encoder_model.onnx",
    "model/whisper-small-onnx/decoder_model_merged.onnx",
    "model/gte-multilingual-base/tokenizer.json",
]


def _env(name):
    value = os.environ.get(name)
    if not value:
        raise SystemExit(f"::error::Required environment variable {name} is not set")
    return value


def _gh_download(tag, repo, dest, patterns):
    cmd = ["gh", "release", "download", tag, "-R", repo, "-D", str(dest), "--clobber"]
    for p in patterns:
        cmd += ["-p", p]
    subprocess.run(cmd, check=True)


def _extract(tar_path, dest):
    with tarfile.open(tar_path) as tf:
        tf.extractall(dest, filter="data")


def _trim_hf_cache():
    print("==> Trim HF cache to just the roberta-base tokenizer (~1.4 GB saved)")
    hub = MODEL / "huggingface" / "hub"
    for name in ("models--bert-base-uncased", "models--facebook--bart-base"):
        shutil.rmtree(hub / name, ignore_errors=True)
    rb = hub / "models--roberta-base"
    if not rb.is_dir():
        return
    blobs = rb / "blobs"
    if blobs.is_dir():
        for f in blobs.rglob("*"):
            if f.is_file() and not f.is_symlink() and f.stat().st_size > _TEN_MB:
                f.unlink()
    snapshots = rb / "snapshots"
    if snapshots.is_dir():
        for f in snapshots.rglob("*"):
            if f.name in ("model.safetensors", "pytorch_model.bin"):
                f.unlink()


def assemble():
    model_release = _env("MODEL_RELEASE")
    dclap_release = _env("DCLAP_RELEASE")
    repo = _env("GITHUB_REPOSITORY")
    MODEL.mkdir(parents=True, exist_ok=True)

    print(f"==> musicnn + CLAP text models (from {model_release})")
    _gh_download(model_release, repo, MODEL,
                 ["musicnn_embedding.onnx", "musicnn_prediction.onnx", "clap_text_model.onnx"])

    print(f"==> DCLAP audio model (from {dclap_release} in the -DCLAP repo)")
    _gh_download(dclap_release, DCLAP_REPO, MODEL,
                 ["model_epoch_36.onnx", "model_epoch_36.onnx.data"])

    print("==> HuggingFace cache (roberta/bert/bart) -- HF_HOME points at model/huggingface")
    tmp_hf = Path(tempfile.mkdtemp())
    try:
        _gh_download(model_release, repo, tmp_hf, ["huggingface_models.tar.gz"])
        (MODEL / "huggingface").mkdir(parents=True, exist_ok=True)
        _extract(tmp_hf / "huggingface_models.tar.gz", MODEL / "huggingface")
    finally:
        shutil.rmtree(tmp_hf, ignore_errors=True)

    _trim_hf_cache()

    print("==> lyrics bundles (whisper / silero / gte)")
    tmp = Path(tempfile.mkdtemp())
    try:
        bundles = ["lyrics_model_whisper", "lyrics_model_silero_vad", "lyrics_model_gte_vnni"]
        _gh_download(model_release, repo, tmp, [f"{b}.tar.gz" for b in bundles])
        for b in bundles:
            _extract(tmp / f"{b}.tar.gz", MODEL)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def verify():
    missing = []
    for f in REQUIRED:
        p = Path(f)
        if not p.exists() or (p.is_file() and p.stat().st_size == 0):
            missing.append(f)
    roberta = MODEL / "huggingface" / "hub" / "models--roberta-base"
    if not any(roberta.rglob("tokenizer.json")):
        missing.append("roberta-base tokenizer.json (after HF-cache prune)")
    if missing:
        for f in missing:
            print(f"::error::Missing or empty: {f}")
        raise SystemExit("::error::Refusing to build an incomplete bundle.")
    total = sum(p.stat().st_size for p in MODEL.rglob("*") if p.is_file() and not p.is_symlink())
    print(f"==> Model assembly verified. model/ is {total / 1e9:.1f} GB")


def main():
    parser = argparse.ArgumentParser(description="Assemble or verify ./model for the standalone build.")
    parser.add_argument("--verify", action="store_true", help="check the assembled model/ is complete")
    args = parser.parse_args()
    if args.verify:
        verify()
    else:
        assemble()


if __name__ == "__main__":
    sys.exit(main())
