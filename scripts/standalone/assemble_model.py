

import argparse
import os
import shutil
import subprocess
import sys
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
    subprocess.run(["tar", "-xzf", str(tar_path), "-C", str(dest)], check=True)


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


def _materialize_hf_symlinks():
    print("==> Materialize HF-cache symlinks into real files (PyInstaller/zip drop symlinks on Windows)")
    hf = MODEL / "huggingface"
    if not hf.is_dir():
        return
    for path in sorted(hf.rglob("*")):
        if not path.is_symlink():
            continue
        target = path.resolve()
        if not target.is_file():
            continue
        data = target.read_bytes()
        path.unlink()
        path.write_bytes(data)


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
    _materialize_hf_symlinks()

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
    snapshots = roberta / "snapshots"
    if not snapshots.is_dir() or not any(snapshots.iterdir()):
        missing.append("roberta-base snapshots/ empty — symlinks not extracted (use tar, not tarfile)")
    tokenizer_files = ("tokenizer.json", "vocab.json", "merges.txt", "config.json")
    for name in tokenizer_files:
        real = [p for p in roberta.rglob(name)
                if p.is_file() and not p.is_symlink() and p.stat().st_size > 0]
        if not real:
            missing.append(f"roberta-base {name} must be a real non-symlink file (run _materialize_hf_symlinks)")
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
