"""
JointBERT for AudioMuse-AI — Orchestrator.

Pure workflow, no model/training logic here. Runs:
    1. train.py  -> fine-tunes JointBERT, saves PyTorch + ONNX
    2. inference.py -> loads ONNX, runs demo queries (5 per tool)
"""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def run_step(label: str, script: str) -> None:
    print(f"\n{'=' * 70}\n{label}\n{'=' * 70}\n", flush=True)
    result = subprocess.run([sys.executable, str(HERE / script)], cwd=HERE)
    if result.returncode != 0:
        sys.exit(f"[orchestrator] {script} failed with exit code {result.returncode}")


def main() -> None:
    run_step("STEP 1 — Train JointBERT (PyTorch) and export to ONNX", "train.py")
    run_step("STEP 2 — Inference demo (ONNX runtime)", "inference.py")
    print("\n[orchestrator] All steps completed successfully.\n")


if __name__ == "__main__":
    main()
