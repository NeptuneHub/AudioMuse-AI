"""
Export the trained best JointBERT model (joint_bert_best.pt) to ONNX format.

Run after training has completed (when joint_bert_best.pt exists).
"""
from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
BEST_PT_PATH = HERE / "joint_bert_best.pt"
BEST_ONNX_PATH = HERE / "joint_bert_best.onnx"
LABELS_PATH = HERE / "labels.json"

MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64


class JointBERT(nn.Module):
    def __init__(self, model_name: str, num_intents: int, num_slots: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.intent_head = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.slot_head = nn.Linear(self.bert.config.hidden_size, num_slots)

    def forward(self, input_ids, attention_mask):
        bert_out = self.bert(input_ids, attention_mask)
        last_hidden = bert_out[0]
        intent_logits = self.intent_head(last_hidden[:, 0, :])
        slot_logits = self.slot_head(last_hidden)
        return intent_logits, slot_logits


def export_onnx(model, onnx_path: Path) -> None:
    model.eval()
    dummy_ids = torch.zeros(1, MAX_LEN, dtype=torch.long)
    dummy_mask = torch.ones(1, MAX_LEN, dtype=torch.long)
    torch.onnx.export(
        model.cpu(),
        (dummy_ids, dummy_mask),
        str(onnx_path),
        input_names=["input_ids", "attention_mask"],
        output_names=["intent_logits", "slot_logits"],
        dynamic_axes={
            "input_ids": {0: "batch"},
            "attention_mask": {0: "batch"},
            "intent_logits": {0: "batch"},
            "slot_logits": {0: "batch"},
        },
        opset_version=14,
    )
    print(f"  ONNX model written: {onnx_path}  ({onnx_path.stat().st_size / 1e6:.1f} MB)")


def main() -> None:
    if not BEST_PT_PATH.exists():
        print(f"[ERROR] {BEST_PT_PATH.name} not found")
        return

    if not LABELS_PATH.exists():
        print(f"[ERROR] {LABELS_PATH.name} not found")
        return

    print(f"[export] loading labels from {LABELS_PATH.name}")
    labels = json.loads(LABELS_PATH.read_text())
    num_intents = len(labels["intents"])
    num_slots = len(labels["slots"])

    print(f"[export] creating JointBERT model ({MODEL_NAME}, {num_intents} intents, {num_slots} slots)")
    model = JointBERT(MODEL_NAME, num_intents, num_slots)

    print(f"[export] loading best weights from {BEST_PT_PATH.name}")
    state_dict = torch.load(BEST_PT_PATH, map_location="cpu")
    model.load_state_dict(state_dict)

    print(f"[export] exporting to ONNX ...")
    export_onnx(model, BEST_ONNX_PATH)
    print(f"[export] done!")


if __name__ == "__main__":
    main()
