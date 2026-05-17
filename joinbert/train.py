"""
Fine-tune JointBERT (intent + slot heads on a shared DistilBERT encoder) for
AudioMuse-AI tool routing, then export to ONNX.

Outputs (written next to this script):
    labels.json            - INTENT_LABELS / SLOT_LABELS used by inference
    joint_bert.pt          - PyTorch state dict (for retraining / debugging)
    joint_bert.onnx        - ONNX model used by inference.py
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "training_data.json"
VAL_PATH = HERE / "validation_data.json"
LABELS_PATH = HERE / "labels.json"
BEST_PT_PATH = HERE / "joint_bert_best.pt"
BEST_ONNX_PATH = HERE / "joint_bert_best.onnx"
LAST_PT_PATH = HERE / "joint_bert_last.pt"
LAST_ONNX_PATH = HERE / "joint_bert_last.onnx"

# Other models tested: answerdotai/ModernBERT-base, sentence-transformers/all-MiniLM-L6-v2 (~80 MB, ~3 ms inference)
MODEL_NAME = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 100               # always run to completion; LR scheduler handles plateaus
LR = 5e-5                  # aggressive start; ReduceLROnPlateau will drop to 5e-5 / 5e-6 if val_loss stalls
LR_FACTOR = 0.1            # drop LR by 10x on plateau
LR_PATIENCE = 5            # epochs of no val_loss improvement before LR drops
LR_MIN = 1e-9              # floor for ReduceLROnPlateau — set low so we never run out of LR drops
SEED = 42

INLINE_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def parse_inline(annotated: str) -> tuple[str, list[dict]]:
    """Convert '[value](type) rest' -> (clean_text, [{start,end,type,value}])."""
    plain = []
    entities = []
    cursor = 0
    out_len = 0
    for m in INLINE_RE.finditer(annotated):
        prefix = annotated[cursor : m.start()]
        plain.append(prefix)
        out_len += len(prefix)
        value, ent_type = m.group(1), m.group(2)
        entities.append({"start": out_len, "end": out_len + len(value), "type": ent_type, "value": value})
        plain.append(value)
        out_len += len(value)
        cursor = m.end()
    plain.append(annotated[cursor:])
    return "".join(plain), entities


def build_label_vocab(examples: list[dict]) -> tuple[list[str], list[str]]:
    intents = set()
    entity_types = set()
    for ex in examples:
        intents.update(ex["intents"])
        _, ents = parse_inline(ex["text"])
        for e in ents:
            entity_types.add(e["type"])
    intent_labels = sorted(intents)
    slot_labels = ["O"]
    for t in sorted(entity_types):
        slot_labels.append(f"B-{t}")
        slot_labels.append(f"I-{t}")
    return intent_labels, slot_labels


def align_bio(tokenizer, text: str, entities: list[dict], slot_to_id: dict[str, int]):
    """Tokenize text and produce per-token BIO label ids aligned to WordPieces."""
    enc = tokenizer(
        text,
        max_length=MAX_LEN,
        padding="max_length",
        truncation=True,
        return_offsets_mapping=True,
    )
    offsets = enc["offset_mapping"]
    n = len(offsets)
    labels = ["O"] * n

    for ent in entities:
        first = True
        for i, (a, b) in enumerate(offsets):
            if a == b == 0:  # special token / padding
                continue
            # Token overlaps entity span -> mark
            if a >= ent["start"] and b <= ent["end"]:
                labels[i] = f"B-{ent['type']}" if first else f"I-{ent['type']}"
                first = False

    label_ids = []
    for i, (a, b) in enumerate(offsets):
        if a == b == 0:
            label_ids.append(-100)  # ignored by CrossEntropyLoss
        else:
            label_ids.append(slot_to_id[labels[i]])
    return enc["input_ids"], enc["attention_mask"], label_ids


class IntentSlotDataset(Dataset):
    def __init__(self, examples, tokenizer, intent_to_id, slot_to_id):
        self.items = []
        for ex in examples:
            text, ents = parse_inline(ex["text"])
            ids, mask, slot_ids = align_bio(tokenizer, text, ents, slot_to_id)
            intent_vec = [0.0] * len(intent_to_id)
            for name in ex["intents"]:
                intent_vec[intent_to_id[name]] = 1.0
            self.items.append(
                {
                    "input_ids": torch.tensor(ids, dtype=torch.long),
                    "attention_mask": torch.tensor(mask, dtype=torch.long),
                    "slot_ids": torch.tensor(slot_ids, dtype=torch.long),
                    "intent_vec": torch.tensor(intent_vec, dtype=torch.float32),
                    "text": text,
                }
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def collate(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "slot_ids": torch.stack([b["slot_ids"] for b in batch]),
        "intent_vec": torch.stack([b["intent_vec"] for b in batch]),
    }


class JointBERT(nn.Module):
    def __init__(self, model_name: str, n_intents: int, n_slots: int):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name, attn_implementation="eager")
        H = self.bert.config.hidden_size
        self.intent_dropout = nn.Dropout(0.1)
        self.slot_dropout = nn.Dropout(0.1)
        self.intent_head = nn.Linear(H, n_intents)
        self.slot_head = nn.Linear(H, n_slots)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        h = out.last_hidden_state
        cls = h[:, 0]
        intent_logits = self.intent_head(self.intent_dropout(cls))
        slot_logits = self.slot_head(self.slot_dropout(h))
        return intent_logits, slot_logits


def _evaluate(model, loader, device) -> tuple[float, float, float]:
    """Return (avg_loss, intent_micro_F1, slot_token_accuracy) on the given loader."""
    if loader is None or len(loader.dataset) == 0:
        return float("nan"), float("nan"), float("nan")

    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    model.eval()

    total_loss = 0.0
    # Intent: micro-F1 across all (example, tool) cells (multi-label)
    tp = fp = fn = 0
    # Slot: per-token accuracy on non-ignored positions
    correct_tokens = 0
    total_tokens = 0

    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            intent_y = batch["intent_vec"].to(device)
            slot_y = batch["slot_ids"].to(device)

            intent_logits, slot_logits = model(ids, mask)
            loss = bce(intent_logits, intent_y) + ce(
                slot_logits.view(-1, slot_logits.size(-1)), slot_y.view(-1)
            )
            total_loss += loss.item()

            intent_pred = (torch.sigmoid(intent_logits) > 0.5).long()
            intent_true = intent_y.long()
            tp += int(((intent_pred == 1) & (intent_true == 1)).sum().item())
            fp += int(((intent_pred == 1) & (intent_true == 0)).sum().item())
            fn += int(((intent_pred == 0) & (intent_true == 1)).sum().item())

            slot_pred = slot_logits.argmax(-1)
            valid = slot_y != -100
            correct_tokens += int(((slot_pred == slot_y) & valid).sum().item())
            total_tokens += int(valid.sum().item())

    avg_loss = total_loss / len(loader)
    intent_f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0.0
    slot_acc = (correct_tokens / total_tokens) if total_tokens > 0 else 0.0
    model.train()
    return avg_loss, intent_f1, slot_acc


def train(model, train_loader, val_loader, optimizer, scheduler, device, epochs):
    """Run training with ReduceLROnPlateau + early stopping on val_loss.

    Returns (best_state_dict, best_epoch, best_val_loss, best_val_f1, best_val_acc).
    The returned state_dict is a CPU deep-copy snapshotted at the epoch where
    val_loss was lowest; main() uses it to write joint_bert_best.{pt,onnx}.
    """
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss(ignore_index=-100)
    print(
        f"  {'epoch':>8} | {'lr':>9} | {'train_loss':>10} | {'val_loss':>10} | "
        f"{'val_intent_F1':>13} | {'val_slot_acc':>12} | note",
        flush=True,
    )
    print(
        f"  {'-' * 8} | {'-' * 9} | {'-' * 10} | {'-' * 10} | "
        f"{'-' * 13} | {'-' * 12} | ----",
        flush=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_f1 = best_val_acc = float("nan")
    best_state_dict: dict | None = None

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            intent_y = batch["intent_vec"].to(device)
            slot_y = batch["slot_ids"].to(device)

            intent_logits, slot_logits = model(ids, mask)
            loss_intent = bce(intent_logits, intent_y)
            loss_slot = ce(slot_logits.view(-1, slot_logits.size(-1)), slot_y.view(-1))
            loss = loss_intent + loss_slot

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        train_loss = total / len(train_loader)
        val_loss, val_f1, val_acc = _evaluate(model, val_loader, device)
        current_lr = optimizer.param_groups[0]["lr"]

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_val_f1, best_val_acc = val_f1, val_acc
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state_dict, BEST_PT_PATH)
            note = "BEST (saved)"
        else:
            note = ""

        # ReduceLROnPlateau still runs every epoch — when val_loss stalls for
        # LR_PATIENCE epochs the LR drops by LR_FACTOR. No early stop: we
        # always run all `epochs` iterations so the LR cascade can fully play
        # out and we can inspect the full loss curve.
        scheduler.step(val_loss)

        print(
            f"  {epoch:02d}/{epochs:02d}    | {current_lr:9.2e} | {train_loss:10.4f} | "
            f"{val_loss:10.4f} | {val_f1:13.4f} | {val_acc:12.4f} | {note}",
            flush=True,
        )

    if best_state_dict is None:
        # Shouldn't happen unless val set is empty; fall back to current weights.
        best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epochs

    return best_state_dict, best_epoch, best_val_loss, best_val_f1, best_val_acc


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


def _verify_val_labels(val_examples, intent_to_id, slot_to_id) -> None:
    """Ensure validation only uses labels the model is being trained to predict."""
    unknown_intents = set()
    unknown_entity_types = set()
    for ex in val_examples:
        for name in ex["intents"]:
            if name not in intent_to_id:
                unknown_intents.add(name)
        _, ents = parse_inline(ex["text"])
        for e in ents:
            if f"B-{e['type']}" not in slot_to_id:
                unknown_entity_types.add(e["type"])
    if unknown_intents or unknown_entity_types:
        raise SystemExit(
            f"[train] validation_data.json uses labels not present in training_data.json:\n"
            f"        unknown intents:      {sorted(unknown_intents)}\n"
            f"        unknown entity types: {sorted(unknown_entity_types)}\n"
            f"        Fix: add training examples for these labels, or remove the val examples that use them."
        )


def main() -> None:
    set_seed(SEED)
    device = pick_device()
    print(f"[train] device = {device}")

    raw = json.loads(DATA_PATH.read_text())
    examples = raw["examples"]
    print(f"[train] loaded {len(examples)} training examples from {DATA_PATH.name}")

    val_examples: list[dict] = []
    if VAL_PATH.exists():
        val_examples = json.loads(VAL_PATH.read_text()).get("examples", [])
        print(f"[train] loaded {len(val_examples)} validation examples from {VAL_PATH.name}")
    else:
        print(f"[train] WARNING: {VAL_PATH.name} not found — training without held-out metrics")

    intent_labels, slot_labels = build_label_vocab(examples)
    intent_to_id = {n: i for i, n in enumerate(intent_labels)}
    slot_to_id = {n: i for i, n in enumerate(slot_labels)}
    print(f"[train] intents  ({len(intent_labels)}): {intent_labels}")
    print(f"[train] slots    ({len(slot_labels)}): {slot_labels}")

    if val_examples:
        _verify_val_labels(val_examples, intent_to_id, slot_to_id)

    LABELS_PATH.write_text(
        json.dumps(
            {"intents": intent_labels, "slots": slot_labels, "model_name": MODEL_NAME, "max_len": MAX_LEN},
            indent=2,
        )
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    train_ds = IntentSlotDataset(examples, tokenizer, intent_to_id, slot_to_id)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate)

    val_loader = None
    if val_examples:
        val_ds = IntentSlotDataset(val_examples, tokenizer, intent_to_id, slot_to_id)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate)

    model = JointBERT(MODEL_NAME, len(intent_labels), len(slot_labels)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, min_lr=LR_MIN,
    )

    print(
        f"[train] training for {EPOCHS} epochs "
        f"(batch={BATCH_SIZE}, lr={LR}, lr_factor={LR_FACTOR}, "
        f"lr_patience={LR_PATIENCE}, no early stop)"
    )
    best_state, best_epoch, best_val_loss, best_val_f1, best_val_acc = train(
        model, train_loader, val_loader, optimizer, scheduler, device, EPOCHS
    )

    # Save LAST weights (current model state — the final epoch we ran).
    torch.save(model.state_dict(), LAST_PT_PATH)
    print(f"[train] last PyTorch weights saved: {LAST_PT_PATH.name}")

    print("[train] exporting LAST to ONNX ...")
    export_onnx(model, LAST_ONNX_PATH)

    # Reload BEST weights into the model (now on CPU after export_onnx) and
    # export those too. BEST_PT_PATH was already written during training.
    model.load_state_dict(best_state)
    print(
        f"[train] best epoch was {best_epoch:02d}: "
        f"val_loss={best_val_loss:.4f}, val_intent_F1={best_val_f1:.4f}, "
        f"val_slot_acc={best_val_acc:.4f}"
    )
    print("[train] exporting BEST to ONNX ...")
    export_onnx(model, BEST_ONNX_PATH)
    print(f"[train] inference.py will load: {BEST_ONNX_PATH.name}")


if __name__ == "__main__":
    main()
