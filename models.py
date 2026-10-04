"""Train one ResNet-18 per OctoNet modality, then test it.

Samples are middle activity windows from ``cut_manual.csv``, read from the
``octonet_entire`` link. Recordings are split by activity so every segment of
one take stays in train, validation, or test.

Examples:

    python models.py --check
    python models.py --users 1,2,3 --epochs 10
    python models.py --modalities IRA,wifi --epochs 20
    python models.py --test-only --modalities IRA
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from models.data import (
    MODALITIES,
    collect_split,
    limit_rows,
    load_metadata,
    recording_key,
    split_recordings,
)
from models.features import SHAPES
from models.resnet import ResNet18

ROOT = Path(__file__).resolve().parent
WEIGHTS = ROOT / "models" / "weights"
RESULTS = ROOT / "models" / "results.json"


def parse_args():
    parser = argparse.ArgumentParser(description="Train and test one ResNet-18 per OctoNet modality.")
    parser.add_argument("--modalities", default="all", help="Comma-separated modality names, or 'all'.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=4, help="Stop after this many epochs without a better validation accuracy.")
    parser.add_argument("--max-recordings", type=int, default=0, help="Use a random subset of recordings. 0 keeps every row.")
    parser.add_argument("--users", default="", help="Comma-separated user ids. Empty keeps every user.")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--test-only", action="store_true", help="Evaluate saved weights on the test split.")
    parser.add_argument("--check", action="store_true", help="Load one window of each modality and run one training step.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def selected_modalities(text):
    if text.strip().lower() == "all":
        return list(MODALITIES)
    names = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in names if name not in MODALITIES]
    if unknown:
        raise SystemExit(f"Unknown modality {unknown}. Choose from: {', '.join(MODALITIES)}")
    return names


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device(name):
    if name == "cuda" or (name == "auto" and torch.cuda.is_available()):
        return torch.device("cuda")
    return torch.device("cpu")


def encode_labels(train_labels, other_labels):
    activities = sorted(set(train_labels))
    index = {name: i for i, name in enumerate(activities)}
    encoded = []
    for labels in other_labels:
        keep = [index[name] for name in labels if name in index]
        encoded.append(keep)
    train_encoded = [index[name] for name in train_labels]
    return activities, train_encoded, encoded


def stack_split(features, labels, activities):
    if len(labels) == 0:
        return None
    index = {name: i for i, name in enumerate(activities)}
    keep = [i for i, name in enumerate(labels) if name in index]
    if not keep:
        return None
    arrays = features[keep]
    targets = [index[labels[i]] for i in keep]
    dataset = TensorDataset(torch.from_numpy(arrays), torch.tensor(targets, dtype=torch.long))
    return dataset


def class_weights(targets, num_classes):
    counts = np.bincount(targets, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    per_class_hit = None
    per_class_count = None
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        if training and inputs.shape[0] < 2:
            continue
        if training:
            optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()
        predictions = logits.argmax(dim=1)
        batch = targets.shape[0]
        total_loss += float(loss.detach()) * batch
        correct += int((predictions == targets).sum())
        seen += batch
        if not training:
            classes = logits.shape[1]
            if per_class_hit is None:
                per_class_hit = torch.zeros(classes, dtype=torch.long)
                per_class_count = torch.zeros(classes, dtype=torch.long)
            matched = predictions.cpu() == targets.cpu()
            per_class_hit += torch.bincount(targets.cpu()[matched], minlength=classes)
            per_class_count += torch.bincount(targets.cpu(), minlength=classes)
    if seen == 0:
        return {"loss": float("nan"), "accuracy": float("nan"), "macro_recall": float("nan"), "count": 0}
    macro = float("nan")
    if per_class_count is not None and int(per_class_count.sum()) > 0:
        present = per_class_count > 0
        macro = float((per_class_hit[present].float() / per_class_count[present].float()).mean())
    return {"loss": total_loss / seen, "accuracy": correct / seen, "macro_recall": macro, "count": seen}


def weight_path(modality):
    return WEIGHTS / f"{modality}_resnet18.pt"


def save_checkpoint(path, model, activities, modality, shape, config, metrics):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "activities": activities,
            "modality": modality,
            "shape": list(shape),
            "config": config,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path, device):
    try:
        blob = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        blob = torch.load(path, map_location=device)
    return blob


def prepare_rows(users, max_recordings, seed):
    user_list = [part.strip() for part in users.split(",") if part.strip()] or None
    rows = load_metadata(users=user_list)
    rows = limit_rows(rows, max_recordings, seed)
    return rows


def feature_splits(modality, parts):
    splits = {}
    for name, rows in parts.items():
        print(f"{modality} {name}: reading {len(rows)} recordings")
        features, labels = collect_split(modality, rows)
        splits[name] = (features, labels)
    return splits


def make_loader(dataset, batch_size, shuffle, drop_incomplete=False):
    if dataset is None or len(dataset) == 0:
        return None
    # Batch-norm in a residual stage rejects a training batch of one sample
    # once the feature map has shrunk to 1x1.
    if len(dataset) >= 2:
        batch = max(2, min(batch_size, len(dataset)))
    else:
        batch = 1
    drop_last = bool(drop_incomplete and len(dataset) > batch and len(dataset) % batch == 1)
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=drop_last)


def train_modality(modality, rows, args, device):
    parts = split_recordings(rows, args.seed)
    print(
        f"\n{modality}: {len(parts['train'])} train / {len(parts['val'])} val / "
        f"{len(parts['test'])} test recordings"
    )
    raw = feature_splits(modality, parts)
    train_x, train_y = raw["train"]
    if len(train_y) < 2:
        print(f"{modality}: not enough training windows ({len(train_y)}). Skipping.")
        return None
    activities, train_targets, (val_targets, test_targets) = encode_labels(
        train_y, [raw["val"][1], raw["test"][1]]
    )
    train_set = stack_split(train_x, train_y, activities)
    val_set = stack_split(raw["val"][0], raw["val"][1], activities)
    test_set = stack_split(raw["test"][0], raw["test"][1], activities)
    shape = SHAPES[modality]
    model = ResNet18(in_channels=shape[0], num_classes=len(activities)).to(device)
    weights = class_weights(train_targets, len(activities)).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    train_loader = make_loader(train_set, args.batch_size, shuffle=True, drop_incomplete=True)
    val_loader = make_loader(val_set, args.batch_size, shuffle=False)
    best_state = None
    best_accuracy = -1.0
    best_epoch = 0
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, optimizer)
        scheduler.step()
        if val_loader is None:
            val_metrics = {
                "loss": float("nan"),
                "accuracy": float("nan"),
                "macro_recall": float("nan"),
                "count": 0,
            }
            score = -train_metrics["loss"]
            improved = best_state is None or score > best_accuracy
            if improved:
                best_accuracy = score
        else:
            val_metrics = run_epoch(model, val_loader, criterion, device)
            improved = val_metrics["accuracy"] > best_accuracy
            if improved:
                best_accuracy = val_metrics["accuracy"]
        history.append({"epoch": epoch, "train": train_metrics, "val": val_metrics})
        if val_metrics["count"] == 0:
            val_text = "n/a"
        else:
            val_text = f"{val_metrics['accuracy']:.3f}"
        print(
            f"  epoch {epoch:03d}  train loss {train_metrics['loss']:.4f} acc {train_metrics['accuracy']:.3f}"
            f"  val acc {val_text}  n_val {val_metrics['count']}"
        )
        if improved:
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
            if val_loader is not None and stale >= args.patience:
                print(f"  early stop at epoch {epoch}")
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    config = {
        "seed": args.seed,
        "users": args.users,
        "max_recordings": args.max_recordings,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "best_epoch": best_epoch,
        "train_recordings": [recording_key(row) for row in parts["train"]],
        "val_recordings": [recording_key(row) for row in parts["val"]],
        "test_recordings": [recording_key(row) for row in parts["test"]],
    }
    test_metrics = evaluate_model(model, test_set, criterion, device, args.batch_size)
    metrics = {
        "best_epoch": best_epoch,
        "best_val_accuracy": None if val_loader is None else best_accuracy,
        "test": test_metrics,
        "history": [
            {
                "epoch": item["epoch"],
                "train_loss": item["train"]["loss"],
                "train_accuracy": item["train"]["accuracy"],
                "val_accuracy": None if item["val"]["count"] == 0 else item["val"]["accuracy"],
            }
            for item in history
        ],
    }
    path = weight_path(modality)
    save_checkpoint(path, model, activities, modality, shape, config, metrics)
    print(
        f"{modality}: test accuracy {test_metrics['accuracy']:.3f} "
        f"macro recall {test_metrics['macro_recall']:.3f} on {test_metrics['count']} windows"
    )
    print(f"{modality}: saved {path}")
    return {"modality": modality, "activities": len(activities), "weights": str(path), **metrics}


def evaluate_model(model, dataset, criterion, device, batch_size):
    loader = make_loader(dataset, batch_size, shuffle=False)
    if loader is None:
        return {"loss": float("nan"), "accuracy": float("nan"), "macro_recall": float("nan"), "count": 0}
    return run_epoch(model, loader, criterion, device)


def test_modality(modality, rows, args, device):
    path = weight_path(modality)
    if not path.is_file():
        print(f"{modality}: no weights at {path}")
        return None
    blob = load_checkpoint(path, device)
    activities = blob["activities"]
    shape = tuple(blob["shape"])
    config = blob.get("config", {})
    test_keys = set(config.get("test_recordings", []))
    if test_keys:
        test_rows = [row for row in rows if recording_key(row) in test_keys]
    else:
        test_rows = split_recordings(rows, args.seed)["test"]
    print(f"\n{modality}: testing {len(test_rows)} recordings")
    features, labels = collect_split(modality, test_rows)
    dataset = stack_split(features, labels, activities)
    model = ResNet18(in_channels=shape[0], num_classes=len(activities)).to(device)
    model.load_state_dict(blob["state_dict"])
    criterion = nn.CrossEntropyLoss()
    metrics = evaluate_model(model, dataset, criterion, device, args.batch_size)
    print(
        f"{modality}: test accuracy {metrics['accuracy']:.3f} "
        f"macro recall {metrics['macro_recall']:.3f} on {metrics['count']} windows"
    )
    return {"modality": modality, "activities": len(activities), "weights": str(path), "test": metrics}


def check_modalities(device):
    """Load the first metadata row and take one optimizer step on each modality."""
    rows = load_metadata()[:1]
    print(f"check recording {recording_key(rows[0])}")
    ok = True
    for modality in MODALITIES:
        features, labels = collect_split(modality, rows)
        if features.ndim != 4 or features.shape[0] == 0:
            print(f"  FAIL {modality}: no windows, shape {getattr(features, 'shape', None)}")
            ok = False
            continue
        expected = (features.shape[0],) + SHAPES[modality]
        if tuple(features.shape) != expected:
            print(f"  FAIL {modality}: got {tuple(features.shape)}, expected {expected}")
            ok = False
            continue
        try:
            model = ResNet18(in_channels=SHAPES[modality][0], num_classes=2).to(device)
            model.train()
            repeated = np.repeat(features[:1], 2, axis=0)
            batch = torch.from_numpy(repeated).to(device)
            target = torch.zeros(batch.shape[0], dtype=torch.long, device=device)
            loss = nn.functional.cross_entropy(model(batch), target)
            loss.backward()
        except Exception as exc:
            print(f"  FAIL {modality}: {exc}")
            ok = False
            continue
        print(f"  ok {modality}: {tuple(features.shape)} loss {float(loss.detach()):.4f} ({len(labels)} windows)")
    if not ok:
        raise SystemExit(1)


def summarize_saved_weights():
    """Read every saved checkpoint so a one-modality run keeps the other results."""
    order = {name: index for index, name in enumerate(MODALITIES)}
    rows = []
    for path in WEIGHTS.glob("*_resnet18.pt"):
        blob = load_checkpoint(path, torch.device("cpu"))
        metrics = blob.get("metrics") or {}
        rows.append({
            "modality": blob.get("modality", path.stem),
            "activities": len(blob.get("activities") or []),
            "weights": str(path),
            "best_epoch": metrics.get("best_epoch"),
            "best_val_accuracy": metrics.get("best_val_accuracy"),
            "test": metrics.get("test") or {},
        })
    rows.sort(key=lambda row: order.get(row["modality"], len(order)))
    return rows


def store_results(records):
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    saved = summarize_saved_weights()
    if not saved:
        saved = [record for record in records if record]
    serializable = json.loads(json.dumps(saved, default=_json_default))
    RESULTS.write_text(json.dumps(serializable, indent=2) + "\n")
    print(f"wrote {RESULTS}")
    return serializable


def _json_default(value):
    if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
        return None
    raise TypeError(f"cannot encode {type(value)}")


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)
    print(f"device {device}")
    if args.check:
        check_modalities(device)
        return
    modalities = selected_modalities(args.modalities)
    rows = prepare_rows(args.users, args.max_recordings, args.seed)
    print(f"{len(rows)} recordings, modalities: {', '.join(modalities)}")
    records = []
    for modality in modalities:
        if args.test_only:
            records.append(test_modality(modality, rows, args, device))
        else:
            records.append(train_modality(modality, rows, args, device))
    saved = store_results(records)
    print("\nsummary")
    for record in saved:
        test = record.get("test") or {}
        accuracy = test.get("accuracy")
        count = test.get("count")
        text = "n/a" if accuracy is None or (isinstance(accuracy, float) and np.isnan(accuracy)) else f"{accuracy:.3f}"
        print(f"  {record['modality']}: test accuracy {text} on {count} windows, {record['activities']} classes")


if __name__ == "__main__":
    main()
