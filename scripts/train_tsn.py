"""Train TSN / TSN+GRU return prediction model.

Two modes:
  Default (evaluation):  train on train split, validate on valid split
  --include-valid (competition): merge train+valid+test for training,
                                 hold out last 10% by trade_date for early stopping
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modeling import (  # noqa: E402
    SequenceReturnDataset,
    TemporalSegmentNet,
    TemporalSegmentGRU,
    attach_codes,
    evaluate_model,
    json_safe,
    make_loader,
    train_regression_model,
)


class ArrayDataset(torch.utils.data.Dataset):
    """Lightweight dataset from in-memory arrays, avoids .npz re-read."""

    def __init__(self, x: np.ndarray, y: np.ndarray, trade_date: np.ndarray, ts_code: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))
        self.trade_date = torch.from_numpy(trade_date.astype(np.int64))
        self.ts_code = ts_code.astype(str)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int):
        return self.x[index], self.y[index], self.trade_date[index]


def load_npz(path: Path) -> dict[str, np.ndarray]:
    return dict(np.load(path, allow_pickle=True))


def merge_splits(splits: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Concatenate multiple split dicts into one."""
    keys = ["X", "y", "trade_date", "feature_end_date", "label_end_date",
            "feature_dates", "ts_code", "weight", "industry"]
    result = {}
    for key in keys:
        if key in splits[0]:
            arrays = [s[key] for s in splits if key in s]
            if arrays[0].dtype == object or key in ("ts_code", "industry", "feature_columns"):
                result[key] = np.concatenate(arrays)
            else:
                result[key] = np.concatenate([a.astype(np.float32 if "float" in str(a.dtype) else a.dtype) for a in arrays])
    return result


def split_by_date(arrays: dict[str, np.ndarray], valid_ratio: float = 0.10):
    """Split merged arrays by trade_date, keeping last valid_ratio as validation."""
    dates = arrays["trade_date"]
    unique_dates = np.unique(dates)
    split_idx = int(len(unique_dates) * (1 - valid_ratio))
    if split_idx >= len(unique_dates):
        split_idx = len(unique_dates) - 1
    cutoff_date = unique_dates[split_idx]

    train_mask = dates <= cutoff_date
    valid_mask = dates > cutoff_date

    train_arrays = {k: v[train_mask] for k, v in arrays.items()
                    if k not in ("feature_columns",)}
    valid_arrays = {k: v[valid_mask] for k, v in arrays.items()
                    if k not in ("feature_columns",)}
    return train_arrays, valid_arrays


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the TSN return prediction model.")
    parser.add_argument("--data-dir", default="outputs/preprocessed_seq")
    parser.add_argument("--model-dir", default="outputs/models")
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument(
        "--model-type",
        choices=["tsn", "tsn_gru"],
        default="tsn_gru",
        help="Model architecture: tsn (original) or tsn_gru (with GRU, recommended).",
    )
    parser.add_argument(
        "--include-valid",
        action="store_true",
        help="Competition mode: merge all splits, train on 2019-2026.5.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    tag = "full" if args.include_valid else "eval"

    train_path = data_dir / "sequences_train.npz"
    valid_path = data_dir / "sequences_valid.npz"
    test_path = data_dir / "sequences_test.npz"

    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Run scripts/preprocess_sequences.py before training TSN")

    feature_columns = list(np.load(train_path, allow_pickle=True)["feature_columns"])
    input_dim = len(feature_columns)

    if args.include_valid:
        if not test_path.exists():
            raise FileNotFoundError(
                "Test split required for competition mode. "
                "Re-run preprocessing with test_start set."
            )
        # Merge ALL splits for maximum training data (2019-2026.5)
        merged = merge_splits([
            load_npz(train_path),
            load_npz(valid_path),
            load_npz(test_path),
        ])
        train_arr, valid_arr = split_by_date(merged, valid_ratio=0.10)
        train_data = ArrayDataset(train_arr["X"], train_arr["y"],
                                  train_arr["trade_date"], train_arr["ts_code"])
        valid_data = ArrayDataset(valid_arr["X"], valid_arr["y"],
                                  valid_arr["trade_date"], valid_arr["ts_code"])
        train_dates = np.unique(train_arr["trade_date"])
        valid_dates = np.unique(valid_arr["trade_date"])
        print(f"Competition mode: total samples={len(merged['y'])}, "
              f"train={len(train_data)} (dates {train_dates[0]}-{train_dates[-1]}), "
              f"valid={len(valid_data)} (dates {valid_dates[0]}-{valid_dates[-1]})")
    else:
        train_data = SequenceReturnDataset(train_path)
        valid_data = SequenceReturnDataset(valid_path)
        print(f"Evaluation mode: train={len(train_data)} (2019-2024), "
              f"valid={len(valid_data)} (2025)")

    if len(train_data) == 0:
        raise ValueError("Training split is empty")
    if len(valid_data) == 0:
        raise ValueError("Validation split is empty")

    device = torch.device(args.device)
    if args.model_type == "tsn_gru":
        model = TemporalSegmentGRU(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_segments=4,
            dropout=args.dropout,
        ).to(device)
    else:
        model = TemporalSegmentNet(
            input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_segments=4,
            dropout=args.dropout,
        ).to(device)

    checkpoint_path = Path(args.model_dir) / f"tsn_{tag}_best.pt"
    history = train_regression_model(
        model=model,
        train_loader=make_loader(train_data, args.batch_size, shuffle=True),
        valid_loader=make_loader(valid_data, args.batch_size, shuffle=False),
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        rank_weight=args.rank_weight,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        checkpoint_extra={
            "model_type": args.model_type,
            "input_dim": input_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "feature_columns": feature_columns,
        },
    )
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(Path(args.model_dir) / f"tsn_{tag}_history.csv", index=False)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("WARNING: No checkpoint saved. Using untrained model for predictions.")
        checkpoint = {"model_type": args.model_type, "input_dim": input_dim}

    prediction_dir = Path(args.prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}

    for split in ["train", "valid", "test"]:
        path = data_dir / f"sequences_{split}.npz"
        if not path.exists():
            continue
        dataset = SequenceReturnDataset(path)
        if len(dataset) == 0:
            continue
        metric, pred_frame = evaluate_model(
            model,
            make_loader(dataset, args.batch_size, shuffle=False),
            device,
        )
        pred_frame = attach_codes(pred_frame, dataset.ts_code)
        pred_frame["model"] = f"tsn_{tag}"
        pred_frame.to_csv(prediction_dir / f"tsn_{tag}_{split}_predictions.csv", index=False)
        metrics[split] = metric.__dict__

    safe_metrics = json_safe(metrics)
    (prediction_dir / f"tsn_{tag}_metrics.json").write_text(
        json.dumps(safe_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(safe_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
