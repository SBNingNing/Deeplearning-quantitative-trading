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
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train_path = data_dir / "sequences_train.npz"
    valid_path = data_dir / "sequences_valid.npz"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError("Run scripts/preprocess_sequences.py before training TSN")

    train_data = SequenceReturnDataset(train_path)
    valid_data = SequenceReturnDataset(valid_path)
    if len(train_data) == 0:
        raise ValueError("Training split is empty")
    if len(valid_data) == 0:
        raise ValueError("Validation split is empty")

    input_dim = int(train_data.x.shape[-1])
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
    checkpoint_path = Path(args.model_dir) / "tsn_best.pt"
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
            "feature_columns": np.load(train_path, allow_pickle=True)["feature_columns"].tolist(),
        },
    )
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(Path(args.model_dir) / "tsn_history.csv", index=False)

    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
    else:
        print("WARNING: No checkpoint saved (all scores NaN). Using untrained model for predictions.")
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
        pred_frame["model"] = "tsn"
        pred_frame.to_csv(prediction_dir / f"tsn_{split}_predictions.csv", index=False)
        metrics[split] = metric.__dict__

    safe_metrics = json_safe(metrics)
    (prediction_dir / "tsn_metrics.json").write_text(
        json.dumps(safe_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(safe_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
