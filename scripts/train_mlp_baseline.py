from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from modeling import (  # noqa: E402
    MLPRegressor,
    TabularReturnDataset,
    attach_codes,
    evaluate_model,
    json_safe,
    make_loader,
    train_regression_model,
)


ID_COLUMNS = {"trade_date", "feature_end_date", "label_end_date", "ts_code", "weight", "industry"}
LABEL_COLUMNS = {"label_return_1d", "label_index_return_1d", "label_excess_1d"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an MLP baseline on tabular features.")
    parser.add_argument("--data-dir", default="outputs/preprocessed")
    parser.add_argument("--model-dir", default="outputs/models")
    parser.add_argument("--prediction-dir", default="outputs/predictions")
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def read_split(data_dir: Path, split: str) -> pd.DataFrame:
    path = data_dir / f"features_{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_parquet(path)


def main() -> int:
    args = parse_args()
    data_dir = Path(args.data_dir)
    train = read_split(data_dir, "train")
    valid = read_split(data_dir, "valid")
    if train.empty:
        raise ValueError("Training split is empty")
    if valid.empty:
        raise ValueError("Validation split is empty")

    feature_columns = [
        column
        for column in train.columns
        if column not in ID_COLUMNS and column not in LABEL_COLUMNS
    ]
    train_data = TabularReturnDataset(train, feature_columns)
    valid_data = TabularReturnDataset(valid, feature_columns)
    device = torch.device(args.device)
    model = MLPRegressor(
        input_dim=len(feature_columns),
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    checkpoint_path = Path(args.model_dir) / "mlp_best.pt"
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
            "model_type": "mlp",
            "input_dim": len(feature_columns),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "feature_columns": feature_columns,
        },
    )
    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(Path(args.model_dir) / "mlp_history.csv", index=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    prediction_dir = Path(args.prediction_dir)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, dict[str, float]] = {}
    for split in ["train", "valid", "test"]:
        frame = read_split(data_dir, split)
        if frame.empty:
            continue
        dataset = TabularReturnDataset(frame, feature_columns)
        metric, pred_frame = evaluate_model(
            model,
            make_loader(dataset, args.batch_size, shuffle=False),
            device,
        )
        pred_frame = attach_codes(pred_frame, dataset.ts_code)
        pred_frame["model"] = "mlp"
        pred_frame.to_csv(prediction_dir / f"mlp_{split}_predictions.csv", index=False)
        metrics[split] = metric.__dict__

    safe_metrics = json_safe(metrics)
    (prediction_dir / "mlp_metrics.json").write_text(
        json.dumps(safe_metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(safe_metrics, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
