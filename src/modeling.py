from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SequenceReturnDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, path: str | Path) -> None:
        data = np.load(path, allow_pickle=True)
        self.x = torch.from_numpy(data["X"].astype(np.float32))
        self.y = torch.from_numpy(data["y"].astype(np.float32))
        self.trade_date = torch.from_numpy(data["trade_date"].astype(np.int64))
        self.ts_code = data["ts_code"].astype(str)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.trade_date[index]


class TabularReturnDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, frame: pd.DataFrame, feature_columns: list[str]) -> None:
        self.frame = frame.reset_index(drop=True)
        self.x = torch.from_numpy(self.frame[feature_columns].to_numpy(dtype=np.float32))
        self.y = torch.from_numpy(self.frame["label_excess_1d"].to_numpy(dtype=np.float32))
        self.trade_date = torch.from_numpy(self.frame["trade_date"].to_numpy(dtype=np.int64))
        self.ts_code = self.frame["ts_code"].astype(str).to_numpy()

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index], self.trade_date[index]


class TemporalSegmentNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_segments: int = 4,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_segments = num_segments
        self.encoder = nn.Sequential(
            nn.Conv1d(input_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )
        self.attention = nn.Linear(hidden_dim, 1)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got shape {tuple(x.shape)}")
        batch, steps, features = x.shape
        if features != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {features}")
        if steps % self.num_segments != 0:
            raise ValueError("Time dimension must be divisible by num_segments")

        segment_length = steps // self.num_segments
        segments = x.reshape(batch, self.num_segments, segment_length, features)
        segments = segments.reshape(batch * self.num_segments, segment_length, features)
        segments = segments.transpose(1, 2)
        encoded = self.encoder(segments).squeeze(-1)
        encoded = encoded.reshape(batch, self.num_segments, self.hidden_dim)
        weights = torch.softmax(self.attention(encoded).squeeze(-1), dim=1).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 128, dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def pairwise_rank_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    trade_date: torch.Tensor,
) -> torch.Tensor:
    losses: list[torch.Tensor] = []
    for date in torch.unique(trade_date):
        mask = trade_date == date
        if int(mask.sum()) < 2:
            continue
        p = pred[mask]
        y = target[mask]
        target_diff = y[:, None] - y[None, :]
        sign = torch.sign(target_diff)
        valid = sign != 0
        if not bool(valid.any()):
            continue
        pred_diff = p[:, None] - p[None, :]
        losses.append(torch.nn.functional.softplus(-pred_diff[valid] * sign[valid]).mean())
    if not losses:
        return pred.new_tensor(0.0)
    return torch.stack(losses).mean()


@dataclass(frozen=True)
class MetricResult:
    loss: float
    daily_ic: float
    icir: float
    direction_accuracy: float


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[MetricResult, pd.DataFrame]:
    model.eval()
    preds: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    dates: list[np.ndarray] = []
    total_loss = 0.0
    total_count = 0
    loss_fn = nn.MSELoss(reduction="sum")
    with torch.no_grad():
        for x, y, trade_date in loader:
            x = x.to(device)
            y = y.to(device)
            out = model(x)
            total_loss += float(loss_fn(out, y).item())
            total_count += int(y.numel())
            preds.append(out.detach().cpu().numpy())
            targets.append(y.detach().cpu().numpy())
            dates.append(trade_date.numpy())

    frame = pd.DataFrame(
        {
            "trade_date": np.concatenate(dates) if dates else np.array([], dtype=np.int64),
            "y": np.concatenate(targets) if targets else np.array([], dtype=np.float32),
            "score": np.concatenate(preds) if preds else np.array([], dtype=np.float32),
        }
    )
    metric = compute_metrics(frame, total_loss / max(total_count, 1))
    return metric, frame


def compute_metrics(frame: pd.DataFrame, loss: float) -> MetricResult:
    if frame.empty:
        return MetricResult(loss=math.nan, daily_ic=math.nan, icir=math.nan, direction_accuracy=math.nan)

    ics: list[float] = []
    for _, group in frame.groupby("trade_date"):
        if group.shape[0] < 2:
            continue
        if group["score"].std(ddof=0) == 0 or group["y"].std(ddof=0) == 0:
            continue
        corr = group["score"].corr(group["y"])
        if pd.notna(corr):
            ics.append(float(corr))
    if ics:
        daily_ic = float(np.mean(ics))
        ic_std = float(np.std(ics, ddof=0))
        icir = daily_ic / ic_std if ic_std > 0 else math.nan
    else:
        daily_ic = math.nan
        icir = math.nan
    direction_accuracy = float((np.sign(frame["score"]) == np.sign(frame["y"])).mean())
    return MetricResult(loss=float(loss), daily_ic=daily_ic, icir=icir, direction_accuracy=direction_accuracy)


def make_loader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_regression_model(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    rank_weight: float,
    patience: int,
    checkpoint_path: str | Path,
    checkpoint_extra: dict,
) -> list[dict[str, float]]:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    mse = nn.MSELoss()
    best_score = -math.inf
    best_epoch = 0
    history: list[dict[str, float]] = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_count = 0
        for x, y, trade_date in train_loader:
            x = x.to(device)
            y = y.to(device)
            trade_date = trade_date.to(device)
            pred = model(x)
            loss = mse(pred, y) + rank_weight * pairwise_rank_loss(pred, y, trade_date)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * int(y.numel())
            train_count += int(y.numel())

        valid_metrics, _ = evaluate_model(model, valid_loader, device)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss / max(train_count, 1),
            "valid_loss": valid_metrics.loss,
            "valid_daily_ic": valid_metrics.daily_ic,
            "valid_icir": valid_metrics.icir,
            "valid_direction_accuracy": valid_metrics.direction_accuracy,
        }
        history.append(row)
        print(
            "epoch={epoch} train_loss={train_loss:.6f} valid_loss={valid_loss:.6f} "
            "valid_ic={valid_daily_ic:.4f} valid_icir={valid_icir:.4f} "
            "valid_dir_acc={valid_direction_accuracy:.4f}".format(**row)
        )

        score = valid_metrics.icir
        if math.isnan(score):
            score = -valid_metrics.loss
        if score > best_score:
            best_score = score
            best_epoch = epoch
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_score": best_score,
                    **checkpoint_extra,
                },
                checkpoint_path,
            )
        elif epoch - best_epoch >= patience:
            break

    return history


def attach_codes(frame: pd.DataFrame, codes: Iterable[str]) -> pd.DataFrame:
    result = frame.copy()
    result.insert(1, "ts_code", list(codes)[: len(result)])
    return result


def json_safe(value):
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
