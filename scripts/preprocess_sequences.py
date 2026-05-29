from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess_seq import (  # noqa: E402
    SequencePreprocessConfig,
    build_sequence_dataset,
    write_sequence_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe sequence tensors for TSN stock return prediction."
    )
    parser.add_argument("--data-dir", default="A股数据")
    parser.add_argument("--output-dir", default="outputs/preprocessed_seq")
    parser.add_argument("--start-date", type=int, default=20190101)
    parser.add_argument("--end-date", type=int, default=None)
    parser.add_argument("--window", type=int, default=20)
    parser.add_argument("--min-amount", type=float, default=10_000.0)
    parser.add_argument("--index-code", default="000300.SH")
    parser.add_argument("--train-end", type=int, default=20241231)
    parser.add_argument("--valid-start", type=int, default=20250101)
    parser.add_argument("--valid-end", type=int, default=20251231)
    parser.add_argument("--test-start", type=int, default=20260101)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = SequencePreprocessConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        index_code=args.index_code,
        start_date=args.start_date,
        end_date=args.end_date,
        window=args.window,
        min_amount=args.min_amount,
        train_end=args.train_end,
        valid_start=args.valid_start,
        valid_end=args.valid_end,
        test_start=args.test_start,
    )
    splits, summary = build_sequence_dataset(config)
    write_sequence_outputs(splits, summary, config)
    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

