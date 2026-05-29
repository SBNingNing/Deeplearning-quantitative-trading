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

from stock_pool import DEFAULT_INDEX_CODE, DEFAULT_MIN_AMOUNT, build_stock_pool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an as-of CSI 300 index-enhancement stock pool."
    )
    parser.add_argument(
        "--data-dir",
        default="A股数据",
        help="Data root containing basic.csv, daily/, stock_st/, and index_weight/.",
    )
    parser.add_argument(
        "--date",
        default=None,
        help="As-of date in YYYYMMDD. Defaults to the latest daily csv date.",
    )
    parser.add_argument(
        "--index-code",
        default=DEFAULT_INDEX_CODE,
        help="Index code used for constituents. Default: 000300.SH.",
    )
    parser.add_argument(
        "--min-amount",
        type=float,
        default=DEFAULT_MIN_AMOUNT,
        help="Minimum daily amount in thousand yuan. Default: 10000.",
    )
    parser.add_argument(
        "--allow-same-day-weight",
        action="store_true",
        help="Allow using an index weight snapshot dated the same day as --date.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional csv output path. If omitted, only summary is printed.",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional json output path for the build summary.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    pool, summary = build_stock_pool(
        data_dir=args.data_dir,
        as_of_date=args.date,
        index_code=args.index_code,
        min_amount=args.min_amount,
        strict_weight_before_date=not args.allow_same_day_weight,
    )

    summary_dict = asdict(summary)
    print(json.dumps(summary_dict, ensure_ascii=False, indent=2))

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        pool.to_csv(output, index=False, encoding="utf-8-sig")
        print(f"saved_pool={output}")

    if args.summary_output:
        output = Path(args.summary_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(summary_dict, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"saved_summary={output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
