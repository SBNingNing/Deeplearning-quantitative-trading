"""Local web dashboard for competition trading recommendations.

Run:
    python scripts/competition_dashboard.py

Then open the printed localhost URL. The dashboard can call
scripts/predict_latest.py and display generated orders, holdings, and scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import mimetypes
import subprocess
import sys
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HOLDINGS = "outputs/competition_holdings.csv"
DEFAULT_TRADES = "outputs/competition_trades.csv"
DEFAULT_CHECKPOINT = "outputs/models/tsn_full_best.pt"
DEFAULT_PORT = 8765
DEFAULT_DATA_DIR = "A股数据"
DEFAULT_COMPETITION_START = 20260601
DEFAULT_COMPETITION_END = 20260612


def read_csv_rows(path: Path, limit: int | None = None) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[:limit] if limit is not None else rows


def write_trade_records(rows: list[dict[str, object]]) -> str:
    """Persist manual buy/sell fills and rebuild the single competition holdings file."""
    output = PROJECT_ROOT / DEFAULT_TRADES
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["trade_date", "action", "ts_code", "shares", "price", "amount"]
    cleaned: list[dict[str, object]] = []
    for row in rows:
        date = str(row.get("trade_date", "")).strip()
        action = str(row.get("action", "")).strip().lower()
        code = str(row.get("ts_code", "")).strip()
        shares = numeric(row.get("shares"), default=0.0)
        price = numeric(row.get("price"), default=0.0)
        amount = numeric(row.get("amount"), default=0.0)
        if not date or action not in {"buy", "sell"} or not code or shares <= 0:
            continue
        computed_amount = round(shares * price, 2) if price > 0 else 0.0
        if amount <= 0 and computed_amount > 0:
            amount = computed_amount
        elif computed_amount > 0 and not (computed_amount * 0.5 <= amount <= computed_amount * 1.5):
            amount = computed_amount
        cleaned.append(
            {
                "trade_date": date,
                "action": action,
                "ts_code": code,
                "shares": int(shares),
                "price": price,
                "amount": amount,
            }
        )
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(cleaned)
    rebuild_holdings_from_trades(cleaned)
    return safe_rel_path(output)


def read_close_price_map(codes: set[str], trade_date: int | None = None) -> dict[str, float]:
    """Read close prices from the latest available complete daily file."""
    if not codes:
        return {}
    dates = available_signal_dates()
    if not dates:
        return {}
    date = trade_date or dates[-1]
    path = PROJECT_ROOT / DEFAULT_DATA_DIR / "daily" / f"{date}.csv"
    if not path.exists():
        return {}
    rows = read_csv_rows(path)
    out: dict[str, float] = {}
    for row in rows:
        code = str(row.get("ts_code", ""))
        if code in codes:
            out[code] = numeric(row.get("close"), default=0.0)
    return out


def rebuild_holdings_from_trades(rows: list[dict[str, object]], initial_cash: float = 1_000_000.0) -> dict:
    """Apply trade records, value holdings at latest close, and write holdings CSV."""
    shares_by_code: dict[str, int] = {}
    cash = float(initial_cash)
    latest_trade_date = ""
    for row in rows:
        code = str(row.get("ts_code", "")).strip()
        action = str(row.get("action", "")).lower()
        shares = int(numeric(row.get("shares"), default=0.0))
        amount = numeric(row.get("amount"), default=0.0)
        if amount <= 0:
            amount = round(shares * numeric(row.get("price"), default=0.0), 2)
        if not code or shares <= 0:
            continue
        latest_trade_date = str(row.get("trade_date", latest_trade_date))
        if action == "buy":
            shares_by_code[code] = shares_by_code.get(code, 0) + shares
            cash -= amount
        elif action == "sell":
            shares_by_code[code] = shares_by_code.get(code, 0) - shares
            cash += amount

    shares_by_code = {code: shares for code, shares in shares_by_code.items() if shares > 0}
    prices = read_close_price_map(set(shares_by_code))
    position_value = round(sum(shares * prices.get(code, 0.0) for code, shares in shares_by_code.items()), 2)
    total_asset = round(cash + position_value, 2)
    holdings_path = PROJECT_ROOT / DEFAULT_HOLDINGS
    fields = ["ts_code", "buy_date", "weight", "shares", "last_price", "market_value"]
    with holdings_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for code, shares in sorted(shares_by_code.items()):
            price = prices.get(code, 0.0)
            market_value = round(shares * price, 2)
            writer.writerow(
                {
                    "ts_code": code,
                    "buy_date": latest_trade_date,
                    "weight": market_value / total_asset if total_asset > 0 else 0.0,
                    "shares": shares,
                    "last_price": price,
                    "market_value": market_value,
                }
            )
    return {
        "cash": cash,
        "position_value": position_value,
        "total_asset": total_asset,
        "valuation_date": available_signal_dates()[-1] if available_signal_dates() else "",
        "holdings": safe_rel_path(holdings_path),
    }


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def stock_suggestions(query: str, limit: int = 20) -> list[dict[str, str]]:
    """Search stock basic info by code, symbol, name, pinyin, or industry."""
    query = query.strip().lower()
    if not query:
        return []
    path = PROJECT_ROOT / DEFAULT_DATA_DIR / "basic.csv"
    if not path.exists():
        return []
    rows = read_csv_rows(path)
    matches: list[dict[str, str]] = []
    for row in rows:
        haystack = " ".join(
            str(row.get(col, ""))
            for col in ["ts_code", "symbol", "name", "cnspell", "industry"]
        ).lower()
        if query not in haystack:
            continue
        matches.append(
            {
                "ts_code": str(row.get("ts_code", "")),
                "symbol": str(row.get("symbol", "")),
                "name": str(row.get("name", "")),
                "industry": str(row.get("industry", "")),
                "cnspell": str(row.get("cnspell", "")),
            }
        )
        if len(matches) >= limit:
            break
    return matches


def latest_file(pattern: str) -> Path | None:
    matches = sorted(PROJECT_ROOT.glob(pattern), key=lambda item: item.name)
    return matches[-1] if matches else None


def available_signal_dates(data_dir: str = DEFAULT_DATA_DIR) -> list[int]:
    daily_dir = PROJECT_ROOT / data_dir / "daily"
    if not daily_dir.exists():
        return []
    return sorted(
        int(path.stem)
        for path in daily_dir.glob("*.csv")
        if path.stem.isdigit() and len(path.stem) == 8
    )


def safe_rel_path(path: Path | None) -> str:
    if path is None:
        return ""
    try:
        return str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def numeric(value: object, default: float = 0.0) -> float:
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def next_business_day_yyyymmdd(value: object) -> str:
    """Estimate the next business day from a YYYYMMDD signal date."""
    if value in ("", None):
        return ""
    try:
        day = datetime.strptime(str(int(value)), "%Y%m%d").date()
    except (TypeError, ValueError):
        return ""
    day += timedelta(days=1)
    while day.weekday() >= 5:
        day += timedelta(days=1)
    return day.strftime("%Y%m%d")


def business_days_between(start: int, end: int) -> list[int]:
    day = datetime.strptime(str(start), "%Y%m%d").date()
    end_day = datetime.strptime(str(end), "%Y%m%d").date()
    out: list[int] = []
    while day <= end_day:
        if day.weekday() < 5:
            out.append(int(day.strftime("%Y%m%d")))
        day += timedelta(days=1)
    return out


def signal_for_execution_date(signal_dates: list[int], execution_date: int) -> int | None:
    previous = [date for date in signal_dates if date < execution_date]
    if not previous:
        return None
    signal_date = previous[-1]
    return signal_date if next_business_day_yyyymmdd(signal_date) == str(execution_date) else None


def competition_schedule(signal_dates: list[int]) -> list[dict[str, str | int]]:
    rows = []
    for execution_date in business_days_between(DEFAULT_COMPETITION_START, DEFAULT_COMPETITION_END):
        previous = [date for date in signal_dates if date < execution_date]
        signal_date = signal_for_execution_date(signal_dates, execution_date)
        rows.append(
            {
                "executionDate": execution_date,
                "signalDate": signal_date or (previous[-1] if previous else ""),
                "status": "READY" if signal_date else "WAIT_DATA",
            }
        )
    return rows


def build_status(holdings_csv: str = DEFAULT_HOLDINGS) -> dict:
    holdings_csv = DEFAULT_HOLDINGS
    holdings_path = PROJECT_ROOT / holdings_csv
    trades_path = PROJECT_ROOT / DEFAULT_TRADES
    has_holdings_file = holdings_path.exists()
    orders_path = latest_file("outputs/orders/orders_*.csv") if has_holdings_file else None
    state_path = None
    if orders_path is not None:
        state_path = orders_path.with_name(orders_path.name.replace("orders_", "state_").replace(".csv", ".json"))
    predictions_path = latest_file("outputs/predictions/latest_predictions_*.csv") if has_holdings_file else None

    orders = read_csv_rows(orders_path) if orders_path else []
    holdings = read_csv_rows(holdings_path)
    trades = read_csv_rows(trades_path)
    predictions = read_csv_rows(predictions_path, limit=30) if predictions_path else []
    state = read_json(state_path) if state_path else {}

    # Re-value holdings at latest available close prices so that
    # position_value, total_asset, and displayed prices stay current.
    if holdings:
        codes = {str(row.get("ts_code", "")) for row in holdings}
        latest_prices = read_close_price_map(codes)
        for row in holdings:
            code = str(row.get("ts_code", ""))
            price = latest_prices.get(code, numeric(row.get("last_price")))
            shares = numeric(row.get("shares"))
            row["last_price"] = price
            row["market_value"] = round(shares * price, 2)
        # Recalculate weight based on latest market values
        total_mv = sum(numeric(row.get("market_value")) for row in holdings)
        if total_mv > 0:
            for row in holdings:
                row["weight"] = numeric(row.get("market_value")) / total_mv

    buy_value = sum(numeric(row.get("est_trade_value")) for row in orders if row.get("action") == "buy")
    sell_value = sum(numeric(row.get("est_trade_value")) for row in orders if row.get("action") == "sell")
    position_value = round(sum(numeric(row.get("market_value")) for row in holdings), 2)
    trade_cash = 1_000_000.0
    for row in trades:
        amount = numeric(row.get("amount"))
        if amount <= 0:
            amount = round(numeric(row.get("shares")) * numeric(row.get("price")), 2)
        if row.get("action") == "buy":
            trade_cash -= amount
        elif row.get("action") == "sell":
            trade_cash += amount
    if trades:
        trade_cash = round(trade_cash, 2)
        total_asset = round(trade_cash + position_value, 2)
    elif holdings:
        # No trade records, but we have holdings (from predict_latest).
        # Recalculate total using latest position value + estimated cash.
        estimated_cash = numeric(state.get("estimated_cash_after_rounding"), default=0.0)
        total_asset = round(estimated_cash + position_value, 2)
    else:
        total_asset = numeric(state.get("portfolio_value", 1_000_000))
    buy_count = sum(1 for row in orders if row.get("action") == "buy")
    sell_count = sum(1 for row in orders if row.get("action") == "sell")

    checkpoint_paths = sorted(PROJECT_ROOT.glob("outputs/models/*.pt"), key=lambda item: item.name)
    signal_dates = available_signal_dates()
    schedule = competition_schedule(signal_dates)
    checkpoint_default = (
        DEFAULT_CHECKPOINT
        if (PROJECT_ROOT / DEFAULT_CHECKPOINT).exists()
        else (safe_rel_path(checkpoint_paths[-1]) if checkpoint_paths else DEFAULT_CHECKPOINT)
    )

    signal_date = state.get("signal_date", state.get("trade_date", ""))
    execution_date = state.get("execution_date", next_business_day_yyyymmdd(signal_date))
    return {
        "files": {
            "orders": safe_rel_path(orders_path),
            "state": safe_rel_path(state_path),
            "predictions": safe_rel_path(predictions_path),
            "holdings": holdings_csv,
            "trades": DEFAULT_TRADES,
            "targetHoldings": safe_rel_path(PROJECT_ROOT / state["target_holdings_path"]) if state.get("target_holdings_path") else "",
        },
        "defaults": {
            "checkpoint": checkpoint_default,
            "portfolioValue": total_asset,
            "weightMethod": "gmv",
            "tradeDate": state.get("trade_date", signal_dates[-1] if signal_dates else ""),
            "executionDate": execution_date or (schedule[0]["executionDate"] if schedule else ""),
        },
        "checkpoints": [safe_rel_path(path) for path in checkpoint_paths],
        "signalDates": signal_dates[-260:],
        "competitionSchedule": schedule,
        "summary": {
            "tradeDate": signal_date,
            "executionDate": execution_date,
            "buyCount": buy_count,
            "sellCount": sell_count,
            "orderCount": len([row for row in orders if numeric(row.get("delta_shares")) != 0]),
            "holdingCount": len(holdings),
            "estimatedBuyValue": buy_value,
            "estimatedSellValue": sell_value,
            "estimatedCash": state.get("estimated_cash_after_rounding", ""),
            "positionValue": position_value,
            "cash": round(trade_cash, 2) if trades else state.get("estimated_cash_after_rounding", ""),
            "totalAsset": total_asset,
            "cashWeight": state.get("cash_weight", ""),
            "preControlVol": state.get("pre_control_annualized_vol", ""),
            "postControlVol": state.get("post_control_annualized_vol", ""),
            "minWeight": state.get("min_weight", ""),
            "maxWeight": state.get("max_weight", ""),
            "scoreTilt": state.get("gmv_score_tilt", ""),
            "buyPriceBuffer": state.get("buy_price_buffer", ""),
            "sellPriceBuffer": state.get("sell_price_buffer", ""),
        },
        "orders": orders,
        "holdings": holdings,
        "trades": trades,
        "predictions": predictions,
        "state": state,
    }


def run_prediction(payload: dict) -> dict:
    checkpoint = str(payload.get("checkpoint") or DEFAULT_CHECKPOINT)
    holdings_csv = DEFAULT_HOLDINGS
    portfolio_value = str(payload.get("portfolioValue") or "1000000")
    weight_method = str(payload.get("weightMethod") or "gmv")
    trade_date = str(payload.get("tradeDate") or "")
    execution_date = str(payload.get("executionDate") or "")
    reset_holdings = bool(payload.get("resetHoldings"))
    gmv_score_tilt = str(payload.get("gmvScoreTilt") or "0.003")
    allow_cash_vol_control = bool(payload.get("allowCashVolControl"))
    buy_price_buffer = str(payload.get("buyPriceBuffer") or "0.01")
    sell_price_buffer = str(payload.get("sellPriceBuffer") or "0.005")
    n_holdings = str(payload.get("nHoldings") or "10")
    max_sell = str(payload.get("maxSell") or "2")
    buffer_rank = str(payload.get("bufferRank") or "30")

    command = [
        sys.executable,
        "scripts/predict_latest.py",
        "--checkpoint",
        checkpoint,
        "--portfolio-value",
        portfolio_value,
        "--holdings-csv",
        holdings_csv,
        "--weight-method",
        weight_method,
        "--n-holdings",
        n_holdings,
        "--max-sell",
        max_sell,
        "--buffer-rank",
        buffer_rank,
        "--gmv-score-tilt",
        gmv_score_tilt,
        "--buy-price-buffer",
        buy_price_buffer,
        "--sell-price-buffer",
        sell_price_buffer,
    ]
    if execution_date:
        command.extend(["--execution-date", execution_date])
    elif trade_date:
        command.extend(["--trade-date", trade_date])
    if reset_holdings:
        command.append("--reset-holdings")
    if allow_cash_vol_control:
        command.append("--allow-cash-vol-control")
    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        text=True,
        capture_output=True,
        timeout=900,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "status": build_status(holdings_csv),
    }


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "CompetitionDashboard/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self.send_text(INDEX_HTML, "text/html; charset=utf-8")
            return
        if parsed.path == "/api/status":
            self.send_json(build_status())
            return
        if parsed.path == "/api/stocks":
            query = parse_qs(parsed.query)
            keyword = query.get("q", [""])[0]
            self.send_json({"items": stock_suggestions(keyword)})
            return
        self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path not in {"/api/run", "/api/save-trades"}:
            self.send_error(404)
            return
        length = int(self.headers.get("content-length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        payload = json.loads(body) if body else {}
        if parsed.path == "/api/save-trades":
            try:
                path = write_trade_records(payload.get("rows") or [])
                self.send_json({"ok": True, "path": path, "status": build_status()})
            except Exception as exc:  # pragma: no cover - user-facing server guard
                self.send_json({"ok": False, "error": str(exc)})
            return
        try:
            result = run_prediction(payload)
        except subprocess.TimeoutExpired as exc:
            result = {
                "ok": False,
                "returncode": None,
                "stdout": exc.stdout or "",
                "stderr": f"Prediction timed out after {exc.timeout} seconds.",
                "status": build_status(),
            }
        except Exception as exc:  # pragma: no cover - user-facing server guard
            result = {
                "ok": False,
                "returncode": None,
                "stdout": "",
                "stderr": str(exc),
                "status": build_status(),
            }
        self.send_json(result)

    def log_message(self, format: str, *args: object) -> None:
        sys.stderr.write("%s - %s\n" % (self.log_date_time_string(), format % args))

    def send_json(self, data: object) -> None:
        payload = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json; charset=utf-8")
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def send_text(self, text: str, content_type: str | None = None) -> None:
        payload = text.encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", content_type or mimetypes.types_map.get(".html", "text/html"))
        self.send_header("content-length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


INDEX_HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>比赛交易面板</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f7f9;
      --panel: #ffffff;
      --line: #d9dee7;
      --text: #17202a;
      --muted: #64748b;
      --buy: #b42318;
      --sell: #087443;
      --accent: #2457a6;
      --warn: #9a6700;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif;
      font-size: 14px;
    }
    header {
      border-bottom: 1px solid var(--line);
      background: var(--panel);
      padding: 18px 24px;
    }
    h1 { margin: 0 0 4px; font-size: 22px; font-weight: 700; }
    .subtle { color: var(--muted); }
    main {
      display: grid;
      grid-template-columns: 330px 1fr;
      gap: 18px;
      padding: 18px 24px 28px;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    h2 { margin: 0 0 12px; font-size: 16px; }
    label { display: block; margin: 12px 0 5px; color: var(--muted); font-size: 13px; }
    input, select {
      width: 100%;
      height: 36px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fff;
      color: var(--text);
      padding: 0 10px;
      font-size: 14px;
    }
    .checkline {
      display: flex;
      align-items: center;
      gap: 8px;
      margin-top: 12px;
      color: var(--muted);
      font-size: 13px;
    }
    .checkline input { width: 16px; height: 16px; }
    button {
      width: 100%;
      height: 38px;
      margin-top: 14px;
      border: 0;
      border-radius: 6px;
      background: var(--accent);
      color: white;
      font-weight: 700;
      cursor: pointer;
    }
    button:disabled { opacity: .65; cursor: wait; }
    .grid {
      display: grid;
      grid-template-columns: repeat(5, minmax(120px, 1fr));
      gap: 10px;
      margin-bottom: 18px;
    }
    .metric {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      min-height: 72px;
    }
    .metric .label { color: var(--muted); font-size: 12px; }
    .metric .value { margin-top: 8px; font-size: 20px; font-weight: 700; }
    .stack { display: grid; gap: 18px; }
    table {
      width: 100%;
      border-collapse: collapse;
      table-layout: fixed;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 7px;
      text-align: right;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
    }
    th { color: var(--muted); font-size: 12px; font-weight: 600; background: #fafbfc; }
    td:first-child, th:first-child,
    td:nth-child(2), th:nth-child(2) { text-align: left; }
    .buy { color: var(--buy); font-weight: 700; }
    .sell { color: var(--sell); font-weight: 700; }
    .hold { color: var(--muted); }
    .message {
      min-height: 90px;
      margin-top: 12px;
      padding: 10px;
      border: 1px solid var(--line);
      border-radius: 6px;
      background: #fbfcfd;
      color: var(--muted);
      white-space: pre-wrap;
      overflow: auto;
      max-height: 260px;
      font-family: Consolas, monospace;
      font-size: 12px;
    }
    .fileline {
      margin-top: 8px;
      color: var(--muted);
      font-size: 12px;
      word-break: break-all;
    }
    .warning { color: var(--warn); font-weight: 600; }
    .editor-grid {
      display: grid;
      grid-template-columns: .9fr .8fr 1.2fr .9fr .9fr 1fr 38px;
      gap: 8px;
      align-items: center;
      margin-bottom: 8px;
    }
    .editor-grid select { height: 32px; }
    .editor-grid input { height: 32px; }
    .iconbtn {
      width: 38px;
      margin: 0;
      background: #eef2f7;
      color: var(--text);
    }
    .editor-head {
      display: grid;
      grid-template-columns: .9fr .8fr 1.2fr .9fr .9fr 1fr 38px;
      gap: 8px;
      color: var(--muted);
      font-size: 12px;
      font-weight: 600;
      margin-bottom: 6px;
    }
    @media (max-width: 980px) {
      main { grid-template-columns: 1fr; padding: 14px; }
      .grid { grid-template-columns: repeat(2, minmax(120px, 1fr)); }
    }
  </style>
</head>
<body>
  <header>
    <h1>比赛交易面板</h1>
    <div class="subtle">生成每日模拟交易订单，展示买卖股数、目标持仓和模型预测排名。</div>
  </header>
  <main>
    <datalist id="stockSuggestions"></datalist>
    <aside>
      <section>
        <h2>运行参数</h2>
        <label for="checkpoint">模型 checkpoint</label>
        <select id="checkpoint"></select>
        <label for="executionDate">比赛下单日</label>
        <select id="executionDate"></select>
        <label for="tradeDate">信号日期</label>
        <select id="tradeDate"></select>
        <label for="portfolioValue">账户总资产</label>
        <input id="portfolioValue" type="number" min="0" step="1000" value="1000000" />
        <label for="weightMethod">权重方式</label>
        <select id="weightMethod">
          <option value="gmv">GMV 风控权重</option>
          <option value="equal">等权</option>
        </select>
        <label for="nHoldings">目标持股数</label>
        <input id="nHoldings" type="number" min="1" step="1" value="10" />
        <label for="maxSell">每日最多卖出</label>
        <input id="maxSell" type="number" min="1" step="1" value="2" />
        <label for="bufferRank">持仓缓冲排名</label>
        <input id="bufferRank" type="number" min="1" step="1" value="30" />
        <label for="gmvScoreTilt">GMV 分数倾斜</label>
        <input id="gmvScoreTilt" type="number" min="0" step="0.0001" value="0.003" />
        <label for="buyPriceBuffer">买入价格缓冲</label>
        <input id="buyPriceBuffer" type="number" min="0" step="0.001" value="0.01" />
        <label for="sellPriceBuffer">卖出价格缓冲</label>
        <input id="sellPriceBuffer" type="number" min="0" step="0.001" value="0.005" />
        <div class="checkline">
          <input id="resetHoldings" type="checkbox" />
          <span>从空仓开始（仅首日使用）</span>
        </div>
        <div class="checkline">
          <input id="allowCashVolControl" type="checkbox" />
          <span>允许波动率控制留现金</span>
        </div>
        <button id="runBtn">生成今日交易建议</button>
        <button id="refreshBtn" type="button">刷新面板</button>
        <div id="message" class="message">等待操作。</div>
      </section>
    </aside>
    <div class="stack">
      <div class="grid" id="metrics"></div>
      <section>
        <h2>订单明细</h2>
        <div id="ordersFile" class="fileline"></div>
        <div id="orders"></div>
      </section>
      <section>
        <h2>比赛日程</h2>
        <div id="schedule"></div>
      </section>
      <section>
        <h2>当前持仓</h2>
        <div id="holdingsFile" class="fileline"></div>
        <div id="holdings"></div>
      </section>
      <section>
        <h2>每日成交记录</h2>
        <div class="subtle">这里记录真实成交，不是模型建议。保存后系统会按全部成交记录重算当前持仓、可用现金和账户总资产。</div>
        <div id="actualEditor" style="margin-top:12px;"></div>
        <button id="addActualRowBtn" type="button" style="background:#087443;">+ 添加一行交易记录</button>
        <button id="fillFromOrdersBtn" type="button">用模型建议生成成交记录模板</button>
        <button id="saveActualBtn" type="button">保存成交记录并更新持仓</button>
        <div id="actualFile" class="fileline"></div>
      </section>
      <section>
        <h2>预测排名 Top 30</h2>
        <div id="predictionsFile" class="fileline"></div>
        <div id="predictions"></div>
      </section>
    </div>
  </main>
  <script>
    const fmtMoney = value => {
      const n = Number(value);
      if (!Number.isFinite(n)) return "-";
      return n.toLocaleString("zh-CN", { maximumFractionDigits: 0 });
    };
    const fmtPct = value => {
      const n = Number(value);
      if (!Number.isFinite(n)) return "-";
      return (n * 100).toFixed(2) + "%";
    };
    const cls = action => action === "buy" ? "buy" : (action === "sell" ? "sell" : "hold");
    const text = value => value === undefined || value === null || value === "" ? "-" : value;
    let latestStatus = null;

    function table(headers, rows, mapRow) {
      if (!rows || rows.length === 0) return "<div class='subtle'>暂无数据。</div>";
      const head = headers.map(h => `<th>${h}</th>`).join("");
      const body = rows.map(row => `<tr>${mapRow(row).join("")}</tr>`).join("");
      return `<table><thead><tr>${head}</tr></thead><tbody>${body}</tbody></table>`;
    }

    function setSelectOptions(select, options, selected) {
      const all = options.length ? options : [selected];
      select.innerHTML = all.map(option => {
        const isSelected = String(option) === String(selected) ? "selected" : "";
        return `<option value="${option}" ${isSelected}>${option}</option>`;
      }).join("");
    }

    async function loadStatus() {
      const resp = await fetch("/api/status");
      const data = await resp.json();
      render(data);
    }

    function render(data) {
      latestStatus = data;
      setSelectOptions(document.getElementById("checkpoint"), data.checkpoints || [], data.defaults.checkpoint);
      setSelectOptions(document.getElementById("tradeDate"), data.signalDates || [], data.defaults.tradeDate);
      setSelectOptions(
        document.getElementById("executionDate"),
        (data.competitionSchedule || []).map(row => row.executionDate),
        data.defaults.executionDate
      );
      document.getElementById("portfolioValue").value = data.defaults.portfolioValue || 1000000;
      document.getElementById("weightMethod").value = data.defaults.weightMethod || "gmv";

      const s = data.summary || {};
      const metrics = [
        ["信号日期", text(s.tradeDate)],
        ["预计下单日", text(s.executionDate)],
        ["订单数", text(s.orderCount)],
        ["买入/卖出", `${s.buyCount || 0} / ${s.sellCount || 0}`],
        ["估算买入", fmtMoney(s.estimatedBuyValue)],
        ["取整后现金", fmtMoney(s.estimatedCash)],
        ["持仓数", text(s.holdingCount)],
        ["持仓市值", fmtMoney(s.positionValue)],
        ["账户总资产", fmtMoney(s.totalAsset)],
        ["现金余额", fmtMoney(s.cash)],
        ["估算卖出", fmtMoney(s.estimatedSellValue)],
        ["现金权重", fmtPct(s.cashWeight)],
        ["风控前波动", fmtPct(s.preControlVol)],
        ["风控后波动", fmtPct(s.postControlVol)],
        ["权重范围", `${fmtPct(s.minWeight)} - ${fmtPct(s.maxWeight)}`],
        ["买入缓冲", fmtPct(s.buyPriceBuffer)],
        ["订单文件", data.files.orders ? "已生成" : "未生成"],
        ["持仓文件", data.files.holdings || "-"],
      ];
      document.getElementById("metrics").innerHTML = metrics.map(([label, value]) =>
        `<div class="metric"><div class="label">${label}</div><div class="value">${value}</div></div>`
      ).join("");

      document.getElementById("ordersFile").textContent = data.files.orders ? `文件：${data.files.orders}` : "尚未生成订单。";
      if (data.files.targetHoldings) {
        document.getElementById("ordersFile").textContent += ` | 目标持仓参考：${data.files.targetHoldings}`;
      }
      document.getElementById("orders").innerHTML = table(
        ["操作", "股票", "现有股", "目标股", "变动股", "价格", "金额", "目标权重", "排名"],
        data.orders,
        row => [
          `<td class="${cls(row.action)}">${String(row.action || "").toUpperCase()}</td>`,
          `<td>${text(row.ts_code)}</td>`,
          `<td>${fmtMoney(row.current_shares)}</td>`,
          `<td>${fmtMoney(row.target_shares)}</td>`,
          `<td>${fmtMoney(row.delta_shares)}</td>`,
          `<td>${Number(row.est_price || 0).toFixed(2)}</td>`,
          `<td>${fmtMoney(row.est_trade_value)}</td>`,
          `<td>${fmtPct(row.target_weight)}</td>`,
          `<td>${text(row.rank)}</td>`,
        ]
      );

      document.getElementById("schedule").innerHTML = table(
        ["下单日", "信号日", "状态"],
        data.competitionSchedule || [],
        row => [
          `<td>${text(row.executionDate)}</td>`,
          `<td>${text(row.signalDate)}</td>`,
          `<td class="${row.status === "READY" ? "sell" : "warning"}">${text(row.status)}</td>`,
        ]
      );

      document.getElementById("holdingsFile").textContent = `文件：${data.files.holdings}`;
      document.getElementById("actualFile").textContent = data.files.trades ? `成交记录：${data.files.trades}` : "";
      document.getElementById("holdings").innerHTML = table(
        ["股票", "股数", "价格", "市值", "权重", "日期"],
        data.holdings,
        row => [
          `<td>${text(row.ts_code)}</td>`,
          `<td>${fmtMoney(row.shares)}</td>`,
          `<td>${Number(row.last_price || 0).toFixed(2)}</td>`,
          `<td>${fmtMoney(row.market_value)}</td>`,
          `<td>${fmtPct(row.weight)}</td>`,
          `<td>${text(row.buy_date)}</td>`,
        ]
      );

      document.getElementById("predictionsFile").textContent = data.files.predictions ? `文件：${data.files.predictions}` : "尚未生成预测。";
      document.getElementById("predictions").innerHTML = table(
        ["股票", "分数", "交易日"],
        data.predictions,
        row => [
          `<td>${text(row.ts_code)}</td>`,
          `<td>${Number(row.score || 0).toFixed(6)}</td>`,
          `<td>${text(row.trade_date)}</td>`,
        ]
      );
      renderActualEditor(data.trades || []);
    }

    function renderActualEditor(rows) {
      const normalized = rows.length ? rows.map(row => ({
        trade_date: row.trade_date || document.getElementById("executionDate").value || "",
        action: row.action || "buy",
        ts_code: row.ts_code || "",
        shares: row.shares || row.target_shares || "",
        price: row.price || row.fill_price || row.last_price || row.est_price || "",
        amount: row.amount || row.market_value || "",
      })) : [{ trade_date: document.getElementById("executionDate").value || "", action: "buy", ts_code: "", shares: "", price: "", amount: "" }];
      const head = `
        <div class="editor-head">
          <span>交易日</span><span>方向</span><span>股票代码</span><span>成交股数</span><span>成交均价</span><span>成交总额</span><span></span>
        </div>`;
      document.getElementById("actualEditor").innerHTML = head + normalized.map((row, idx) => editorRow(row, idx)).join("");
    }

    function editorRow(row, idx) {
      return `
        <div class="editor-grid" data-row="${idx}">
          <input class="actual-date" placeholder="YYYYMMDD" value="${text(row.trade_date) === "-" ? "" : row.trade_date}" />
          <select class="actual-action">
            <option value="buy" ${row.action === "buy" ? "selected" : ""}>买入</option>
            <option value="sell" ${row.action === "sell" ? "selected" : ""}>卖出</option>
          </select>
          <input class="actual-code" list="stockSuggestions" placeholder="输入代码/名称/拼音" value="${text(row.ts_code) === "-" ? "" : row.ts_code}" />
          <input class="actual-shares" type="number" min="0" step="100" placeholder="成交股数" title="实际成交的股票数量，单位是股。A股通常按100股一手填写。" value="${text(row.shares) === "-" ? "" : row.shares}" />
          <input class="actual-price" type="number" min="0" step="0.01" placeholder="成交均价" title="实际成交均价，单位是元/股。" value="${text(row.price) === "-" ? "" : row.price}" />
          <input class="actual-value" type="number" min="0" step="1" placeholder="可空，自动计算" title="成交总额，单位是元。可不填，系统会按成交股数 × 成交均价计算。" value="${text(row.amount) === "-" ? "" : row.amount}" />
          <button class="iconbtn" type="button" onclick="removeActualRow(this)">×</button>
        </div>`;
    }

    function collectActualRows() {
      return Array.from(document.querySelectorAll("#actualEditor .editor-grid")).map(row => ({
        trade_date: row.querySelector(".actual-date").value.trim(),
        action: row.querySelector(".actual-action").value,
        ts_code: row.querySelector(".actual-code").value.trim(),
        shares: row.querySelector(".actual-shares").value,
        price: row.querySelector(".actual-price").value,
        amount: row.querySelector(".actual-value").value,
      })).filter(row => row.ts_code && Number(row.shares) > 0);
    }

    function removeActualRow(button) {
      const grid = button.closest(".editor-grid");
      const editor = document.getElementById("actualEditor");
      const rows = editor.querySelectorAll(".editor-grid");
      if (rows.length <= 1) {
        document.getElementById("message").textContent = "至少保留一行交易记录。";
        return;
      }
      grid.remove();
    }

    function addActualRow() {
      const editor = document.getElementById("actualEditor");
      const idx = editor.querySelectorAll(".editor-grid").length;
      const executionDate = document.getElementById("executionDate").value || "";
      const newRow = { trade_date: executionDate, action: "buy", ts_code: "", shares: "", price: "", amount: "" };
      const div = document.createElement("div");
      div.innerHTML = editorRow(newRow, idx);
      editor.appendChild(div.firstElementChild);
    }

    let stockSearchTimer = null;
    async function updateStockSuggestions(keyword) {
      const q = String(keyword || "").trim();
      if (q.length < 1) {
        document.getElementById("stockSuggestions").innerHTML = "";
        return;
      }
      const resp = await fetch(`/api/stocks?q=${encodeURIComponent(q)}`);
      const data = await resp.json();
      document.getElementById("stockSuggestions").innerHTML = (data.items || []).map(item => {
        const label = `${item.name || ""} ${item.industry || ""} ${item.cnspell || ""}`;
        return `<option value="${item.ts_code}" label="${label}"></option>`;
      }).join("");
    }

    document.addEventListener("input", event => {
      if (!event.target.classList || !event.target.classList.contains("actual-code")) {
        return;
      }
      clearTimeout(stockSearchTimer);
      const value = event.target.value;
      stockSearchTimer = setTimeout(() => updateStockSuggestions(value), 180);
    });

    function fillActualFromOrders() {
      const source = latestStatus || {};
      const executionDate = document.getElementById("executionDate").value;
      const rows = (source.orders || []).filter(row => Number(row.delta_shares) !== 0).map(row => ({
        trade_date: executionDate,
        action: Number(row.delta_shares) > 0 ? "buy" : "sell",
        ts_code: row.ts_code,
        shares: Math.abs(Number(row.delta_shares)),
        price: row.close_price || row.est_price,
        amount: "",
      }));
      renderActualEditor(rows);
    }

    async function saveActualHoldings() {
      const rows = collectActualRows();
      const msg = document.getElementById("message");
      const resp = await fetch("/api/save-trades", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ rows }),
      });
      const result = await resp.json();
      if (result.ok) {
        document.getElementById("actualFile").textContent = `已保存：${result.path}`;
        msg.textContent = `成交记录已保存并重算持仓：${result.path}`;
        render(result.status);
      } else {
        msg.textContent = `保存失败：${result.error}`;
      }
    }

    async function runPrediction() {
      const btn = document.getElementById("runBtn");
      const msg = document.getElementById("message");
      btn.disabled = true;
      msg.textContent = "正在运行 predict_latest.py，请等待...";
      const payload = {
        checkpoint: document.getElementById("checkpoint").value,
        executionDate: document.getElementById("executionDate").value,
        tradeDate: document.getElementById("tradeDate").value,
        portfolioValue: document.getElementById("portfolioValue").value,
        weightMethod: document.getElementById("weightMethod").value,
        nHoldings: document.getElementById("nHoldings").value,
        maxSell: document.getElementById("maxSell").value,
        bufferRank: document.getElementById("bufferRank").value,
        gmvScoreTilt: document.getElementById("gmvScoreTilt").value,
        buyPriceBuffer: document.getElementById("buyPriceBuffer").value,
        sellPriceBuffer: document.getElementById("sellPriceBuffer").value,
        resetHoldings: document.getElementById("resetHoldings").checked,
        allowCashVolControl: document.getElementById("allowCashVolControl").checked,
      };
      try {
        const resp = await fetch("/api/run", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify(payload),
        });
        const result = await resp.json();
        msg.textContent = `${result.ok ? "运行完成" : "运行失败"}\n\n${result.stdout || ""}\n${result.stderr || ""}`;
        render(result.status);
      } catch (err) {
        msg.textContent = "请求失败：" + err;
      } finally {
        btn.disabled = false;
      }
    }

    document.getElementById("runBtn").addEventListener("click", runPrediction);
    document.getElementById("refreshBtn").addEventListener("click", loadStatus);
    document.getElementById("addActualRowBtn").addEventListener("click", addActualRow);
    document.getElementById("fillFromOrdersBtn").addEventListener("click", fillActualFromOrders);
    document.getElementById("saveActualBtn").addEventListener("click", saveActualHoldings);
    loadStatus();
  </script>
</body>
</html>
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Start the competition trading dashboard.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    port = args.port
    for offset in range(20):
        try:
            server = ThreadingHTTPServer((args.host, port + offset), DashboardHandler)
            break
        except OSError:
            if offset == 19:
                raise
    url = f"http://{args.host}:{server.server_port}"
    print(f"Competition dashboard running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping dashboard.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
