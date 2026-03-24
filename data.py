"""
data.py — Data ingestion layer
Supports: yfinance (live), CSV files, NSE/BSE format CSVs, synthetic data
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  YFINANCE LOADER
# ─────────────────────────────────────────────
def load_yfinance(
    symbol: str,
    start: str,
    end: str,
    interval: str = "1d"
) -> pd.DataFrame:
    """Download OHLCV data from Yahoo Finance."""
    try:
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end, interval=interval)
        if df.empty:
            raise ValueError(f"No data returned for {symbol} between {start} and {end}.")
        df.index = pd.to_datetime(df.index).tz_localize(None)
        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })
        df = validate_ohlcv(df, source=f"yfinance:{symbol}")
        print(f"[data] Loaded {len(df)} bars for {symbol} from yfinance")
        return df
    except ImportError:
        raise RuntimeError("yfinance not installed. Use: pip install yfinance")


# ─────────────────────────────────────────────
#  CSV LOADER  (generic + NSE/BSE format)
# ─────────────────────────────────────────────
_NSE_COL_MAP = {
    # NSEpy / NSE direct download headers
    "Date": "date", "Symbol": "symbol",
    "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Last": "last",
    "Turnover": "turnover", "Trades": "trades",
    "Volume": "volume", "Deliverable Volume": "deliverable_volume",
    "%Deliverble": "pct_deliverable",
    # Alternate casing
    "OPEN": "open", "HIGH": "high", "LOW": "low",
    "CLOSE": "close", "VOLUME": "volume",
}


def validate_ohlcv(df: pd.DataFrame, source: str = "dataset") -> pd.DataFrame:
    """Standardize and validate an OHLCV frame for downstream use."""
    if df is None or df.empty:
        raise ValueError(f"{source} is empty.")

    frame = df.copy()
    frame.columns = [str(col).strip().lower() for col in frame.columns]

    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns {missing} in {source}. Found: {list(frame.columns)}")

    for col in required + ["volume"]:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise TypeError(f"{source} must be indexed by a DatetimeIndex.")

    frame = frame.sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame.dropna(subset=required, inplace=True)

    invalid_rows = (frame["high"] < frame["low"]) | (frame["open"] <= 0) | (frame["close"] <= 0)
    if invalid_rows.any():
        frame = frame.loc[~invalid_rows].copy()

    if frame.empty:
        raise ValueError(f"{source} has no valid OHLCV rows after cleaning.")

    return frame[["open", "high", "low", "close", "volume"]]


def load_csv(
    filepath: str,
    date_col: str | None = None,
    format: str = "generic"  # reserved for future format-specific logic
) -> pd.DataFrame:
    """Load OHLCV from CSV. Handles NSE/BSE export formats automatically."""
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {filepath}")

    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()

    # Rename known columns
    df.rename(columns=_NSE_COL_MAP, inplace=True)

    # Parse date
    date_candidates = [date_col] if date_col else []
    date_candidates += ["date", "Date", "DATE", "timestamp", "Timestamp", "Datetime", "datetime"]
    for col in date_candidates:
        if col and col in df.columns:
            df["date"] = pd.to_datetime(df[col], dayfirst=True, errors="coerce")
            break
    if "date" not in df.columns:
        raise ValueError(f"Could not locate a date column in {path.name}.")
    df.set_index("date", inplace=True)
    df = validate_ohlcv(df, source=path.name)

    print(f"[data] Loaded {len(df)} bars from {path.name}")
    return df


# ─────────────────────────────────────────────
#  SYNTHETIC DATA GENERATOR
# ─────────────────────────────────────────────
def generate_synthetic(
    n_bars: int = 1000,
    start_price: float = 1000.0,
    annual_return: float = 0.15,
    annual_vol: float = 0.20,
    start_date: str = "2021-01-01",
    seed: int = 42,
    regime_switches: bool = True
) -> pd.DataFrame:
    """
    Generate realistic synthetic OHLCV data using GBM with optional
    regime-switching (bull/bear/choppy) for strategy stress testing.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start_date, periods=n_bars)

    dt = 1 / 252
    prices = [start_price]

    if regime_switches:
        # Regime: 0=bull, 1=bear, 2=choppy
        regimes = rng.choice([0, 1, 2], size=n_bars,
                             p=[0.5, 0.25, 0.25])
        regime_params = {
            0: (annual_return, annual_vol * 0.8),
            1: (-0.20,         annual_vol * 1.5),
            2: (0.02,          annual_vol * 1.2),
        }
    else:
        regimes = np.zeros(n_bars, dtype=int)
        regime_params = {0: (annual_return, annual_vol)}

    for i in range(1, n_bars):
        mu, sigma = regime_params[regimes[i]]
        ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * rng.standard_normal()
        prices.append(prices[-1] * np.exp(ret))

    closes = np.array(prices)

    # Build OHLC from close
    daily_vol = annual_vol / np.sqrt(252)
    opens  = closes * np.exp(rng.normal(0, daily_vol * 0.3, n_bars))
    highs  = np.maximum(opens, closes) * np.exp(np.abs(rng.normal(0, daily_vol * 0.5, n_bars)))
    lows   = np.minimum(opens, closes) * np.exp(-np.abs(rng.normal(0, daily_vol * 0.5, n_bars)))
    volume = rng.integers(500_000, 5_000_000, size=n_bars).astype(float)
    # Volume spikes on big moves
    big_move = np.abs(closes / np.roll(closes, 1) - 1) > 0.02
    volume[big_move] *= rng.uniform(1.5, 3.0, size=big_move.sum())

    df = pd.DataFrame({
        "open":   opens,
        "high":   highs,
        "low":    lows,
        "close":  closes,
        "volume": volume,
    }, index=dates[:n_bars])

    print(f"[data] Generated {len(df)} synthetic bars | "
          f"start={start_price:.0f} | "
          f"end={closes[-1]:.0f} | "
          f"raw_return={closes[-1]/start_price - 1:.1%}")
    return validate_ohlcv(df, source="synthetic")


# ─────────────────────────────────────────────
#  MULTI-ASSET LOADER
# ─────────────────────────────────────────────
def load_multi(
    symbols: list[str],
    start: str,
    end: str,
    source: str = "yfinance"
) -> dict[str, pd.DataFrame]:
    """Load multiple symbols. Returns dict of {symbol: df}."""
    data = {}
    for sym in symbols:
        try:
            if source == "yfinance":
                data[sym] = load_yfinance(sym, start, end)
            else:
                raise ValueError(f"Unknown source: {source}")
        except Exception as e:
            print(f"[data] WARNING: Failed to load {sym}: {e}")
    return data


if __name__ == "__main__":
    df = generate_synthetic(n_bars=756, seed=42)
    print(df.tail())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Date range: {df.index[0].date()} → {df.index[-1].date()}")
