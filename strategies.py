"""
strategies.py — Pluggable strategy library
Each strategy returns (entries: pd.Series[bool], exits: pd.Series[bool])
All signals are computed using only past data (no look-ahead).
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional


# ─────────────────────────────────────────────
#  BASE CLASS
# ─────────────────────────────────────────────
class Strategy(ABC):
    name: str = "BaseStrategy"

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
        """
        Args:
            df: OHLCV DataFrame with columns [open, high, low, close, volume]
        Returns:
            entries: boolean Series (True = go long)
            exits:   boolean Series (True = close position)
        """
        ...

    def __repr__(self):
        params = {k: v for k, v in vars(self).items() if not k.startswith("_")}
        return f"{self.name}({params})"


# ─────────────────────────────────────────────
#  INDICATORS LIBRARY  (no TA-Lib dependency)
# ─────────────────────────────────────────────
class Indicators:

    @staticmethod
    def sma(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean()

    @staticmethod
    def ema(series: pd.Series, period: int) -> pd.Series:
        return series.ewm(span=period, adjust=False).mean()

    @staticmethod
    def rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        hl   = df["high"] - df["low"]
        hpc  = (df["high"] - df["close"].shift(1)).abs()
        lpc  = (df["low"]  - df["close"].shift(1)).abs()
        tr   = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    @staticmethod
    def bollinger(series: pd.Series, period: int = 20, std: float = 2.0):
        mid  = series.rolling(period).mean()
        band = series.rolling(period).std()
        return mid + std * band, mid, mid - std * band

    @staticmethod
    def macd(series: pd.Series,
             fast: int = 12, slow: int = 26, signal: int = 9):
        fast_ema   = series.ewm(span=fast,   adjust=False).mean()
        slow_ema   = series.ewm(span=slow,   adjust=False).mean()
        macd_line  = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram  = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def donchian(df: pd.DataFrame, period: int = 20):
        upper = df["high"].rolling(period).max()
        lower = df["low"].rolling(period).min()
        mid   = (upper + lower) / 2
        return upper, mid, lower

    @staticmethod
    def momentum(series: pd.Series, period: int = 12) -> pd.Series:
        return series / series.shift(period) - 1

    @staticmethod
    def zscore(series: pd.Series, period: int = 20) -> pd.Series:
        mean = series.rolling(period).mean()
        std  = series.rolling(period).std()
        return (series - mean) / std.replace(0, np.nan)

    @staticmethod
    def volume_sma(df: pd.DataFrame, period: int = 20) -> pd.Series:
        return df["volume"].rolling(period).mean()


I = Indicators()


# ─────────────────────────────────────────────
#  STRATEGY 1: Moving Average Crossover
# ─────────────────────────────────────────────
class MACrossStrategy(Strategy):
    """
    Classic MA crossover: go long when fast MA crosses above slow MA.
    Optional: EMA or SMA, ATR-based stop built in.
    """
    name = "MACross"

    def __init__(
        self,
        fast: int = 20,
        slow: int = 50,
        ma_type: str = "ema",   # "ema" | "sma"
        trend_filter: bool = True,   # 200-day trend filter
        volume_confirm: bool = True,
    ):
        self.fast   = fast
        self.slow   = slow
        self.ma_type = ma_type
        self.trend_filter  = trend_filter
        self.volume_confirm = volume_confirm

    def generate_signals(self, df: pd.DataFrame):
        close = df["close"]
        _ma = I.ema if self.ma_type == "ema" else I.sma

        fast_ma = _ma(close, self.fast)
        slow_ma = _ma(close, self.slow)

        # Crossover detection (no look-ahead: compare [t] vs [t-1])
        cross_up   = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
        cross_down = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

        entries = cross_up.copy()
        exits   = cross_down.copy()

        if self.trend_filter:
            trend_ma = _ma(close, 200)
            entries  = entries & (close > trend_ma)

        if self.volume_confirm:
            vol_avg = I.volume_sma(df, 20)
            entries = entries & (df["volume"] > vol_avg)

        return entries.fillna(False), exits.fillna(False)


# ─────────────────────────────────────────────
#  STRATEGY 2: Momentum
# ─────────────────────────────────────────────
class MomentumStrategy(Strategy):
    """
    12-1 month momentum: rank stocks by 12-month return excluding last month.
    Entry when momentum is positive + RSI not overbought.
    Exit on momentum reversal or RSI overbought + price below EMA.
    """
    name = "Momentum"

    def __init__(
        self,
        lookback: int  = 252,   # ~12 months
        skip: int      = 21,    # skip last month (avoid reversal)
        rsi_period: int = 14,
        rsi_entry_cap: float = 70.0,  # don't enter if overbought
        rsi_exit_below: float = 30.0,
        trend_ma: int  = 100,
    ):
        self.lookback     = lookback
        self.skip         = skip
        self.rsi_period   = rsi_period
        self.rsi_entry_cap = rsi_entry_cap
        self.rsi_exit_below = rsi_exit_below
        self.trend_ma     = trend_ma

    def generate_signals(self, df: pd.DataFrame):
        close = df["close"]

        # 12-1 momentum
        mom = close.shift(self.skip) / close.shift(self.lookback) - 1
        rsi = I.rsi(close, self.rsi_period)
        ema = I.ema(close, self.trend_ma)

        # Entry: positive momentum, RSI not overbought, above trend
        entries = (
            (mom > 0) &
            (mom.shift(1) <= 0) &          # fresh signal
            (rsi < self.rsi_entry_cap) &
            (close > ema)
        )

        # Exit: momentum flips negative OR RSI oversold with price below ema
        exits = (
            (mom < 0) |
            ((rsi < self.rsi_exit_below) & (close < ema))
        )

        # Only exit when in a position (handled by engine), but signal both
        return entries.fillna(False), exits.fillna(False)


# ─────────────────────────────────────────────
#  STRATEGY 3: Mean Reversion (Bollinger + Z-Score)
# ─────────────────────────────────────────────
class MeanReversionStrategy(Strategy):
    """
    Buy extreme oversold conditions, sell at mean reversion.
    Uses Bollinger Bands + Z-score confirmation.
    Works well on high-liquidity, range-bound instruments.
    """
    name = "MeanReversion"

    def __init__(
        self,
        bb_period:     int   = 20,
        bb_std:        float = 2.0,
        zscore_period: int   = 20,
        zscore_entry:  float = -2.0,   # enter when z < this
        zscore_exit:   float = 0.0,    # exit when z crosses 0
        rsi_confirm:   int   = 14,
        rsi_oversold:  float = 35.0,
    ):
        self.bb_period     = bb_period
        self.bb_std        = bb_std
        self.zscore_period = zscore_period
        self.zscore_entry  = zscore_entry
        self.zscore_exit   = zscore_exit
        self.rsi_confirm   = rsi_confirm
        self.rsi_oversold  = rsi_oversold

    def generate_signals(self, df: pd.DataFrame):
        close = df["close"]
        upper, mid, lower = I.bollinger(close, self.bb_period, self.bb_std)
        z     = I.zscore(close, self.zscore_period)
        rsi   = I.rsi(close, self.rsi_confirm)

        # Entry: price touches lower band + z-score extreme + RSI oversold
        entries = (
            (close < lower) &
            (z < self.zscore_entry) &
            (rsi < self.rsi_oversold)
        )

        # Exit: price crosses back to mid band OR z-score crosses 0
        exits = (
            (close >= mid) |
            ((z >= self.zscore_exit) & (z.shift(1) < self.zscore_exit))
        )

        return entries.fillna(False), exits.fillna(False)


# ─────────────────────────────────────────────
#  STRATEGY 4: MACD + RSI Hybrid
# ─────────────────────────────────────────────
class MACDRSIStrategy(Strategy):
    """
    MACD histogram flip as primary signal, RSI as confirmation filter.
    One of the most robust multi-timeframe confirmation patterns.
    """
    name = "MACD_RSI"

    def __init__(
        self,
        macd_fast:   int   = 12,
        macd_slow:   int   = 26,
        macd_signal: int   = 9,
        rsi_period:  int   = 14,
        rsi_low:     float = 40.0,   # RSI must be above this to enter
        rsi_high:    float = 75.0,   # exit when RSI goes above this
        trend_filter: bool = True,
    ):
        self.macd_fast    = macd_fast
        self.macd_slow    = macd_slow
        self.macd_signal  = macd_signal
        self.rsi_period   = rsi_period
        self.rsi_low      = rsi_low
        self.rsi_high     = rsi_high
        self.trend_filter = trend_filter

    def generate_signals(self, df: pd.DataFrame):
        close = df["close"]
        macd, sig, hist = I.macd(close, self.macd_fast, self.macd_slow, self.macd_signal)
        rsi = I.rsi(close, self.rsi_period)

        # MACD histogram flip from negative to positive
        hist_cross_up   = (hist > 0) & (hist.shift(1) <= 0)
        hist_cross_down = (hist < 0) & (hist.shift(1) >= 0)

        entries = hist_cross_up & (rsi > self.rsi_low)
        exits   = hist_cross_down | (rsi > self.rsi_high)

        if self.trend_filter:
            ema200 = I.ema(close, 200)
            entries = entries & (close > ema200)

        return entries.fillna(False), exits.fillna(False)


# ─────────────────────────────────────────────
#  STRATEGY 5: Donchian Breakout (Turtle-style)
# ─────────────────────────────────────────────
class DonchianBreakoutStrategy(Strategy):
    """
    Turtle Trader system: enter on N-day high breakout,
    exit on M-day low breakdown (M < N).
    """
    name = "DonchianBreakout"

    def __init__(
        self,
        entry_period: int = 20,
        exit_period:  int = 10,
        atr_period:   int = 14,
        vol_confirm:  bool = True,
    ):
        self.entry_period = entry_period
        self.exit_period  = exit_period
        self.atr_period   = atr_period
        self.vol_confirm  = vol_confirm

    def generate_signals(self, df: pd.DataFrame):
        close = df["close"]
        upper, _, _ = I.donchian(df, self.entry_period)
        _, _, lower = I.donchian(df, self.exit_period)
        vol_avg = I.volume_sma(df, 20)

        # Breakout: today's close exceeds yesterday's N-day high
        entries = close > upper.shift(1)
        exits   = close < lower.shift(1)

        if self.vol_confirm:
            entries = entries & (df["volume"] > vol_avg * 1.2)

        return entries.fillna(False), exits.fillna(False)


# ─────────────────────────────────────────────
#  REGISTRY — easy lookup by name
# ─────────────────────────────────────────────
STRATEGY_REGISTRY = {
    "ma_cross":         MACrossStrategy,
    "momentum":         MomentumStrategy,
    "mean_reversion":   MeanReversionStrategy,
    "macd_rsi":         MACDRSIStrategy,
    "donchian_breakout": DonchianBreakoutStrategy,
}


def get_strategy(name: str, **kwargs) -> Strategy:
    if name not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY)}")
    return STRATEGY_REGISTRY[name](**kwargs)


if __name__ == "__main__":
    # Quick sanity check
    import sys; sys.path.insert(0, ".")
    from data import generate_synthetic
    df = generate_synthetic(756)
    for name, cls in STRATEGY_REGISTRY.items():
        s = cls()
        entries, exits = s.generate_signals(df)
        print(f"{name:25s} | entries={entries.sum():4d} | exits={exits.sum():4d}")
