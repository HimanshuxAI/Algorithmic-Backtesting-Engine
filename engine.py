"""
engine.py — Core backtesting event loop
Bar-by-bar simulation. No look-ahead. Tracks portfolio, trades, equity curve.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
import warnings
warnings.filterwarnings("ignore")

from execution import FillSimulator, CostConfig, RiskConfig, PositionSizer
from strategies import Strategy


# ─────────────────────────────────────────────
#  TRADE RECORD
# ─────────────────────────────────────────────
@dataclass
class Trade:
    trade_id:       int
    symbol:         str
    direction:      str   # "long" | "short"

    entry_date:     pd.Timestamp = None
    entry_price:    float        = 0.0
    entry_fill:     float        = 0.0    # after slippage
    entry_cost:     float        = 0.0
    quantity:       int          = 0

    exit_date:      pd.Timestamp = None
    exit_price:     float        = 0.0
    exit_fill:      float        = 0.0
    exit_cost:      float        = 0.0

    stop_loss:      float        = 0.0
    take_profit:    float        = 0.0

    gross_pnl:      float        = 0.0
    total_cost:     float        = 0.0
    net_pnl:        float        = 0.0
    return_pct:     float        = 0.0
    r_multiple:     float        = 0.0   # P&L in units of initial risk
    hold_days:      int          = 0
    status:         str          = "open"   # "open" | "closed"
    exit_reason:    str          = ""       # "signal" | "stop" | "take_profit" | "end_of_data"
    slippage_total: float        = 0.0


# ─────────────────────────────────────────────
#  PORTFOLIO STATE
# ─────────────────────────────────────────────
@dataclass
class Portfolio:
    initial_capital: float
    cash:            float   = field(init=False)
    open_trades:     list    = field(default_factory=list)
    closed_trades:   list    = field(default_factory=list)
    equity_history:  list    = field(default_factory=list)
    trade_counter:   int     = 0

    def __post_init__(self):
        self.cash = self.initial_capital

    @property
    def open_positions_value(self) -> float:
        return sum(t.quantity * t.entry_fill for t in self.open_trades)

    @property
    def total_value(self) -> float:
        return self.cash + self.open_positions_value

    def mark_to_market(self, price_map: dict) -> float:
        """Recalculate portfolio value using current prices."""
        mtm = self.cash
        for t in self.open_trades:
            mtm += t.quantity * price_map.get(t.symbol, t.entry_fill)
        return mtm

    def is_invested(self, symbol: str) -> bool:
        return any(t.symbol == symbol for t in self.open_trades)


# ─────────────────────────────────────────────
#  BACKTESTING ENGINE
# ─────────────────────────────────────────────
class BacktestEngine:
    """
    Event-driven backtesting engine.
    Bar-by-bar: compute indicators → generate signals → simulate fills → track portfolio.
    """

    def __init__(
        self,
        strategy:    Strategy,
        cost_config: CostConfig = None,
        risk_config: RiskConfig = None,
        symbol:      str        = "ASSET",
        verbose:     bool       = False,
    ):
        self.strategy    = strategy
        self.cost_config = cost_config or CostConfig()
        self.risk_config = risk_config or RiskConfig()
        self.symbol      = symbol
        self.verbose     = verbose
        self.simulator   = FillSimulator(self.cost_config, self.risk_config)

    def run(self, df: pd.DataFrame) -> "BacktestResult":
        """
        Main entry point. df must have columns: open, high, low, close, volume.
        Returns BacktestResult with full trade log + equity curve.
        """
        from data import validate_ohlcv

        df = validate_ohlcv(df.copy(), source="backtest_input")
        if len(df) < 20:
            raise ValueError("Backtest requires at least 20 bars of valid OHLCV data.")

        portfolio = Portfolio(initial_capital=self.risk_config.initial_capital)
        sizer     = PositionSizer(self.risk_config)

        # Pre-compute signals and indicators on FULL data (look-ahead on signals
        # is prevented by strategy design; here we just avoid recomputing per bar)
        entries, exits = self.strategy.generate_signals(df)
        entries = self._normalize_signal(entries, df.index, "entries")
        exits = self._normalize_signal(exits, df.index, "exits")

        # Pre-compute ATR for stop loss and sizing
        from strategies import Indicators
        atr_series = Indicators.atr(df, period=14)

        # Rolling stats for Kelly sizing (expanding window, updated each trade)
        _win_rates  = []
        _avg_wins   = []
        _avg_losses = []

        def _get_kelly_params():
            if len(_win_rates) < 5:
                return 0.55, 0.04, 0.02   # sensible defaults until we have data
            return (
                np.mean(_win_rates[-20:]),
                np.mean(_avg_wins[-20:]),
                np.mean(_avg_losses[-20:]),
            )

        equity_curve = []

        for i, (date, row) in enumerate(df.iterrows()):
            close  = row["close"]
            high   = row["high"]
            low    = row["low"]
            volume = row.get("volume", 0)
            atr    = atr_series.iloc[i] if not np.isnan(atr_series.iloc[i]) else close * 0.02
            adv    = volume * close if volume > 0 else close * 1_000_000
            daily_vol = max(atr / close, 1e-6)

            # ── MARK-TO-MARKET ──
            mtm = portfolio.mark_to_market({self.symbol: close})
            equity_curve.append({"date": date, "equity": mtm, "cash": portfolio.cash})

            # ── CHECK STOPS on open trades ──
            closed_this_bar = []
            for t in list(portfolio.open_trades):
                stop_triggered = (t.stop_loss > 0 and low <= t.stop_loss)
                tp_triggered   = (t.take_profit > 0 and high >= t.take_profit)

                if stop_triggered or tp_triggered:
                    exit_px = t.stop_loss if stop_triggered else t.take_profit
                    reason  = "stop" if stop_triggered else "take_profit"
                    self._close_trade(
                        t, exit_px, date, portfolio, reason,
                        adv=adv, daily_vol=daily_vol
                    )
                    closed_this_bar.append(t)
                    _win_rates.append(1 if t.net_pnl > 0 else 0)
                    _avg_wins.append(t.return_pct  if t.return_pct  > 0 else 0)
                    _avg_losses.append(-t.return_pct if t.return_pct <= 0 else 0)
                    if self.verbose:
                        print(f"[{date.date()}] {reason.upper()} {t.symbol} "
                              f"@ ₹{exit_px:.2f} | PnL ₹{t.net_pnl:.2f}")

            portfolio.open_trades = [t for t in portfolio.open_trades
                                     if t not in closed_this_bar]

            # ── PROCESS STRATEGY EXIT SIGNALS ──
            if exits.iloc[i]:
                for t in list(portfolio.open_trades):
                    if t.symbol == self.symbol:
                        self._close_trade(
                            t, close, date, portfolio, "signal",
                            adv=adv, daily_vol=daily_vol
                        )
                        portfolio.open_trades.remove(t)
                        _win_rates.append(1 if t.net_pnl > 0 else 0)
                        _avg_wins.append(t.return_pct  if t.return_pct  > 0 else 0)
                        _avg_losses.append(-t.return_pct if t.return_pct <= 0 else 0)
                        if self.verbose:
                            print(f"[{date.date()}] EXIT SIGNAL {t.symbol} "
                                  f"@ ₹{close:.2f} | PnL ₹{t.net_pnl:.2f}")

            # ── PROCESS STRATEGY ENTRY SIGNALS ──
            already_in = portfolio.is_invested(self.symbol)
            too_many   = len(portfolio.open_trades) >= self.risk_config.max_open_trades
            has_entry  = entries.iloc[i]

            if has_entry and not already_in and not too_many:
                # Size position
                wr, aw, al = _get_kelly_params()
                qty, alloc_pct = sizer.size(
                    portfolio_value=mtm,
                    price=close,
                    atr=atr,
                    win_rate=wr,
                    avg_win=aw,
                    avg_loss=al
                )

                # Check we can afford it
                required_cash = close * qty * 1.01   # small buffer
                if portfolio.cash < required_cash:
                    qty = max(1, int(portfolio.cash * 0.95 / close))

                if qty < 1:
                    continue

                # Simulate fill
                fill = self.simulator.fill(
                    raw_price=close,
                    side="buy",
                    quantity=qty,
                    adv=adv,
                    daily_vol=daily_vol,
                )

                if fill["net_value"] > portfolio.cash:
                    affordable_qty = int(portfolio.cash / max(close * 1.02, 1))
                    if affordable_qty < 1:
                        continue
                    fill = self.simulator.fill(
                        raw_price=close,
                        side="buy",
                        quantity=affordable_qty,
                        adv=adv,
                        daily_vol=daily_vol,
                    )
                    qty = affordable_qty
                    if fill["net_value"] > portfolio.cash:
                        continue

                # Deduct cash
                portfolio.cash -= fill["net_value"]

                # Set stops
                stop   = fill["fill_price"] - self.risk_config.stop_loss_atr * atr
                tp_atr = self.risk_config.take_profit_atr
                tp     = fill["fill_price"] + tp_atr * atr if tp_atr > 0 else 0

                portfolio.trade_counter += 1
                t = Trade(
                    trade_id    = portfolio.trade_counter,
                    symbol      = self.symbol,
                    direction   = "long",
                    entry_date  = date,
                    entry_price = close,
                    entry_fill  = fill["fill_price"],
                    entry_cost  = fill["total_cost"],
                    quantity    = qty,
                    stop_loss   = stop,
                    take_profit = tp,
                    slippage_total = fill["slippage_pct"],
                )
                portfolio.open_trades.append(t)

                if self.verbose:
                    print(f"[{date.date()}] BUY {qty}×{self.symbol} "
                          f"@ ₹{fill['fill_price']:.2f} "
                          f"| stop=₹{stop:.2f} | cost=₹{fill['total_cost']:.2f}")

        # ── CLOSE ALL REMAINING AT EOD ──
        last_close = df["close"].iloc[-1]
        last_date  = df.index[-1]
        last_volume = df["volume"].iloc[-1] if "volume" in df.columns else 0
        last_atr = atr_series.iloc[-1] if not np.isnan(atr_series.iloc[-1]) else last_close * 0.02
        last_adv = last_volume * last_close if last_volume > 0 else last_close * 1_000_000
        last_daily_vol = max(last_atr / last_close, 1e-6)
        for t in list(portfolio.open_trades):
            self._close_trade(
                t, last_close, last_date, portfolio, "end_of_data",
                adv=last_adv, daily_vol=last_daily_vol
            )
        portfolio.open_trades.clear()

        equity_df = pd.DataFrame(equity_curve).set_index("date")
        return BacktestResult(
            portfolio      = portfolio,
            equity_curve   = equity_df,
            df             = df,
            strategy       = self.strategy,
            entries        = entries,
            exits          = exits,
        )

    def _close_trade(
        self,
        t:          Trade,
        exit_price: float,
        date:       pd.Timestamp,
        portfolio:  Portfolio,
        reason:     str,
        adv:        float,
        daily_vol:  float,
    ):
        fill = self.simulator.fill(
            raw_price  = exit_price,
            side       = "sell",
            quantity   = t.quantity,
            adv        = adv,
            daily_vol  = daily_vol,
        )

        portfolio.cash   += fill["net_value"]
        t.exit_date       = date
        t.exit_price      = exit_price
        t.exit_fill       = fill["fill_price"]
        t.exit_cost       = fill["total_cost"]
        t.hold_days       = (date - t.entry_date).days
        t.total_cost      = t.entry_cost + t.exit_cost
        t.slippage_total += fill["slippage_pct"]

        t.gross_pnl  = (t.exit_fill - t.entry_fill) * t.quantity
        t.net_pnl    = t.gross_pnl - t.total_cost
        invested     = t.entry_fill * t.quantity
        t.return_pct = t.net_pnl / invested if invested > 0 else 0

        # R-multiple: P&L in units of initial risk
        initial_risk = (t.entry_fill - t.stop_loss) * t.quantity if t.stop_loss > 0 else invested * 0.02
        t.r_multiple = t.net_pnl / initial_risk if initial_risk > 0 else 0

        t.exit_reason = reason
        t.status      = "closed"
        portfolio.closed_trades.append(t)

    @staticmethod
    def _normalize_signal(signal, index: pd.Index, label: str) -> pd.Series:
        """Ensure strategy outputs are boolean series aligned to the input index."""
        if not isinstance(signal, pd.Series):
            signal = pd.Series(signal, index=index)
        if len(signal) != len(index):
            raise ValueError(
                f"Strategy produced {label} of length {len(signal)} for index length {len(index)}."
            )
        signal = signal.reindex(index)
        if signal.isna().all():
            return pd.Series(False, index=index)
        return signal.fillna(False).astype(bool)


# ─────────────────────────────────────────────
#  BACKTEST RESULT (raw output)
# ─────────────────────────────────────────────
@dataclass
class BacktestResult:
    portfolio:    Portfolio
    equity_curve: pd.DataFrame
    df:           pd.DataFrame
    strategy:     Strategy
    entries:      pd.Series
    exits:        pd.Series

    @property
    def trades(self) -> list[Trade]:
        return self.portfolio.closed_trades

    @property
    def trade_df(self) -> pd.DataFrame:
        if not self.trades:
            return pd.DataFrame()
        rows = []
        for t in self.trades:
            rows.append({
                "trade_id":     t.trade_id,
                "symbol":       t.symbol,
                "direction":    t.direction,
                "entry_date":   t.entry_date,
                "exit_date":    t.exit_date,
                "entry_fill":   round(t.entry_fill, 4),
                "exit_fill":    round(t.exit_fill, 4),
                "quantity":     t.quantity,
                "hold_days":    t.hold_days,
                "gross_pnl":    round(t.gross_pnl, 2),
                "total_cost":   round(t.total_cost, 4),
                "net_pnl":      round(t.net_pnl, 2),
                "return_pct":   round(t.return_pct * 100, 4),
                "r_multiple":   round(t.r_multiple, 4),
                "slippage_pct": round(t.slippage_total, 4),
                "exit_reason":  t.exit_reason,
                "stop_loss":    round(t.stop_loss, 4),
            })
        return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys; sys.path.insert(0, ".")
    from data import generate_synthetic
    from strategies import MACrossStrategy

    df = generate_synthetic(756)
    strategy = MACrossStrategy(fast=20, slow=50)
    engine = BacktestEngine(strategy, verbose=True,
                            risk_config=RiskConfig(sizing_method="kelly_fractional"))
    result = engine.run(df)

    print(f"\nClosed trades: {len(result.trades)}")
    print(result.trade_df.tail(5).to_string())
