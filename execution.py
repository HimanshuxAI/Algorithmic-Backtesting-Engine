"""
execution.py — Realistic fill simulation
Covers: slippage (sqrt-impact model), Indian market transaction costs,
        fractional Kelly position sizing, risk management.
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────
#  TRANSACTION COST CONFIG
# ─────────────────────────────────────────────
@dataclass
class CostConfig:
    """
    Indian equity market cost structure (NSE/BSE).
    All rates are per trade unless noted.
    """
    # Brokerage
    brokerage_flat: float    = 20.0       # ₹20 flat per order (Zerodha/Groww style)
    brokerage_pct:  float    = 0.0003     # 0.03% — whichever is higher
    brokerage_max:  float    = 20.0       # cap

    # Statutory charges
    stt_buy:        float    = 0.0        # no STT on buy (delivery: 0.1%, but intraday: 0)
    stt_sell:       float    = 0.001      # 0.1% on sell (delivery)
    exchange_txn:   float    = 0.0000335  # NSE exchange txn charge
    sebi_charge:    float    = 0.000001   # ₹10 per crore = 0.000001 per ₹
    stamp_duty:     float    = 0.00015    # 0.015% on buy
    gst_rate:       float    = 0.18       # 18% on (brokerage + exchange charge)

    # Slippage model
    slippage_eta:   float    = 0.1        # market impact coefficient
    slippage_alpha: float    = 0.5        # power law exponent (0.5 = sqrt model)
    slippage_cap:   float    = 0.005      # max slippage 0.5% per trade

    # Spread (bid-ask)
    spread_bps:     float    = 5.0        # 5 basis points (liquid large-cap)


@dataclass
class RiskConfig:
    """Position sizing and risk management rules."""
    sizing_method:   str   = "kelly_fractional"  # "fixed_pct" | "kelly_fractional" | "atr_based" | "equal"
    kelly_fraction:  float = 0.25         # conservative quarter-Kelly
    max_position_pct: float = 0.08        # max 8% NAV per position
    min_position_pct: float = 0.01        # min 1% NAV (avoid dust trades)
    atr_risk_pct:    float  = 0.01        # risk 1% NAV per ATR unit
    initial_capital: float  = 1_000_000.0 # ₹10 lakh default
    max_open_trades: int    = 10
    stop_loss_atr:   float  = 2.0         # stop = entry - 2 × ATR
    take_profit_atr: float  = 0.0         # 0 = disabled


# ─────────────────────────────────────────────
#  SLIPPAGE MODEL
# ─────────────────────────────────────────────
class SlippageModel:
    """
    Almgren-Chriss square-root market impact model.
    impact = eta × sigma × (order_size / ADV)^alpha
    """

    def __init__(self, config: CostConfig):
        self.cfg = config

    def calc_impact(
        self,
        order_value: float,    # ₹ value of order
        adv: float,            # average daily volume in ₹
        daily_vol: float,      # daily return std dev
        side: str = "buy"
    ) -> float:
        """Returns slippage as fraction of price."""
        if adv <= 0:
            return self.cfg.slippage_cap

        pov = order_value / max(adv, 1)  # participation rate
        impact = self.cfg.slippage_eta * daily_vol * (pov ** self.cfg.slippage_alpha)

        # Add half bid-ask spread
        spread = self.cfg.spread_bps / 10_000 / 2
        total  = impact + spread

        return float(np.clip(total, 0, self.cfg.slippage_cap))

    def apply(self, price: float, side: str, impact: float) -> float:
        """Apply slippage to price."""
        direction = 1 if side == "buy" else -1
        return price * (1 + direction * impact)


# ─────────────────────────────────────────────
#  TRANSACTION COST CALCULATOR
# ─────────────────────────────────────────────
class CostCalculator:

    def __init__(self, config: CostConfig):
        self.cfg = config

    def calc(
        self,
        price: float,
        quantity: int,
        side: str = "buy"
    ) -> dict:
        """
        Calculate all-in transaction cost for a trade.
        Returns breakdown dict + total_cost in ₹.
        """
        turnover = price * quantity
        cfg = self.cfg

        # Brokerage
        brokerage = min(
            max(cfg.brokerage_flat, turnover * cfg.brokerage_pct),
            cfg.brokerage_max
        )

        # STT
        stt = turnover * (cfg.stt_buy if side == "buy" else cfg.stt_sell)

        # Exchange charges
        exchange = turnover * cfg.exchange_txn

        # SEBI
        sebi = turnover * cfg.sebi_charge

        # Stamp duty (buy only)
        stamp = turnover * cfg.stamp_duty if side == "buy" else 0

        # GST on brokerage + exchange
        gst = (brokerage + exchange) * cfg.gst_rate

        total = brokerage + stt + exchange + sebi + stamp + gst

        return {
            "brokerage": round(brokerage, 4),
            "stt":       round(stt, 4),
            "exchange":  round(exchange, 4),
            "sebi":      round(sebi, 4),
            "stamp":     round(stamp, 4),
            "gst":       round(gst, 4),
            "total":     round(total, 4),
            "pct":       round(total / turnover * 100, 4) if turnover > 0 else 0,
        }


# ─────────────────────────────────────────────
#  POSITION SIZER
# ─────────────────────────────────────────────
class PositionSizer:

    def __init__(self, config: RiskConfig):
        self.cfg = config

    def size(
        self,
        portfolio_value: float,
        price: float,
        atr: float,
        win_rate: float   = 0.55,
        avg_win:  float   = 0.04,
        avg_loss: float   = 0.02,
    ) -> tuple[int, float]:
        """
        Returns (quantity, allocation_pct).
        """
        cfg = self.cfg
        method = cfg.sizing_method

        if method == "fixed_pct":
            alloc = portfolio_value * cfg.max_position_pct

        elif method == "kelly_fractional":
            if avg_loss > 0:
                b       = avg_win / avg_loss
                p, q    = win_rate, 1 - win_rate
                kelly_f = (b * p - q) / b
                kelly_f = max(kelly_f, 0)   # no negative sizing
                safe_f  = kelly_f * cfg.kelly_fraction
                alloc   = portfolio_value * safe_f
            else:
                alloc = portfolio_value * cfg.min_position_pct

        elif method == "atr_based":
            if atr > 0 and price > 0:
                risk_per_share = atr * cfg.stop_loss_atr
                risk_budget    = portfolio_value * cfg.atr_risk_pct
                shares = risk_budget / risk_per_share
                alloc  = shares * price
            else:
                alloc = portfolio_value * cfg.min_position_pct

        elif method == "equal":
            alloc = portfolio_value / cfg.max_open_trades

        else:
            raise ValueError(f"Unknown sizing method: {method}")

        # Clamp to [min, max] NAV bands
        alloc = np.clip(
            alloc,
            portfolio_value * cfg.min_position_pct,
            portfolio_value * cfg.max_position_pct
        )

        quantity = max(1, int(alloc / price))
        alloc_pct = (quantity * price) / portfolio_value

        return quantity, alloc_pct


# ─────────────────────────────────────────────
#  COMBINED FILL SIMULATOR
# ─────────────────────────────────────────────
class FillSimulator:
    """
    Wraps slippage + cost into a single fill() call.
    Called by the backtesting engine per trade.
    """

    def __init__(
        self,
        cost_config: CostConfig = None,
        risk_config: RiskConfig = None,
    ):
        self.cost_cfg = cost_config or CostConfig()
        self.risk_cfg = risk_config or RiskConfig()
        self.slippage  = SlippageModel(self.cost_cfg)
        self.costs     = CostCalculator(self.cost_cfg)
        self.sizer     = PositionSizer(self.risk_cfg)

    def fill(
        self,
        raw_price: float,
        side: str,           # "buy" | "sell"
        quantity: int,
        adv: float,
        daily_vol: float,
    ) -> dict:
        """
        Simulate a realistic fill.

        Returns:
          fill_price     - adjusted price after slippage
          quantity       - shares
          gross_value    - fill_price × quantity
          cost_breakdown - dict of all charges
          total_cost     - ₹ total transaction cost
          slippage_pct   - slippage as % of price
          net_value      - gross_value + total_cost (buy) or - total_cost (sell)
        """
        order_value = raw_price * quantity
        impact      = self.slippage.calc_impact(order_value, adv, daily_vol, side)
        fill_price  = self.slippage.apply(raw_price, side, impact)
        cost_info   = self.costs.calc(fill_price, quantity, side)

        gross_value = fill_price * quantity
        total_cost  = cost_info["total"]

        if side == "buy":
            net_value = gross_value + total_cost   # cash out
        else:
            net_value = gross_value - total_cost   # cash in

        return {
            "fill_price":    round(fill_price, 4),
            "quantity":      quantity,
            "gross_value":   round(gross_value, 2),
            "cost_breakdown": cost_info,
            "total_cost":    round(total_cost, 4),
            "slippage_pct":  round(impact * 100, 4),
            "net_value":     round(net_value, 2),
        }


if __name__ == "__main__":
    sim = FillSimulator()
    sizer = sim.sizer

    print("=== POSITION SIZING TEST ===")
    for method in ["fixed_pct", "kelly_fractional", "atr_based", "equal"]:
        sim.risk_cfg.sizing_method = method
        sizer = PositionSizer(sim.risk_cfg)
        qty, pct = sizer.size(
            portfolio_value=1_000_000,
            price=2000,
            atr=45,
            win_rate=0.58,
            avg_win=0.04,
            avg_loss=0.02
        )
        print(f"  {method:22s} → qty={qty:5d} | alloc={pct:.2%}")

    print("\n=== FILL SIMULATION TEST ===")
    sim2 = FillSimulator()
    for side in ["buy", "sell"]:
        fill = sim2.fill(
            raw_price=2000.0,
            side=side,
            quantity=100,
            adv=50_000_000,    # ₹5cr ADV (liquid large-cap)
            daily_vol=0.015
        )
        print(f"\n  {side.upper()} @ ₹2000 × 100 shares")
        print(f"    fill_price  : ₹{fill['fill_price']:.4f}  (slip={fill['slippage_pct']:.4f}%)")
        print(f"    total_cost  : ₹{fill['total_cost']:.2f}")
        print(f"    net_value   : ₹{fill['net_value']:,.2f}")
        print(f"    cost_breakdown: {fill['cost_breakdown']}")
