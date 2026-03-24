"""
app.py — Minimal frontend for browsing backtest outputs.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import pandas as pd
from flask import Flask, abort, flash, redirect, render_template, request, send_from_directory, url_for

from main import run_pipeline


DEFAULT_OUTPUT_DIR = "output"
SOURCE_OPTIONS = ["synthetic", "yfinance", "csv"]
SIZING_OPTIONS = ["kelly_fractional", "fixed_pct", "atr_based", "equal"]
VISUAL_ARTIFACTS = [
    ("Best Strategy Dashboard", "dashboard.png"),
    ("Strategy Comparison", "strategy_comparison.png"),
    ("Walk-Forward Validation", "walk_forward.png"),
    ("Monte Carlo Simulation", "monte_carlo.png"),
]
DOWNLOAD_ARTIFACTS = [
    "research_report.md",
    "research_report.json",
    "performance_summary.csv",
    "strategy_rankings.csv",
    "strategy_health.csv",
    "regime_summary.csv",
    "run_manifest.json",
    "pipeline.log",
]


def create_app(output_dir: str | Path = DEFAULT_OUTPUT_DIR) -> Flask:
    app = Flask(__name__)
    app.config["OUTPUT_DIR"] = str(Path(output_dir))
    app.secret_key = os.environ.get("FLASK_SECRET_KEY", "quant-backtest-dashboard")

    @app.template_filter("pretty")
    def pretty(value):
        return _pretty(value)

    @app.route("/")
    def index():
        context = _build_dashboard_context(Path(app.config["OUTPUT_DIR"]))
        return render_template("index.html", **context)

    @app.post("/run")
    def run_backtest():
        output_path = Path(app.config["OUTPUT_DIR"])
        form_values = _extract_form_values(request.form)
        mode = request.form.get("mode", "manual")

        try:
            if mode == "demo":
                run_pipeline(
                    data_source="synthetic",
                    output_dir=output_path,
                    initial_capital=1_000_000,
                    sizing_method="kelly_fractional",
                    strict=False,
                )
                flash("Demo backtest finished. Dashboard updated with fresh demo output.", "success")
                return redirect(url_for("index"))

            run_kwargs = _build_run_kwargs(form_values, output_path)
            run_pipeline(**run_kwargs)
            flash("Manual backtest finished successfully.", "success")
            return redirect(url_for("index"))
        except Exception as exc:
            context = _build_dashboard_context(output_path, form_values=form_values)
            context["form_error"] = str(exc)
            return render_template("index.html", **context), 400

    @app.route("/artifacts/<path:filename>")
    def artifacts(filename: str):
        output_path = Path(app.config["OUTPUT_DIR"])
        target = (output_path / filename).resolve()
        if output_path.resolve() not in target.parents and target != output_path.resolve():
            abort(404)
        if not target.exists():
            abort(404)
        return send_from_directory(output_path, filename)

    return app


def _build_dashboard_context(output_dir: Path, form_values: dict | None = None) -> dict:
    manifest = _load_json(output_dir / "run_manifest.json")
    report = _load_json(output_dir / "research_report.json")
    health = _load_csv_rows(output_dir / "strategy_health.csv")
    leaderboard = report.get("leaderboard", [])
    best = report.get("best_strategy")
    dataset = report.get("dataset", {})

    visuals = [
        {"label": label, "filename": filename}
        for label, filename in VISUAL_ARTIFACTS
        if (output_dir / filename).exists()
    ]
    downloads = [
        {"label": filename, "filename": filename}
        for filename in DOWNLOAD_ARTIFACTS
        if (output_dir / filename).exists()
    ]

    return {
        "title": "Quant Backtest Dashboard",
        "output_ready": bool(manifest or report),
        "manifest": manifest,
        "dataset": dataset,
        "leaderboard": leaderboard,
        "best": best,
        "health": health,
        "visuals": visuals,
        "downloads": downloads,
        "has_failures": any(row.get("status") == "failed" for row in health),
        "form_values": form_values or _default_form_values(),
        "source_options": SOURCE_OPTIONS,
        "sizing_options": SIZING_OPTIONS,
        "form_error": None,
    }


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return _sanitize(json.load(handle))


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    return _sanitize(frame.to_dict(orient="records"))


def _sanitize(value):
    if isinstance(value, dict):
        return {key: _sanitize(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return round(value, 4)
    return value


def _pretty(value):
    if value is None or value == "":
        return "N/A"
    if isinstance(value, (int,)):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def _default_form_values() -> dict:
    return {
        "source": "yfinance",
        "file": "",
        "symbol": "RELIANCE.NS",
        "start": "2021-01-01",
        "end": "2024-01-01",
        "capital": "1000000",
        "sizing": "kelly_fractional",
        "strict": "",
    }


def _extract_form_values(form_data) -> dict:
    defaults = _default_form_values()
    values = {
        key: form_data.get(key, defaults.get(key, "")).strip()
        for key in defaults
    }
    if form_data.get("strict"):
        values["strict"] = "on"
    return values


def _build_run_kwargs(form_values: dict, output_dir: Path) -> dict:
    capital_raw = form_values.get("capital", "1000000")
    try:
        capital = float(capital_raw)
    except ValueError as exc:
        raise ValueError("Capital must be a valid number.") from exc

    return {
        "data_source": form_values.get("source", "synthetic"),
        "filepath": form_values.get("file") or None,
        "symbol": form_values.get("symbol") or "RELIANCE.NS",
        "start": form_values.get("start") or "2021-01-01",
        "end": form_values.get("end") or "2024-01-01",
        "output_dir": output_dir,
        "initial_capital": capital,
        "sizing_method": form_values.get("sizing") or "kelly_fractional",
        "strict": form_values.get("strict") == "on",
    }


app = create_app(os.environ.get("BACKTEST_OUTPUT_DIR", DEFAULT_OUTPUT_DIR))


if __name__ == "__main__":
    app.run(debug=False, host="127.0.0.1", port=8000)
