"""
evaluate.py — model evaluation, plots, and strategy table outputs.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
)

from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

PLOT_DIR = Path("data/processed/plots")
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# F1-themed colour palette
COLORS = {
    "primary": "#E8002D",
    "secondary": "#1E1E1E",
    "accent": "#FFFFFF",
    "soft": "#FF6B6B",
    "grid": "#2A2A2A",
    "highlight": "#FFD700",
}


class Evaluator:
    """Evaluation and visualisation for pit predictor."""

    def __init__(self, model, fe, results: dict) -> None:
        self.model = model
        self.fe = fe
        self.results = results

    def run_all(self, race="Netherlands", driver="VER", snapshot_lap=25):
        """Run full eval suite."""
        self.print_metrics()
        self.plot_pr_curve()
        self.plot_pit_probability_curve(race=race, driver=driver)
        self.plot_feature_importance()
        self.print_strategy_table(race=race, lap=snapshot_lap)

    def print_metrics(self):
        """Log PR-AUC and confusion matrix."""
        y_test = self.results["y_test"]
        y_prob = self.results["y_prob_test"]

        pr_auc = average_precision_score(y_test, y_prob)
        recall_at_70p = self._recall_at_precision(y_test, y_prob, 0.70)
        cm = confusion_matrix(y_test, (y_prob >= 0.5).astype(int))

        logger.info("PR-AUC: %.4f | Recall@70%%Prec: %.4f", pr_auc, recall_at_70p)
        logger.info("Confusion Matrix:\n%s", cm)

        return {"pr_auc": pr_auc, "recall_at_70_precision": recall_at_70p, "confusion_matrix": cm}

    def plot_pr_curve(self, save=True):
        """Precision-Recall curve."""
        y_test = self.results["y_test"]
        y_prob = self.results["y_prob_test"]

        precision, recall, _ = precision_recall_curve(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        baseline = y_test.mean()

        fig, ax = plt.subplots(figsize=(8, 6), facecolor=COLORS["secondary"])
        ax.set_facecolor(COLORS["secondary"])
        ax.plot(recall, precision, color=COLORS["primary"], lw=2.5,
                label=f"XGBoost (PR-AUC = {pr_auc:.3f})")
        ax.axhline(baseline, color="grey", lw=1.2, ls="--",
                   label=f"Random = {baseline:.3f}")
        ax.fill_between(recall, precision, alpha=0.12, color=COLORS["primary"])
        _style_ax(ax, "Recall", "Precision", "Precision–Recall Curve")
        ax.legend(facecolor=COLORS["secondary"], labelcolor=COLORS["accent"], fontsize=10)
        plt.tight_layout()

        if save:
            path = PLOT_DIR / "pr_curve.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved %s", path)
        plt.close(fig)

    def plot_pit_probability_curve(self, race="Netherlands", driver="VER", save=True):
        """Pit probability over laps for a driver."""
        df_raw = self.results["df_test_raw"]
        mask = (
            df_raw["race_name"].str.contains(race, case=False, na=False)
            & (df_raw["Driver"] == driver)
        )
        df_driver = df_raw[mask].copy()

        if df_driver.empty:
            logger.warning("No data for %s @ %s", driver, race)
            return

        X_driver, _ = self.fe.transform(df_driver)
        df_driver["pit_prob"] = self.model.predict_proba(X_driver)[:, 1]
        df_driver = df_driver.sort_values("LapNumber")

        # actual pit laps
        pit_laps = df_raw[
            df_raw["race_name"].str.contains(race, case=False, na=False)
            & (df_raw["Driver"] == driver)
            & df_raw["PitInTime"].notna()
        ]["LapNumber"].tolist()

        fig, ax = plt.subplots(figsize=(12, 5), facecolor=COLORS["secondary"])
        ax.set_facecolor(COLORS["secondary"])
        ax.plot(df_driver["LapNumber"], df_driver["pit_prob"],
                color=COLORS["primary"], lw=2, label="Pit probability")
        ax.fill_between(df_driver["LapNumber"], df_driver["pit_prob"],
                        alpha=0.15, color=COLORS["primary"])

        for lap in pit_laps:
            ax.axvline(lap, color=COLORS["highlight"], lw=1.8, ls="--", alpha=0.9)
        ax.axvline(-999, color=COLORS["highlight"], lw=1.8, ls="--", label="Actual pit stop")

        ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
        ax.axhline(0.5, color="grey", lw=0.8, ls=":", alpha=0.6)
        _style_ax(ax, "Lap", "Pit Probability", f"Pit Probability — {driver} @ {race}")
        ax.legend(facecolor=COLORS["secondary"], labelcolor=COLORS["accent"], fontsize=10)
        plt.tight_layout()

        if save:
            path = PLOT_DIR / f"pit_prob_{driver}_{race.lower()}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved %s", path)
        plt.close(fig)

    def plot_feature_importance(self, top_n=15, save=True):
        """Horizontal bar chart of feature gains."""
        importances = self.model.get_booster().get_score(importance_type="gain")
        feature_names = self.fe.get_feature_names()

        imp_series = pd.Series(
            {f: importances.get(f"f{i}", 0) for i, f in enumerate(feature_names)},
        ).sort_values(ascending=True).tail(top_n)

        fig, ax = plt.subplots(figsize=(9, 0.45 * top_n + 2), facecolor=COLORS["secondary"])
        ax.set_facecolor(COLORS["secondary"])
        bars = ax.barh(imp_series.index, imp_series.values,
                       color=COLORS["primary"], edgecolor="none", height=0.65)

        for bar, val in zip(bars, imp_series.values):
            ax.text(bar.get_width() + imp_series.values.max() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.1f}", va="center", color=COLORS["accent"], fontsize=9)

        _style_ax(ax, "Gain", "Feature", "Feature Importance")
        plt.tight_layout()

        if save:
            path = PLOT_DIR / "feature_importance.png"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            logger.info("Saved %s", path)
        plt.close(fig)

    def plot_shap_waterfall(self, race="Netherlands", driver="VER", lap=25, save=True):
        """SHAP waterfall for one prediction."""
        df_raw = self.results["df_test_raw"]
        mask = (
            df_raw["race_name"].str.contains(race, case=False, na=False)
            & (df_raw["Driver"] == driver)
            & (df_raw["LapNumber"] == lap)
        )
        df_snap = df_raw[mask].copy()

        if df_snap.empty:
            logger.warning("No data for %s @ %s lap %d", driver, race, lap)
            return None

        X_snap, _ = self.fe.transform(df_snap)

        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(X_snap)

        fig, ax = plt.subplots(figsize=(10, 6), facecolor=COLORS["secondary"])
        ax.set_facecolor(COLORS["secondary"])
        shap.plots.waterfall(shap_values[0], show=False)

        fig = plt.gcf()
        fig.set_facecolor(COLORS["secondary"])
        ax = plt.gca()
        ax.set_facecolor(COLORS["secondary"])
        ax.tick_params(colors=COLORS["accent"])
        for spine in ax.spines.values():
            spine.set_edgecolor(COLORS["grid"])

        plt.tight_layout()
        if save:
            path = PLOT_DIR / f"shap_{driver}_{race.lower().replace(' ', '_')}_{lap}.png"
            fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            logger.info("Saved %s", path)

        return fig

    def print_strategy_table(self, race="Netherlands", lap=25):
        """Snapshot of all drivers at a lap."""
        df_raw = self.results["df_test_raw"]
        mask = (
            df_raw["race_name"].str.contains(race, case=False, na=False)
            & (df_raw["LapNumber"] == lap)
        )
        df_snap = df_raw[mask].copy()

        if df_snap.empty:
            logger.warning("No data at lap %d for %s", lap, race)
            return pd.DataFrame()

        X_snap, _ = self.fe.transform(df_snap)
        df_snap["PitProb"] = self.model.predict_proba(X_snap)[:, 1]

        table = (
            df_snap[["Driver", "Team", "Position", "TyreLife", "Compound", "PitProb"]]
            .sort_values("PitProb", ascending=False)
            .reset_index(drop=True)
        )
        table["PitProb"] = table["PitProb"].map(lambda x: f"{x:.2%}")
        table = table.rename(columns={"TyreLife": "TyreAge"})

        print(f"\n{'='*60}")
        print(f"  STRATEGY TABLE — {race.upper()} | LAP {lap}")
        print(f"{'='*60}")
        print(table.to_string(index=False))
        print(f"{'='*60}\n")

        return table

    @staticmethod
    def _recall_at_precision(y_true, y_prob, target_precision=0.70):
        """Recall at a given precision threshold."""
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        mask = precision >= target_precision
        return float(recall[mask].max()) if mask.any() else 0.0


def _style_ax(ax, xlabel, ylabel, title):
    """Dark F1-themed axis styling."""
    ax.set_xlabel(xlabel, color=COLORS["accent"], fontsize=11)
    ax.set_ylabel(ylabel, color=COLORS["accent"], fontsize=11)
    ax.set_title(title, color=COLORS["accent"], fontsize=13, fontweight="bold", pad=14)
    ax.tick_params(colors=COLORS["accent"])
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])
    ax.grid(True, color=COLORS["grid"], linewidth=0.6, linestyle="--", alpha=0.5)
