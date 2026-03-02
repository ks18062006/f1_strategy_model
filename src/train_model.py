"""
train_model.py — trains XGBoost pit-stop predictor with race-based splitting.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.utils.class_weight import compute_sample_weight

from src.data_builder import TEST_RACES_2023
from src.feature_engineering import FeatureEngineer

logger = logging.getLogger(__name__)

MODEL_OUTPUT_PATH = Path("data/processed/xgb_pit_model.pkl")
FE_OUTPUT_PATH = Path("data/processed/feature_engineer.pkl")

# tuned for ~5-10% positive rate
XGB_PARAMS = {
    "n_estimators": 400,
    "max_depth": 5,
    "learning_rate": 0.07,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "random_state": 42,
    "n_jobs": -1,
}


def train(
    df_raw: pd.DataFrame,
    test_races: list = TEST_RACES_2023,
    save: bool = True,
    early_stopping_rounds: int = 30,
):
    """Full pipeline: split → features → fit → save."""
    logger.info("=== F1 Pit Window Predictor — Training ===")

    # split by race, not randomly
    df_train_raw, df_test_raw = _split_by_race(df_raw, test_races)
    logger.info(
        "Train: %d rows (%d races) | Test: %d rows (%d races)",
        len(df_train_raw), df_train_raw["race_name"].nunique(),
        len(df_test_raw), df_test_raw["race_name"].nunique(),
    )

    # engineer features
    fe = FeatureEngineer()
    X_train, y_train = fe.fit_transform(df_train_raw)
    X_test, y_test = fe.transform(df_test_raw)

    logger.info(
        "Class balance — Train: %.1f%% pit | Test: %.1f%% pit",
        100 * y_train.mean(), 100 * y_test.mean()
    )

    # handle class imbalance
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    scale_pos_weight = neg / pos
    logger.info("scale_pos_weight = %.2f  (neg=%d, pos=%d)", scale_pos_weight, neg, pos)

    # train xgboost
    params = {
        **XGB_PARAMS,
        "scale_pos_weight": scale_pos_weight,
        "early_stopping_rounds": early_stopping_rounds
    }
    model = xgb.XGBClassifier(**params)

    eval_set = [(X_train, y_train), (X_test, y_test)]
    model.fit(X_train, y_train, eval_set=eval_set, verbose=50)

    logger.info("Best iteration: %d", model.best_iteration)

    # save to disk
    if save:
        _save_artifacts(model, fe)

    # results for evaluation
    y_prob_test = model.predict_proba(X_test)[:, 1]
    results = {
        "best_iteration": model.best_iteration,
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "test_positive_rate": float(y_test.mean()),
        "y_test": y_test,
        "y_prob_test": y_prob_test,
        "X_test": X_test,
        "df_test_raw": df_test_raw,
    }

    logger.info("Training complete.")
    return model, fe, results


def _split_by_race(df: pd.DataFrame, test_races: list):
    """Race-aware split, case-insensitive matching."""
    test_lower = [r.lower() for r in test_races]
    mask = df["race_name"].apply(lambda x: any(t in x.lower() for t in test_lower))
    return df[~mask].copy(), df[mask].copy()


def _save_artifacts(model, fe) -> None:
    """Pickle model and feature engineer."""
    MODEL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(MODEL_OUTPUT_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(FE_OUTPUT_PATH, "wb") as f:
        pickle.dump(fe, f)
    logger.info("Model saved to %s", MODEL_OUTPUT_PATH)
    logger.info("FeatureEngineer saved to %s", FE_OUTPUT_PATH)


def load_model():
    """Load model and fe from disk."""
    with open(MODEL_OUTPUT_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FE_OUTPUT_PATH, "rb") as f:
        fe = pickle.load(f)
    logger.info("Loaded model from disk.")
    return model, fe


if __name__ == "__main__":
    from src.data_builder import RaceDataBuilder, DRY_RACES_2023

    builder = RaceDataBuilder(cache_dir="data/raw")
    df_raw = builder.build_season_dataset(year=2023, race_names=DRY_RACES_2023)
    model, fe, results = train(df_raw)
    print(f"\nDone. Best iteration: {results['best_iteration']}")
