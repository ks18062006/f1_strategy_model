

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# typical stint lengths per compound
COMPOUND_AVG_STINT = {
    "SOFT": 20, "MEDIUM": 30, "HARD": 38,
    "INTERMEDIATE": 25, "WET": 20,
}

FEATURE_COLUMNS = [
    # tyre stuff
    "tyre_age_laps", "stint_number", "tyre_age_normalised",
    "compound_SOFT", "compound_MEDIUM", "compound_HARD",
    "compound_INTERMEDIATE", "compound_WET",
    # pace and degradation
    "lap_time_seconds", "lap_time_delta_vs_race_median",
    "rolling_mean_last3", "rolling_std_last3",
    "lap_time_delta_vs_stint_mean", "degradation_rate",
    # race context
    "lap_number", "lap_number_normalised", "position",
    "gap_ahead", "gap_behind", "safety_car_active",
    "team_encoded", "driver_encoded",
]


class FeatureEngineer:
    """Stateful fit/transform pipeline for pit prediction features."""

    def __init__(self, rolling_window: int = 3) -> None:
        self.rolling_window = rolling_window
        self._race_lap_medians = {}
        self._race_total_laps = {}
        self._driver_encoder = LabelEncoder()
        self._team_encoder = LabelEncoder()
        self._fitted = False

    def fit_transform(self, df: pd.DataFrame):
        """Fit on train data, return (X, y)."""
        self._fit(df)
        return self._transform(df)

    def transform(self, df: pd.DataFrame):
        """Transform held-out data using fitted stats."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform() first.")
        return self._transform(df, is_train=False)

    def get_feature_names(self):
        """Get ordered feature column names."""
        return FEATURE_COLUMNS.copy()

    def _fit(self, df: pd.DataFrame) -> None:
        """Learn race-level stats and encoders."""
        self._race_lap_medians = df.groupby("race_id")["LapTimeSeconds"].median().to_dict()
        self._race_total_laps = df.groupby("race_id")["LapNumber"].max().to_dict()

        # unknown catch-all for unseen values
        drivers = list(df["Driver"].unique()) + ["UNKNOWN"]
        teams = list(df["Team"].unique()) + ["UNKNOWN"]

        self._driver_encoder.fit(drivers)
        self._team_encoder.fit(teams)
        self._fitted = True
        logger.info("Fitted — %d races, %d drivers, %d teams",
                     len(self._race_lap_medians), len(drivers) - 1, len(teams) - 1)

    def _transform(self, df: pd.DataFrame, is_train: bool = True):
        """Run the full feature pipeline."""
        df = df.copy().sort_values(["race_id", "Driver", "LapNumber"])

        df = self._tyre_features(df)
        df = self._pace_features(df)
        df = self._race_context_features(df)
        df = self._encode_identities(df, is_train)

        # fill any missing features with 0
        for col in FEATURE_COLUMNS:
            if col not in df.columns:
                df[col] = 0

        X = df[FEATURE_COLUMNS].copy()
        y = df["WillPitNext3"].astype(int)

        logger.info("Transform done — X: %s | positive: %.2f%%", X.shape, 100 * y.mean())
        return X, y

    def _tyre_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compound, age, stint, normalised age."""
        df = df.copy()

        df["tyre_age_laps"] = df["TyreLife"].fillna(0).astype(float)

        # stint number via forward fill
        df["stint_number"] = (
            df.groupby(["race_id", "Driver"])["Stint"]
            .transform(lambda x: x.ffill().astype(int))
        )

        # age relative to typical stint
        df["tyre_age_normalised"] = df.apply(
            lambda r: r["tyre_age_laps"] / COMPOUND_AVG_STINT.get(str(r["Compound"]).upper(), 30),
            axis=1,
        )

        # one-hot compounds
        dummies = pd.get_dummies(df["Compound"].str.upper(), prefix="compound")
        for col in ["compound_SOFT", "compound_MEDIUM", "compound_HARD",
                     "compound_INTERMEDIATE", "compound_WET"]:
            if col in dummies.columns:
                df[col] = dummies[col].astype(int)
            else:
                df[col] = 0

        return df

    def _pace_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lap deltas, rolling stats, deg rate."""
        df = df.copy()

        df["lap_time_seconds"] = df["LapTimeSeconds"]

        # pace vs race median
        df["lap_time_delta_vs_race_median"] = df.apply(
            lambda r: r["LapTimeSeconds"] - self._race_lap_medians.get(r["race_id"], r["LapTimeSeconds"]),
            axis=1,
        )

        # rolling window stats
        group = df.groupby(["race_id", "Driver"])["LapTimeSeconds"]
        df["rolling_mean_last3"] = group.transform(
            lambda x: x.rolling(self.rolling_window, min_periods=1).mean()
        )
        df["rolling_std_last3"] = group.transform(
            lambda x: x.rolling(self.rolling_window, min_periods=1).std().fillna(0)
        )

        # vs stint average
        stint_mean = df.groupby(["race_id", "Driver", "stint_number"])["LapTimeSeconds"].transform("mean")
        df["lap_time_delta_vs_stint_mean"] = df["LapTimeSeconds"] - stint_mean

        # deg rate vs lap 3 of stint
        def degradation_rate(group):
            group = group.copy()
            ref_mask = group["tyre_age_laps"] == 3
            if ref_mask.any():
                ref_time = group.loc[ref_mask, "LapTimeSeconds"].iloc[0]
            else:
                ref_time = group["LapTimeSeconds"].iloc[0]
            group["degradation_rate"] = group["LapTimeSeconds"] - ref_time
            return group

        df = df.groupby(
            ["race_id", "Driver", "stint_number"], group_keys=False
        ).apply(degradation_rate)

        return df

    def _race_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Position, gaps, lap progress, SC flag."""
        df = df.copy()

        df["lap_number"] = df["LapNumber"].astype(float)

        # race progress 0→1
        df["lap_number_normalised"] = df.apply(
            lambda r: r["LapNumber"] / max(self._race_total_laps.get(r["race_id"], r["LapNumber"]), 1),
            axis=1,
        )

        df["position"] = pd.to_numeric(df["Position"], errors="coerce").fillna(10)

        # gap to car ahead/behind
        df["gap_ahead"] = (
            pd.to_numeric(df.get("GapToLeader", pd.Series(dtype=float)), errors="coerce")
            .fillna(0).clip(upper=60)
        )
        df["gap_behind"] = df["gap_ahead"].shift(-1).fillna(0)

        df["safety_car_active"] = df.get("SafetyCarActive", pd.Series(0, index=df.index)).fillna(0).astype(int)

        return df

    def _encode_identities(self, df: pd.DataFrame, is_train: bool) -> pd.DataFrame:
        """Encode driver and team labels."""
        df = df.copy()

        known_drivers = set(self._driver_encoder.classes_)
        known_teams = set(self._team_encoder.classes_)

        df["_driver_safe"] = df["Driver"].apply(lambda x: x if x in known_drivers else "UNKNOWN")
        df["_team_safe"] = df["Team"].apply(lambda x: x if x in known_teams else "UNKNOWN")

        df["driver_encoded"] = self._driver_encoder.transform(df["_driver_safe"])
        df["team_encoded"] = self._team_encoder.transform(df["_team_safe"])

        df = df.drop(columns=["_driver_safe", "_team_safe"])
        return df
