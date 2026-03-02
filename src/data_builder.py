"""
data_builder.py — loads F1 race data via FastF1 and builds per-driver-lap DataFrames.
"""

import logging
from pathlib import Path
from typing import Optional

import fastf1
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

# dry races only, no rain
DRY_RACES_2023 = [
    "Bahrain", "Saudi Arabia", "Australia", "Azerbaijan", "Miami",
    "Monaco", "Spain", "Canada", "Austria", "British",
    "Hungary", "Belgium", "Netherlands", "Italy",
]

# held out for testing
TEST_RACES_2023 = ["Netherlands", "Italy"]


class RaceDataBuilder:
    """Builds flat driver-lap DataFrames from FastF1 sessions."""

    def __init__(self, cache_dir: str = "data/raw") -> None:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        fastf1.Cache.enable_cache(cache_dir)
        logger.info("FastF1 cache enabled at: %s", cache_dir)

    def build_season_dataset(
        self,
        year: int = 2023,
        race_names: list = DRY_RACES_2023,
    ) -> pd.DataFrame:
        """Stack all races into one DataFrame."""
        frames = []

        for race_name in race_names:
            logger.info("Loading %s %s ...", year, race_name)
            try:
                df = self._load_single_race(year, race_name)
                if df is not None and not df.empty:
                    frames.append(df)
                    logger.info("  ✓ %s — %d rows", race_name, len(df))
            except Exception as exc:
                logger.warning("  ✗ Skipping %s: %s", race_name, exc)

        if not frames:
            raise RuntimeError("No race data loaded. Check FastF1 connectivity.")

        combined = pd.concat(frames, ignore_index=True)
        logger.info(
            "Dataset assembled: %d rows, %d races",
            len(combined), combined["race_name"].nunique(),
        )
        return combined

    def load_single_race(self, year: int, race_name: str) -> pd.DataFrame:
        """Public wrapper for debugging."""
        return self._load_single_race(year, race_name)

    def _load_single_race(self, year: int, race_name: str) -> Optional[pd.DataFrame]:
        """Fetch, clean, and label one race."""
        session = fastf1.get_session(year, race_name, "R")
        session.load(telemetry=False, weather=True, messages=True)

        laps = session.laps.copy()
        laps["LapTimeSeconds"] = laps["LapTime"].dt.total_seconds()

        # target needs PitInTime, so build before cleaning
        laps = self._add_pit_target(laps)
        laps = self._add_safety_car_flag(laps, session)
        laps = self._basic_cleaning(laps)
        laps = self._add_race_metadata(laps, race_name, year)

        return laps

    def _basic_cleaning(self, laps: pd.DataFrame) -> pd.DataFrame:
        """Drop junk rows and outlap noise."""
        laps = laps.copy()

        required = ["LapTimeSeconds", "TyreLife", "Compound", "Position"]
        laps = laps.dropna(subset=required)

        # ditch out-laps
        laps = laps[laps["PitOutTime"].isna()]

        # trim extreme lap times
        q_low = laps["LapTimeSeconds"].quantile(0.01)
        q_high = laps["LapTimeSeconds"].quantile(0.99)
        laps = laps[laps["LapTimeSeconds"].between(q_low, q_high)]

        logger.debug("  After cleaning: %d laps", len(laps))
        return laps

    def _add_safety_car_flag(self, laps: pd.DataFrame, session) -> pd.DataFrame:
        """Flag laps under SC/VSC."""
        laps = laps.copy()

        try:
            track_status = session.track_status
            sc_laps = set()
            for _, row in track_status.iterrows():
                if row["Status"] in ("4", "5"):
                    mask = (laps["Time"] >= row["Time"]) & (
                        laps["Time"] <= row.get("EndTime", laps["Time"].max())
                    )
                    sc_laps.update(laps.loc[mask, "LapNumber"].unique().tolist())

            laps["SafetyCarActive"] = laps["LapNumber"].isin(sc_laps).astype(int)
        except Exception:
            laps["SafetyCarActive"] = 0

        return laps

    def _add_pit_target(self, laps: pd.DataFrame) -> pd.DataFrame:
        """y=1 if driver pits within next 3 laps."""
        laps = laps.copy()
        laps = laps.sort_values(["Driver", "LapNumber"])

        pit_laps = laps[laps["PitInTime"].notna()][["Driver", "LapNumber"]].copy()
        pit_laps = pit_laps.rename(columns={"LapNumber": "PitLap"})

        laps = laps.merge(pit_laps, on="Driver", how="left")

        def compute_target(row):
            if pd.isna(row["PitLap"]):
                return 0
            gap = row["PitLap"] - row["LapNumber"]
            return int(1 <= gap <= 3)

        laps["WillPitNext3"] = laps.apply(compute_target, axis=1)

        # drop actual pit laps (leakage)
        laps = laps[laps["PitInTime"].isna()].copy()
        laps = laps.drop(columns=["PitLap"], errors="ignore")

        return laps

    def _add_race_metadata(self, laps: pd.DataFrame, race_name: str, year: int) -> pd.DataFrame:
        """Tag rows for train/test splitting."""
        laps = laps.copy()
        laps["race_name"] = race_name
        laps["season"] = year
        laps["race_id"] = f"{year}_{race_name.lower().replace(' ', '_')}"
        return laps
