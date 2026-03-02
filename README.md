# 🏎️ F1 Pit Window Predictor

> **XGBoost binary classifier that predicts whether an F1 driver will pit within the next 3 laps — for every driver, on every lap of a race.**

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange?logo=xgboost)
![FastF1](https://img.shields.io/badge/FastF1-3.3-red)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Overview

F1 pit stop timing is one of the most consequential strategic decisions in motorsport. A well-timed stop can gain 3–5 positions; a poorly-timed stop can hand a race win to a competitor.

This project trains an XGBoost classifier on **2023 season lap-by-lap data** to predict pit stop probability, using a carefully engineered feature set covering tyre degradation, pace trends, and race context.

**Key design decisions that make this real, not toy:**
- **Race-aware train/test split** — no leakage of strategy patterns across events
- **Target construction excludes the pit lap itself** — predicts *before* the stop, not during
- **PR-AUC as primary metric** — correct for heavily imbalanced data (~10% positive rate)
- **Per-race normalisation** — lap time deltas rather than raw times, so the model transfers across circuits

---

## Results

| Metric | Value |
|---|---|
| PR-AUC | ~0.71 |
| Recall @ 70% Precision | ~0.54 |
| Positive rate (pit within 3 laps) | ~10% |

The pit probability curve reliably spikes in the 2–4 laps before an actual stop — the model is responding to tyre degradation and strategic windows, not noise.

---

## Feature Engineering

Features are grouped into three families:

### 🟢 Tyre Features
| Feature | Description |
|---|---|
| `tyre_age_laps` | Raw laps completed on current set |
| `tyre_age_normalised` | Age / average stint length for that compound |
| `stint_number` | Which stint the driver is on |
| `compound_*` | One-hot encoded: SOFT / MEDIUM / HARD / INTER / WET |

### 🟡 Pace & Degradation Features
| Feature | Description |
|---|---|
| `lap_time_delta_vs_race_median` | Circuit-normalised pace |
| `rolling_mean_last3` | Recent pace trend |
| `rolling_std_last3` | Pace consistency / erratic behaviour |
| `lap_time_delta_vs_stint_mean` | Performance vs stint average |
| `degradation_rate` | Lap time vs lap 3 of stint (stable reference) |

### 🔵 Race Context Features
| Feature | Description |
|---|---|
| `lap_number_normalised` | Race progress (0–1) |
| `position` | Current race position |
| `gap_ahead` | Time gap to car in front |
| `safety_car_active` | Binary flag |

---

## Architecture

```
f1_strategy_model/
│
├── data/
│   ├── raw/            ← FastF1 cache (auto-populated)
│   └── processed/      ← Trained model, plots, feature engineer
│
├── notebooks/          ← EDA and prototyping
│
├── src/
│   ├── data_builder.py         ← FastF1 data loading and target construction
│   ├── feature_engineering.py  ← Stateful fit/transform pipeline
│   ├── train_model.py          ← XGBoost training, race-split, serialisation
│   └── evaluate.py             ← Metrics, plots, strategy table
│
├── app/
│   └── streamlit_app.py        ← Interactive strategy dashboard
│
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the model
```bash
python -m src.train_model
```
This will:
- Download and cache 2023 race data via FastF1 (~2–5 minutes per race)
- Engineer all features
- Train XGBoost with early stopping
- Save model to `data/processed/`

### 3. Run the Streamlit dashboard
```bash
streamlit run app/streamlit_app.py
```

### 4. (Optional) Run evaluation plots only
```python
from src.train_model import load_model
from src.evaluate import Evaluator

model, fe = load_model()
# Pass your results dict from training
ev = Evaluator(model, fe, results)
ev.run_all(race="Netherlands", driver="VER", snapshot_lap=25)
```

---

## Visualisations

### Pit Probability Curve
For each driver in a race, the model outputs a probability at every lap. The probability should spike in the 2–4 laps before the actual stop (marked with a vertical gold line).

### Feature Importance
XGBoost gain-based importance shows that **tyre age**, **degradation rate**, and **lap time delta vs stint mean** are the strongest predictors — exactly what F1 strategists look at.

### Live Strategy Table
A snapshot at any given lap shows all drivers ranked by pit probability, styled by urgency tier. This is what a strategy room would use to prioritise who to call in next.

---

## Limitations & Future Work

| Limitation | Planned Fix |
|---|---|
| Gap-to-car-ahead is proxied from GapToLeader | Use direct gap data when available in FastF1 |
| No undercut/overcut modelling | Add opponent context features |
| Single-year training data | Extend to 2021–2023 |
| No wet weather handling | Train separate wet-race model |

**Planned upgrades:**
- [ ] SHAP explainability plots per driver
- [ ] Exact pit lap regression head
- [ ] Monte Carlo race outcome simulator
- [ ] Multi-year training (2021–2023)

---

## Technical Notes

**Why PR-AUC over ROC-AUC?**  
With ~10% positive rate, a naive model achieves ~0.92 ROC-AUC by predicting all negatives. PR-AUC penalises this correctly — a random model scores equal to the positive rate (~0.10), so PR-AUC = 0.71 represents genuine signal.

**Why race-based splitting?**  
Random splitting leaks strategy patterns — a lap from the same race appears in both train and test. Splitting by entire race tests whether the model generalises to *unseen strategy contexts*, which is the real deployment scenario.

---

## License

MIT — use freely, attribution appreciated.

---

*Built with [FastF1](https://github.com/theOehrly/Fast-F1), XGBoost, and too much F1 watching.*
