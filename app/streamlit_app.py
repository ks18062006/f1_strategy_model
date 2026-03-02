"""

Run with:
    streamlit run app/streamlit_app.py
"""

import pickle
import sys
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import streamlit as st

# Allow imports from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_builder import RaceDataBuilder, DRY_RACES_2023
from src.evaluate import _style_ax, COLORS
from src.train_model import MODEL_OUTPUT_PATH, FE_OUTPUT_PATH

# Constants
COMPOUND_COLORS = {
    "SOFT": "#FF3333",
    "MEDIUM": "#FFC906",
    "HARD": "#EEEEEE",
    "INTERMEDIATE": "#39B54A",
    "WET": "#00AEEF",
}
COMPOUND_TEXT_COLOR = {
    "SOFT": "#fff", "MEDIUM": "#000", "HARD": "#000",
    "INTERMEDIATE": "#fff", "WET": "#fff",
}

# Typical pit stop time loss in seconds (pit lane + stationary)
PIT_STOP_LOSS = 22.0

# Page config
st.set_page_config(
    page_title="F1 Pit Strategy AI",
    page_icon="🏎️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;900&display=swap');
    
    *, body, .stApp { font-family: 'Inter', sans-serif; }
    body, .stApp { background-color: #0D0D0D; color: #F0F0F0; }
    
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #141414 0%, #0D0D0D 100%);
        border-right: 1px solid #222;
    }
    
    .tyre-pill {
        display: inline-block; padding: 4px 14px; border-radius: 20px;
        font-weight: 700; font-size: 13px; margin: 2px 4px; letter-spacing: 0.5px;
    }
    .tyre-SOFT { background: #FF3333; color: #fff; }
    .tyre-MEDIUM { background: #FFC906; color: #000; }
    .tyre-HARD { background: #EEEEEE; color: #000; }
    .tyre-INTERMEDIATE { background: #39B54A; color: #fff; }
    .tyre-WET { background: #00AEEF; color: #fff; }
    
    .driver-card {
        background: #1A1A1A; border: 1px solid #2A2A2A;
        border-radius: 12px; padding: 20px 24px; margin: 12px 0;
    }
    .driver-name { font-size: 28px; font-weight: 900; color: #fff; }
    .team-name {
        font-size: 14px; color: #777; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1px;
    }
    .stat-label {
        font-size: 11px; color: #666; text-transform: uppercase;
        letter-spacing: 1px; margin-bottom: 2px;
    }
    .stat-value { font-size: 22px; font-weight: 700; color: #fff; }
    .pit-window {
        background: linear-gradient(135deg, #E8002D22, #E8002D08);
        border: 1px solid #E8002D44; border-radius: 10px;
        padding: 12px 16px; text-align: center;
    }
    .pit-window-label {
        font-size: 11px; color: #E8002D; text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 700;
    }
    .pit-window-value { font-size: 26px; font-weight: 900; color: #E8002D; }
    .stint-bar {
        display: flex; align-items: center; gap: 0; width: 100%;
        height: 32px; border-radius: 6px; overflow: hidden; margin: 8px 0;
    }
    .stint-segment {
        height: 100%; display: flex; align-items: center;
        justify-content: center; font-size: 11px; font-weight: 700; min-width: 30px;
    }
    .sim-card {
        background: #141414; border: 1px solid #2A2A2A;
        border-radius: 12px; padding: 20px 24px; margin: 16px 0;
    }
    .sim-title {
        font-size: 18px; font-weight: 800; color: #E8002D;
        margin-bottom: 12px; letter-spacing: -0.3px;
    }
    .time-result {
        background: linear-gradient(135deg, #1A1A2E, #16213E);
        border: 1px solid #0F3460; border-radius: 10px;
        padding: 16px; text-align: center; margin: 8px 0;
    }
    .time-label {
        font-size: 11px; color: #53A8B6; text-transform: uppercase;
        letter-spacing: 1.5px; font-weight: 700;
    }
    .time-value { font-size: 30px; font-weight: 900; color: #fff; }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helpers


@st.cache_resource
def load_model_and_fe():
    with open(MODEL_OUTPUT_PATH, "rb") as f:
        model = pickle.load(f)
    with open(FE_OUTPUT_PATH, "rb") as f:
        fe = pickle.load(f)
    return model, fe


@st.cache_data
def load_race_data(race_name: str) -> pd.DataFrame:
    builder = RaceDataBuilder(cache_dir="data/raw")
    return builder.load_single_race(2023, race_name)


def get_stints(df_driver: pd.DataFrame) -> list:
    """Extract stint info: compound, start lap, end lap."""
    df = df_driver.sort_values("LapNumber")
    stints, current, start, prev = [], None, None, None
    for _, row in df.iterrows():
        comp = str(row.get("Compound", "UNKNOWN")).upper()
        lap = int(row["LapNumber"])
        if comp != current:
            if current is not None:
                stints.append({"compound": current, "start": start, "end": prev})
            current, start = comp, lap
        prev = lap
    if current is not None:
        stints.append({"compound": current, "start": start, "end": prev})
    return stints


def find_recommended_pit_lap(df: pd.DataFrame) -> Optional[int]:
    if df.empty or "pit_prob" not in df.columns:
        return None
    idx = df["pit_prob"].idxmax()
    return int(df.loc[idx, "LapNumber"])


def render_stint_bar_html(stints: list, total_laps: int) -> str:
    if not stints or total_laps == 0:
        return ""
    segments = []
    for s in stints:
        w = max(((s["end"] - s["start"] + 1) / total_laps) * 100, 5)
        c = COMPOUND_COLORS.get(s["compound"], "#888")
        tc = COMPOUND_TEXT_COLOR.get(s["compound"], "#fff")
        segments.append(
            f'<div class="stint-segment" style="width:{w:.1f}%;background:{c};color:{tc};">'
            f'{s["compound"][0]} L{s["start"]}–{s["end"]}</div>'
        )
    return f'<div class="stint-bar">{"".join(segments)}</div>'


def get_compound_pace(df_race: pd.DataFrame) -> dict:
    """Get median lap time per compound from actual race data."""
    pace = {}
    for comp in ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]:
        mask = df_race["Compound"].str.upper() == comp
        if mask.any():
            times = df_race.loc[mask, "LapTimeSeconds"].dropna()
            if not times.empty:
                pace[comp] = float(times.median())
    return pace


def format_race_time(total_seconds: float) -> str:
    """Format total seconds to H:MM:SS.sss"""
    hours = int(total_seconds // 3600)
    mins = int((total_seconds % 3600) // 60)
    secs = total_seconds % 60
    if hours > 0:
        return f"{hours}:{mins:02d}:{secs:05.2f}"
    return f"{mins}:{secs:05.2f}"


def simulate_strategy(total_laps: int, num_stops: int, compounds: list, compound_pace: dict) -> dict:
    """
    Simulate a race strategy given tyre choices.
    
    Returns dict with:
        - stint_plan: list of {compound, start, end, laps}
        - total_time: estimated total race time
        - stint_times: time per stint
    """
    if not compounds or len(compounds) != num_stops + 1:
        return None
    
    # Distribute laps evenly across stints, then adjust
    base_laps = total_laps // (num_stops + 1)
    remainder = total_laps % (num_stops + 1)
    
    stint_plan = []
    current_lap = 1
    
    for i, comp in enumerate(compounds):
        stint_laps = base_laps + (1 if i < remainder else 0)
        stint_plan.append({
            "compound": comp,
            "start": current_lap,
            "end": current_lap + stint_laps - 1,
            "laps": stint_laps,
        })
        current_lap += stint_laps
    
    # Calculate times
    stint_times = []
    for stint in stint_plan:
        comp = stint["compound"]
        base_pace = compound_pace.get(comp, 90.0)
        
        # Simple degradation model: lap time increases with tyre age
        # Soft degrades faster, Hard degrades slower
        deg_rate = {"SOFT": 0.08, "MEDIUM": 0.05, "HARD": 0.03,
                    "INTERMEDIATE": 0.06, "WET": 0.06}.get(comp, 0.05)
        
        stint_time = 0
        for lap_in_stint in range(stint["laps"]):
            lap_time = base_pace + (lap_in_stint * deg_rate)
            stint_time += lap_time
        stint_times.append(stint_time)
    
    total_time = sum(stint_times) + (num_stops * PIT_STOP_LOSS)
    
    return {
        "stint_plan": stint_plan,
        "total_time": total_time,
        "stint_times": stint_times,
        "pit_loss": num_stops * PIT_STOP_LOSS,
    }


# Sidebar

if not MODEL_OUTPUT_PATH.exists():
    st.error("No trained model found. Run `python -m src.train_model` first.")
    st.stop()

model, fe = load_model_and_fe()

with st.sidebar:
    st.markdown("### 🏎️ F1 Strategy AI")
    st.markdown("---")
    
    selected_race = st.selectbox("Race", DRY_RACES_2023, index=8)
    
    with st.spinner("Loading race..."):
        try:
            df_race = load_race_data(selected_race)
        except Exception as e:
            st.error(f"Failed: {e}")
            st.stop()
    
    total_laps = int(df_race["LapNumber"].max())
    st.markdown(f"**{total_laps} laps**")
    st.markdown("---")
    
    # Get drivers sorted by starting position
    driver_first = df_race.sort_values("LapNumber").groupby("Driver").first().reset_index()
    driver_first = driver_first.sort_values("Position")
    drivers = driver_first["Driver"].tolist()
    
    st.markdown("**Select Driver**")
    
    # Driver buttons in sidebar
    selected_driver = None
    for _, row in driver_first.iterrows():
        drv = row["Driver"]
        team = row.get("Team", "")
        pos = int(row["Position"]) if pd.notna(row["Position"]) else "?"
        if st.button(f"P{pos}  {drv}  —  {team}", key=f"drv_{drv}", use_container_width=True):
            st.session_state["selected_driver"] = drv
    
    # Default to first driver
    if "selected_driver" not in st.session_state:
        st.session_state["selected_driver"] = drivers[0] if drivers else None

selected_driver = st.session_state.get("selected_driver")

# Main content

st.markdown(
    f'<h1 style="font-weight:900; letter-spacing:-1px;">Pit Strategy — {selected_race}</h1>',
    unsafe_allow_html=True
)

if not selected_driver:
    st.info("Select a driver from the sidebar.")
    st.stop()

df_d = df_race[df_race["Driver"] == selected_driver].copy()
if df_d.empty:
    st.warning(f"No data for {selected_driver}.")
    st.stop()

# Driver metadata
team = df_d["Team"].iloc[0] if "Team" in df_d.columns else "—"
start_pos = int(df_d["Position"].iloc[0]) if pd.notna(df_d["Position"].iloc[0]) else "—"

# Predict pit probabilities
try:
    X_d, _ = fe.transform(df_d)
    df_d = df_d.iloc[:len(X_d)].copy()
    df_d["pit_prob"] = model.predict_proba(X_d)[:, 1]
    df_d = df_d.sort_values("LapNumber")
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

stints = get_stints(df_d)
rec_lap = find_recommended_pit_lap(df_d)
actual_pits = df_race[
    (df_race["Driver"] == selected_driver) & df_race["PitInTime"].notna()
]["LapNumber"].tolist()

# SECTION 1: Driver Card + Pit Probability Curve

st.markdown(f'<div class="driver-card">', unsafe_allow_html=True)

h_col, chart_col = st.columns([1, 2])

with h_col:
    st.markdown(
        f'<div class="driver-name">{selected_driver}</div>'
        f'<div class="team-name">{team}</div>',
        unsafe_allow_html=True
    )
    st.markdown(
        f'<div style="display:flex; gap:20px; margin-top:16px; flex-wrap:wrap;">'
        f'<div><div class="stat-label">Grid</div><div class="stat-value">P{start_pos}</div></div>'
        f'<div><div class="stat-label">Peak Prob</div><div class="stat-value">{df_d["pit_prob"].max():.0%}</div></div>'
        f'<div><div class="stat-label">Pit Stops</div><div class="stat-value">{len(actual_pits)}</div></div>'
        f'</div>',
        unsafe_allow_html=True
    )
    if rec_lap:
        st.markdown(
            f'<div class="pit-window" style="margin-top:16px;">'
            f'<div class="pit-window-label">Model Recommended Pit</div>'
            f'<div class="pit-window-value">Lap {rec_lap}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    # Stint bar
    st.markdown(f'<div class="stat-label" style="margin-top:16px;">Actual Tyre Strategy</div>', unsafe_allow_html=True)
    st.markdown(render_stint_bar_html(stints, total_laps), unsafe_allow_html=True)

with chart_col:
    fig, ax = plt.subplots(figsize=(11, 3.5), facecolor="#1A1A1A")
    ax.set_facecolor("#1A1A1A")
    ax.plot(df_d["LapNumber"], df_d["pit_prob"], color="#E8002D", lw=2.2)
    ax.fill_between(df_d["LapNumber"], df_d["pit_prob"], alpha=0.12, color="#E8002D")
    
    for lap in actual_pits:
        ax.axvline(lap, color="#FFD700", lw=1.5, ls="--", alpha=0.8)
    if rec_lap:
        ax.axvline(rec_lap, color="#E8002D", lw=1.5, ls=":", alpha=0.5)
    
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1))
    ax.set_xlim(0, total_laps + 1)
    ax.set_ylim(0, max(df_d["pit_prob"].max() * 1.3, 0.1))
    ax.tick_params(colors="#666", labelsize=8)
    ax.set_xlabel("Lap", color="#555", fontsize=9)
    ax.set_ylabel("Pit Prob", color="#555", fontsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor("#2A2A2A")
    ax.grid(True, color="#222", linewidth=0.4, alpha=0.5)
    
    # Legend
    ax.plot([], [], color="#FFD700", lw=1.5, ls="--", label="Actual pit stop")
    ax.plot([], [], color="#E8002D", lw=1.5, ls=":", label="Recommended pit")
    ax.legend(facecolor="#1A1A1A", labelcolor="#aaa", fontsize=8, loc="upper right")
    
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)

st.markdown('</div>', unsafe_allow_html=True)


# SECTION 2: Strategy Simulator


st.markdown("---")
st.markdown(
    '<div class="sim-title" style="font-size:22px;">⚙️ Strategy Simulator</div>',
    unsafe_allow_html=True
)
st.caption("Choose your tyres and number of stops — the model estimates total race time with degradation.")

compound_pace = get_compound_pace(df_race)
available_compounds = list(compound_pace.keys())

if not available_compounds:
    st.warning("No compound pace data available for this race.")
    st.stop()

# Show compound pace reference
st.markdown('<div class="stat-label">Median lap time by compound (from race data)</div>', unsafe_allow_html=True)
pace_cols = st.columns(len(available_compounds))
for i, comp in enumerate(available_compounds):
    with pace_cols[i]:
        pace = compound_pace[comp]
        mins = int(pace // 60)
        secs = pace % 60
        st.markdown(
            f'<div style="text-align:center;">'
            f'<span class="tyre-pill tyre-{comp}">{comp}</span><br>'
            f'<span style="color:#aaa; font-size:18px; font-weight:700;">{mins}:{secs:05.2f}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("")

# Strategy inputs
sim_col1, sim_col2 = st.columns([1, 3])

with sim_col1:
    num_stops = st.radio("Number of pit stops", [1, 2, 3], index=0, key="num_stops")

with sim_col2:
    st.markdown(f"**Choose compound for each stint ({num_stops + 1} stints)**")
    stint_cols = st.columns(num_stops + 1)
    chosen_compounds = []
    for i in range(num_stops + 1):
        with stint_cols[i]:
            comp = st.selectbox(
                f"Stint {i + 1}",
                available_compounds,
                index=min(i, len(available_compounds) - 1),
                key=f"sim_compound_{i}"
            )
            chosen_compounds.append(comp)

# Run simulation
result = simulate_strategy(total_laps, num_stops, chosen_compounds, compound_pace)

if result:
    st.markdown("---")
    
    # Results row
    r1, r2, r3 = st.columns(3)
    
    with r1:
        st.markdown(
            f'<div class="time-result">'
            f'<div class="time-label">Estimated Race Time</div>'
            f'<div class="time-value">{format_race_time(result["total_time"])}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with r2:
        st.markdown(
            f'<div class="time-result">'
            f'<div class="time-label">Pit Stop Time Loss</div>'
            f'<div class="time-value">{result["pit_loss"]:.0f}s ({num_stops} stop{"s" if num_stops > 1 else ""})</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with r3:
        driving_time = result["total_time"] - result["pit_loss"]
        st.markdown(
            f'<div class="time-result">'
            f'<div class="time-label">Driving Time</div>'
            f'<div class="time-value">{format_race_time(driving_time)}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    
    # Simulated stint plan
    st.markdown('<div class="stat-label" style="margin-top:16px;">Simulated Strategy</div>', unsafe_allow_html=True)
    sim_stint_html = render_stint_bar_html(result["stint_plan"], total_laps)
    st.markdown(sim_stint_html, unsafe_allow_html=True)
    
    # Stint breakdown table
    stint_data = []
    for i, stint in enumerate(result["stint_plan"]):
        stint_data.append({
            "Stint": i + 1,
            "Compound": stint["compound"],
            "Laps": f'{stint["start"]}–{stint["end"]} ({stint["laps"]} laps)',
            "Stint Time": format_race_time(result["stint_times"][i]),
        })
    
    st.dataframe(
        pd.DataFrame(stint_data).set_index("Stint"),
        use_container_width=True
    )
    
    # Comparison chart: simulated vs actual
    st.markdown('<div class="stat-label" style="margin-top:16px;">Simulated vs Actual Strategy</div>', unsafe_allow_html=True)
    
    fig, ax = plt.subplots(figsize=(14, 2.5), facecolor="#141414")
    ax.set_facecolor("#141414")
    
    # Actual (top row)
    for stint in stints:
        c = COMPOUND_COLORS.get(stint["compound"], "#888")
        ax.barh(1, stint["end"] - stint["start"] + 1, left=stint["start"],
                height=0.35, color=c, edgecolor="#0D0D0D", linewidth=0.8)
    
    # Simulated (bottom row)
    for stint in result["stint_plan"]:
        c = COMPOUND_COLORS.get(stint["compound"], "#888")
        ax.barh(0, stint["laps"], left=stint["start"],
                height=0.35, color=c, edgecolor="#0D0D0D", linewidth=0.8)
    
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Simulated", "Actual"], fontsize=10, color="#aaa")
    ax.set_xlim(0, total_laps + 2)
    ax.set_xlabel("Lap", color="#555", fontsize=9)
    ax.tick_params(colors="#666", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#222")
    ax.grid(True, axis="x", color="#222", linewidth=0.3, alpha=0.5)
    plt.tight_layout(pad=0.5)
    st.pyplot(fig)
    plt.close(fig)
