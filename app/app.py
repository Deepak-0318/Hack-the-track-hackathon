import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from strategy_engine import (
    predict_next_lap,
    simulate_pit_effect,
    recommend_pit,
    predict_finish_position,
)

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="TRD Live Strategist", layout="wide")

# --- Custom styling for visual polish ---------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #071428 0%, #0b2a44 100%);
        color: #e6eef8;
    }
    .block-container {
        padding: 1.5rem 2rem;
    }
    .stHeader, header {visibility: hidden}
    footer {visibility: hidden}

    .card {
        background: rgba(255,255,255,0.03);
        padding: 1rem;
        border-radius: 12px;
        box-shadow: 0 6px 24px rgba(2,6,23,0.6);
        border: 1px solid rgba(255,255,255,0.04);
        margin-bottom: 12px;
    }

    .stMetric > div {
        background: transparent;
    }

    table {
        background: rgba(255,255,255,0.02);
        color: #e6eef8;
        border-radius: 8px;
        overflow: hidden;
    }

    .stButton>button {
        background: linear-gradient(90deg,#ff7a18,#af002d);
        color: #fff;
        border: none;
        box-shadow: 0 4px 12px rgba(175,0,45,0.2);
    }

    h1 {color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif}
    h2, h3 {color: #dbeafe}
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_processed" / "road-america"

# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
static_logo = Path(__file__).resolve().parent / "static" / "logo.svg"
if static_logo.exists():
    st.image(static_logo, width=160)
else:
    st.title("🏁 TRD Live Strategist")

st.markdown("**AI-Powered Decision Support for Toyota GR Cup Racing**")
st.markdown("---")

# ------------------------------------------------------------------
# Session selector
# ------------------------------------------------------------------
session_label = st.selectbox("Select Race Session", ["Race 1", "Race 2"])

file_map = {"Race 1": "race1_race_state.csv", "Race 2": "race2_race_state.csv"}
session_file = DATA_DIR / file_map[session_label]

if not session_file.exists():
    st.warning(f"{session_label} unavailable — falling back to Race 1.")
    session_label = "Race 1"
    session_file = DATA_DIR / file_map["Race 1"]

race_df = pd.read_csv(session_file)
race_df["lap_number"] = race_df["lap_number"].astype(int)

# Clean bad values
race_df_clean = race_df[
    (race_df["lap_number"] > 0)
    & (race_df["lap_time_s"] > 40)
    & (race_df["lap_time_s"] < 400)
]

real_lap_cap = int(race_df_clean["lap_number"].max())

st.caption(f"📂 Loaded dataset → Road America | {session_label}")
st.markdown("---")

# ------------------------------------------------------------------
# Car selection
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
cars = sorted(race_df["car_id"].unique())
selected_car = st.selectbox("Select Car", cars)

car_laps = race_df[race_df["car_id"] == selected_car].copy()
car_laps = car_laps[
    (car_laps["lap_number"] > 0)
    & (car_laps["lap_number"] <= real_lap_cap)
    & (car_laps["lap_time_s"] > 40)
    & (car_laps["lap_time_s"] < 400)
].sort_values("lap_number")

valid_laps = sorted(car_laps["lap_number"].unique())
lap = st.select_slider("Select Lap", options=valid_laps)

row = car_laps[car_laps["lap_number"] == lap].iloc[0]
max_laps = real_lap_cap

# 🔧 FIX — Ensure these exist before they're used anywhere
median_pace = float(row["median_pace"])
gap_to_front = float(row["gap_to_front_s"])
gap_to_leader = float(row["gap_to_leader_s"])
pit_flag = bool(row["pit_like"])

st.markdown('</div>', unsafe_allow_html=True)
# ------------------------------------------------------------------
# Current race context
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📍 Current Race Context")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Position", int(row["position"]))
c2.metric("Gap → Leader (s)", round(row["gap_to_leader_s"], 1))
c3.metric("Gap → Front (s)", round(row["gap_to_front_s"], 1))
c4.metric("Lap", int(row["lap_number"]))

# Prediction
predicted_text = "N/A"
try:
    pred_pos = predict_finish_position(
        lap_number=int(row["lap_number"]),
        median_pace=float(row["median_pace"]),
        gap_front=float(row["gap_to_front_s"]),
        gap_leader=float(row["gap_to_leader_s"]),
    )
    predicted_text = f"P{pred_pos}"
except:
    pass

c5.metric("Predicted Finish", predicted_text)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Strategy recommendation
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔥 Smart AI Strategy Recommendation")

pit_loss = st.slider("Assumed Pit Time Loss (seconds)", 20, 60, 30, step=2)

future_laps = car_laps[car_laps["lap_number"] > lap]
best_row = None

if not future_laps.empty:
    sims = []
    for _, r in future_laps.iterrows():
        sim = simulate_pit_effect(
            lap_number=int(r["lap_number"]),
            median_pace=float(r["median_pace"]),
            gap_to_front=float(r["gap_to_front_s"]),
            pit_loss_seconds=pit_loss,
            max_laps=max_laps,
        )
        sims.append(sim)

    sim_df_all = pd.DataFrame(sims)
    best_row = sim_df_all.loc[sim_df_all["pit_penalty_effect"].idxmin()]

if best_row is not None:
    opt_lap = int(best_row["lap_number"])
    effect = float(best_row["pit_penalty_effect"])
    if effect < 0:
        st.success(f"BEST PIT LAP → **Lap {opt_lap}** (gain: {abs(effect):.2f}s)")
    else:
        st.warning(f"Best compromise → **Lap {opt_lap}** (+{effect:.2f}s)")
else:
    st.info("No sufficient future data.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")


# ------------------------------------------------------------------
# Lap-based strategy details (current lap) – Strategy A vs B
# ------------------------------------------------------------------
sim = simulate_pit_effect(
    lap_number=lap,
    median_pace=median_pace,
    gap_to_front=gap_to_front,
    pit_loss_seconds=pit_loss,
    max_laps=max_laps,
)
rec = recommend_pit(sim)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Lap-Based Strategy Details")

base_time = sim["no_pit_next_lap_time"]
strategy_rows = [
    {
        "Strategy": "A – Stay out",
        "Next lap time (s)": base_time,
        "Δ vs stay out (s)": 0.0,
    },
    {
        "Strategy": "B – Pit this lap",
        "Next lap time (s)": sim["pit_next_lap_time"],
        "Δ vs stay out (s)": sim["pit_penalty_effect"],
    },
]
sim_df = pd.DataFrame(strategy_rows)
st.table(sim_df)

st.markdown(f"📌 Recommendation → **{rec}**")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Driver performance snapshot
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Driver Performance Snapshot")

best_lap = float(car_laps["lap_time_s"].min())
avg_lap = float(car_laps["lap_time_s"].mean())
consistency = float(car_laps["lap_time_s"].std() or 0.0)
pit_count = int(car_laps["pit_like"].sum())

if consistency < 3:
    rating_label = "🔥 Very consistent"
elif consistency < 7:
    rating_label = "⚡ Medium variability"
else:
    rating_label = "🧊 High variability"

colA, colB, colC, colD = st.columns(4)
colA.metric("Best Lap (s)", f"{best_lap:.2f}")
colB.metric("Average Lap (s)", f"{avg_lap:.2f}")
colC.metric("Consistency Index (σ)", f"{consistency:.2f}")
colD.metric("Pit Count", pit_count)

st.caption(f"Pace profile: {rating_label}")
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# 🎯 Driver Coaching Insights (with text summary)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🎯 Driver Coaching Insights")

# Use only laps completed up to the currently selected lap
completed = car_laps[car_laps["lap_number"] <= lap].copy().sort_values("lap_number")

MIN_LAPS_FOR_COACHING = 4

if len(completed) < MIN_LAPS_FOR_COACHING:
    st.info("Not enough completed laps to generate coaching insights yet.")
else:
    # Core statistics
    completed["rolling_3"] = completed["lap_time_s"].rolling(window=3, min_periods=1).mean()
    median_time = completed["lap_time_s"].median()
    best_idx = completed["lap_time_s"].idxmin()
    best_lap_no = int(completed.loc[best_idx, "lap_number"])
    best_time = float(completed.loc[best_idx, "lap_time_s"])

    # Delta vs median
    completed["delta_vs_median"] = completed["lap_time_s"] - median_time
    SLOW_THRESHOLD = 3.0  # seconds slower than median
    completed["is_slow"] = completed["delta_vs_median"] > SLOW_THRESHOLD

    # High-level coaching summary metrics
    slow_count = int(completed["is_slow"].sum())
    last_delta = float(completed.iloc[-1]["delta_vs_median"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Laps analyzed", len(completed))
    c2.metric("Best lap so far", f"{best_time:.2f}s (Lap {best_lap_no})")
    c3.metric("Slow laps (>+3s vs median)", slow_count)

    if last_delta > SLOW_THRESHOLD:
        st.warning(
            f"Last completed lap was **{last_delta:.2f}s** slower than your median – "
            f"possible mistake or traffic on lap {int(completed.iloc[-1]['lap_number'])}."
        )
    elif last_delta < -1.5:
        st.success(
            f"Last lap was **{abs(last_delta):.2f}s** faster than your median – "
            "great pace improvement, keep this trend!"
        )
    else:
        st.caption("Pace is close to your typical median – small refinements can still bring gains.")

    # --- Race phase stats for later summary & plot ---
    laps_arr = completed["lap_number"].values
    times_arr = completed["lap_time_s"].values
    indices = np.arange(len(completed))
    phase_edges = np.array_split(indices, 3)

    phase_labels = ["Early stint", "Middle stint", "Late stint"]
    phase_means = []
    for idxs in phase_edges:
        if len(idxs) == 0:
            phase_means.append(np.nan)
        else:
            phase_means.append(float(times_arr[idxs].mean()))

    # --- Visualizations in tabs ---
    tab1, tab2, tab3 = st.tabs(["Pace trace", "Lap deltas", "Race phases"])

    # Tab 1: Pace trace (lap times + rolling average + best lap)
    with tab1:
        fig_pace, ax_pace = plt.subplots(figsize=(8, 3))
        ax_pace.plot(
            completed["lap_number"],
            completed["lap_time_s"],
            marker="o",
            label="Lap time",
        )
        ax_pace.plot(
            completed["lap_number"],
            completed["rolling_3"],
            linestyle="--",
            marker=None,
            label="Rolling pace (3-lap)",
        )
        ax_pace.axhline(
            median_time,
            linestyle=":",
            label="Median pace",
        )
        ax_pace.scatter(
            [best_lap_no],
            [best_time],
            s=90,
            edgecolors="lime",
            facecolors="none",
            linewidths=2,
            label="Best lap",
        )
        ax_pace.set_xlabel("Lap")
        ax_pace.set_ylabel("Lap time (s)")
        ax_pace.set_title(f"Pace Trace — {selected_car} ({session_label})")
        ax_pace.grid(True)
        ax_pace.legend()
        st.pyplot(fig_pace)

    # Tab 2: Lap deltas vs median (bar chart)
    with tab2:
        fig_delta, ax_delta = plt.subplots(figsize=(8, 3))
        ax_delta.bar(
            completed["lap_number"],
            completed["delta_vs_median"],
        )
        ax_delta.axhline(0, linewidth=1)
        ax_delta.set_xlabel("Lap")
        ax_delta.set_ylabel("Δ time vs median (s)")
        ax_delta.set_title("Lap Time Delta vs Median Pace")
        ax_delta.grid(True, axis="y", linestyle="--", alpha=0.4)
        st.pyplot(fig_delta)

        st.caption(
            "Bars above zero are slower-than-median laps (potential mistakes / traffic); "
            "bars below zero are faster-than-median (strong laps)."
        )

    # Tab 3: Race phases (early / middle / late)
    with tab3:
        fig_phase, ax_phase = plt.subplots(figsize=(6, 3))
        ax_phase.bar(phase_labels, phase_means)
        ax_phase.set_ylabel("Average lap time (s)")
        ax_phase.set_title("Pace by Race Phase")
        ax_phase.grid(True, axis="y", linestyle="--", alpha=0.4)
        st.pyplot(fig_phase)

        st.caption(
            "Use this to see if you tend to be stronger early (tyre warm-up), "
            "mid-race, or late (tyre degradation / focus)."
        )

    # ---- Text coaching summary under the tabs ----
    st.markdown("---")
    st.markdown("**📝 Coaching summary**")

    summary_lines = []

    # Phase strengths / weaknesses
    try:
        # indices of min / max ignoring NaN
        strong_idx = int(np.nanargmin(phase_means))
        weak_idx = int(np.nanargmax(phase_means))
        strong_label = phase_labels[strong_idx]
        weak_label = phase_labels[weak_idx]
        strong_mean = phase_means[strong_idx]
        weak_mean = phase_means[weak_idx]
        phase_delta = weak_mean - strong_mean

        summary_lines.append(
            f"- Strongest phase: **{strong_label}** (~{strong_mean:.2f}s avg)."
        )
        summary_lines.append(
            f"- Weakest phase: **{weak_label}** (~{weak_mean:.2f}s). "
            f"Matching your best phase could unlock ~**{phase_delta:.2f}s** per lap there."
        )
    except (ValueError, TypeError):
        # nanargmin/argmax can fail if all NaN; just skip this part
        pass

    # Slow laps info
    if slow_count > 0:
        summary_lines.append(
            f"- You had **{slow_count} slow lap(s)** (>+{SLOW_THRESHOLD:.0f}s vs median). "
            "Review these for traffic, braking too deep, or missed apexes."
        )
    else:
        summary_lines.append(
            "- No major slow laps detected so far — consistency looks solid."
        )

    # Last lap commentary
    if last_delta > SLOW_THRESHOLD:
        summary_lines.append(
            "- Last lap was significantly off your usual pace – consider a reset lap to cool tyres "
            "and rebuild rhythm."
        )
    elif last_delta < -1.5:
        summary_lines.append(
            "- Last lap was one of your best relative to median – whatever you changed there is working, "
            "try to repeat that reference."
        )
    else:
        summary_lines.append(
            "- Recent laps are close to your typical pace – push to trim **0.2–0.3s** by focusing on "
            "one corner or braking zone at a time."
        )

    for line in summary_lines:
        st.markdown(line)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Lap time trend + pit visualization (overall)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("⏱ Lap Time Trend for Selected Car")

lt_df = car_laps[["lap_number", "lap_time_s", "pit_like"]].copy()

fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(lt_df["lap_number"], lt_df["lap_time_s"], marker="o")
pit_points = lt_df[lt_df["pit_like"] == 1]
if not pit_points.empty:
    ax.scatter(
        pit_points["lap_number"],
        pit_points["lap_time_s"],
        s=80,
        edgecolors="red",
        facecolors="none",
        linewidths=2,
        label="Pit / Slow laps",
    )

ax.set_xlabel("Lap")
ax.set_ylabel("Lap Time (s)")
ax.set_title(f"Lap Time Trend — {selected_car} ({session_label})")
ax.grid(True)
if not pit_points.empty:
    ax.legend()

st.pyplot(fig)

with st.expander("Show detected pit / slow laps"):
    if pit_points.empty:
        st.write("No pit-like laps detected for this car.")
    else:
        st.table(
            pit_points[["lap_number", "lap_time_s"]]
            .rename(columns={"lap_number": "Lap", "lap_time_s": "Lap time (s)"})
        )

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Race field view (top 5 progression)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🏁 Race Field View")

show_overview = st.checkbox("Show Top 5 Progression", value=True)

if show_overview and not race_df_clean.empty:
    final_lap = real_lap_cap
    top5 = (
        race_df_clean[race_df_clean["lap_number"] == final_lap]
        .sort_values("position")
        .head(5)["car_id"]
        .tolist()
    )

    overview = race_df_clean[race_df_clean["car_id"].isin(top5)]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for car in top5:
        cd = overview[overview["car_id"] == car]
        ax2.plot(cd["lap_number"], cd["position"], marker="o", label=car)

    ax2.invert_yaxis()
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Position")
    ax2.set_title(f"Position vs Lap – Top 5 Cars ({session_label})")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)

st.markdown('</div>', unsafe_allow_html=True)
