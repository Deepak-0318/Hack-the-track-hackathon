import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from strategy_engine import (
    predict_next_lap,
    simulate_pit_effect,
    recommend_pit,
    predict_finish_position,   # AI finish-position model
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
# Paths (project + data directory)
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data_processed" / "road-america"

# ------------------------------------------------------------------
# Header with logo / title
# ------------------------------------------------------------------
static_logo = Path(__file__).resolve().parent / "static" / "logo.svg"
if static_logo.exists():
    st.image(static_logo, width=160)
else:
    st.title("🏁 TRD Live Strategist")

st.markdown("**AI-Powered Decision Support for Toyota GR Cup Racing**")
st.markdown("---")

# ------------------------------------------------------------------
# Session selector (Race 1 / Race 2)
# ------------------------------------------------------------------
session_label = st.selectbox("Select Race Session", ["Race 1", "Race 2"])

file_map = {
    "Race 1": "race1_race_state.csv",
    "Race 2": "race2_race_state.csv",
}

session_file = DATA_DIR / file_map[session_label]

if not session_file.exists():
    st.warning(f"{session_label} unavailable — falling back to Race 1.")
    session_label = "Race 1"
    session_file = DATA_DIR / file_map["Race 1"]

race_df = pd.read_csv(session_file)
race_df["lap_number"] = race_df["lap_number"].astype(int)

# Drop obviously broken rows for global stuff (e.g. field view)
race_df_clean = race_df[
    (race_df["lap_number"] > 0)
    & (race_df["lap_time_s"] > 40)
    & (race_df["lap_time_s"] < 400)
]

# Reasonable maximum lap (ignore 32768 etc.)
real_lap_cap = int(race_df_clean["lap_number"].max())

st.caption(f"📂 Loaded dataset → Road America | {session_label}")
st.markdown("---")

# ------------------------------------------------------------------
# Car & Lap Selection
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
cars = sorted(race_df["car_id"].unique())
selected_car = st.selectbox("Select Car", cars)

# ---- Load laps for the selected car ----
car_laps = race_df[race_df["car_id"] == selected_car].copy()

# ---- Data Cleaning Layer (per-car) ----
car_laps = car_laps[
    (car_laps["lap_number"] > 0)
    & (car_laps["lap_number"] <= real_lap_cap)
    & (car_laps["lap_time_s"] > 40)
    & (car_laps["lap_time_s"] < 400)
]
car_laps = car_laps.sort_values("lap_number").reset_index(drop=True)

valid_laps = sorted(car_laps["lap_number"].unique().tolist())
if not valid_laps:
    st.error("No valid race laps found for this car.")
    st.markdown("</div>", unsafe_allow_html=True)
    st.stop()

default_lap = valid_laps[0]
lap = st.select_slider("Select Lap", options=valid_laps, value=default_lap)

row = car_laps[car_laps["lap_number"] == lap].iloc[0]
max_laps = real_lap_cap

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Current race context + predicted finish
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📍 Current Race Context")

c1, c2, c3, c4, c5 = st.columns(5)
cur_pos = int(row["position"])

c1.metric("Position", cur_pos)
c2.metric("Gap → Leader (s)", round(row["gap_to_leader_s"], 1))
c3.metric("Gap → Front (s)", round(row["gap_to_front_s"], 1))
c4.metric("Lap", int(row["lap_number"]))

# Predicted finishing position (AI model)
predicted_text = "N/A"
delta_text = None
try:
    pred_pos = predict_finish_position(
        lap_number=int(row["lap_number"]),
        median_pace=float(row["median_pace"]),
        gap_front=float(row["gap_to_front_s"]),
        gap_leader=float(row["gap_to_leader_s"]),
    )

    if pred_pos is not None:
        predicted_text = f"P{pred_pos}"
        delta = cur_pos - pred_pos  # positive = gain spots
        if delta > 0:
            delta_text = f"+{delta} places (projected gain)"
        elif delta < 0:
            delta_text = f"{abs(delta)} places (projected loss)"
        else:
            delta_text = "Flat vs current"
except Exception:
    # If model fails for any reason, keep N/A
    pass

c5.metric("Predicted Finish", predicted_text, delta=delta_text)

median_pace = float(row["median_pace"])
gap_to_front = float(row["gap_to_front_s"])
pit_flag = bool(row["pit_like"])

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# 🔥 Smart AI Strategy Recommendation (global best pit lap)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🔥 Smart AI Strategy Recommendation")

pit_loss = st.slider("Assumed Pit Time Loss (seconds)", 20, 60, 30, step=2)

# Compute best pit lap across *future* laps for this car
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
        st.success(
            f"BEST PIT LAP → **Lap {opt_lap}** (projected gain: **{abs(effect):.2f}s** vs staying out)."
        )
    else:
        st.warning(
            f"Best compromise pit lap: **Lap {opt_lap}** (still adds ~**{effect:.2f}s** vs staying out)."
        )
else:
    st.info("Not enough future laps available to compute an optimal pit window.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Lap-based strategy details (current lap) + Strategy A vs B
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

# Strategy A vs B compare:
#   Strategy A  -> Stay out
#   Strategy B  -> Pit this lap
strategy_compare_df = pd.DataFrame(
    [
        {
            "Strategy": "A – Stay out",
            "Next lap time (s)": sim["no_pit_next_lap_time"],
            "Δ vs stay out (s)": 0.0,
        },
        {
            "Strategy": "B – Pit this lap",
            "Next lap time (s)": sim["pit_next_lap_time"],
            "Δ vs stay out (s)": sim["pit_penalty_effect"],
        },
    ]
)

st.table(strategy_compare_df)

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
# Lap time trend + pit visualization
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
