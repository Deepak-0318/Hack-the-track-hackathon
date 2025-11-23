import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from strategy_engine import (
    predict_next_lap,
    simulate_pit_effect,
    recommend_pit,
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

        /* Card-like panels */
        .card {
            background: rgba(255,255,255,0.03);
            padding: 1rem;
            border-radius: 12px;
            box-shadow: 0 6px 24px rgba(2,6,23,0.6);
            border: 1px solid rgba(255,255,255,0.04);
            margin-bottom: 12px;
        }

        /* Tweak Streamlit metrics so they look like cards */
        .stMetric > div {
            background: transparent;
        }

        /* Tables and dataframes */
        table {
            background: rgba(255,255,255,0.02);
            color: #e6eef8;
            border-radius: 8px;
            overflow: hidden;
        }

        /* Buttons */
        .stButton>button {
            background: linear-gradient(90deg,#ff7a18,#af002d);
            color: #fff;
            border: none;
            box-shadow: 0 4px 12px rgba(175,0,45,0.2);
        }

        /* Headings */
        h1 {color: #ffffff; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif}
        h2, h3 {color: #dbeafe}

        </style>
        """,
        unsafe_allow_html=True,
)

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # project root
data_path = BASE_DIR / "data_processed" / "road-america" / "race1_race_state.csv"

race_df = pd.read_csv(data_path)
race_df["lap_number"] = race_df["lap_number"].astype(int)

# ------------------------------------------------------------------
# Header with logo
# ------------------------------------------------------------------
static_logo = Path(__file__).resolve().parent / "static" / "logo.svg"
if static_logo.exists():
    st.image(static_logo, width=160)
else:
    st.title("🏁 TRD Live Strategist")

st.markdown("**AI-Powered Decision Support for Toyota GR Cup Racing**")
st.markdown("---")

# ------------------------------------------------------------------
# Car & Lap Selection
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
cars = sorted(race_df["car_id"].unique())
selected_car = st.selectbox("Select Car", cars)

car_laps = race_df[race_df["car_id"] == selected_car].copy()

# Filter to realistic race laps
car_laps = car_laps[
    (car_laps["lap_number"] > 0)
    & (car_laps["lap_time_s"] > 40)     # ignore formation / super-short laps
    & (car_laps["lap_time_s"] < 1000)   # ignore crazy sentinel values
]

valid_laps = sorted(car_laps["lap_number"].unique().tolist())
if not valid_laps:
    st.error("No valid race laps found for this car.")
    st.stop()

default_lap = valid_laps[0]
lap = st.select_slider("Select Lap", options=valid_laps, value=default_lap)

row = car_laps[car_laps["lap_number"] == lap].iloc[0]
max_laps = int(race_df["lap_number"].max())

st.markdown('</div>', unsafe_allow_html=True)

# ------------------------------------------------------------------
# Current race context
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📍 Current Race Context")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Position", int(row["position"]))
c2.metric("Gap to Leader (s)", round(row["gap_to_leader_s"], 2))
c3.metric("Gap to Front (s)", round(row["gap_to_front_s"], 2))
c4.metric("Laps Completed", int(row["lap_number"]))

median_pace = float(row["median_pace"])
gap_to_front = float(row["gap_to_front_s"])
pit_flag = bool(row["pit_like"])

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Prediction + Strategy Recommendation
# ------------------------------------------------------------------
sim = simulate_pit_effect(
    lap_number=lap,
    median_pace=median_pace,
    gap_to_front=gap_to_front,
    pit_loss_seconds=30,  # default assumption
    max_laps=max_laps,
)
rec = recommend_pit(sim)

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📈 Next-Lap Pace & Pit Impact")

sim_df = pd.DataFrame([sim]).rename(
    columns={
        "lap_number": "Lap",
        "no_pit_next_lap_time": "No-pit next lap (s)",
        "pit_next_lap_time": "Pit next lap (s)",
        "pit_penalty_effect": "Pit penalty effect (s)",
    }
)
st.table(sim_df[["Lap", "No-pit next lap (s)", "Pit next lap (s)", "Pit penalty effect (s)"]])

st.subheader("🧠 Race Engineer Recommendation")
st.success(rec)

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

# ------------------------------------------------------------------
# Driver performance snapshot
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📌 Driver Performance Snapshot")

best_lap = float(car_laps["lap_time_s"].min())
avg_lap = float(car_laps["lap_time_s"].mean())
consistency = float(car_laps["lap_time_s"].std() or 0.0)  # std can be NaN
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
ax.set_title(f"Lap Times – {selected_car}")
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
# Pit Window Explorer (next 3 laps)
# ------------------------------------------------------------------
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧪 Pit Window Explorer (Next 3 Laps)")

pit_loss = st.slider("Assumed Pit Loss (seconds)", 20, 60, 30, step=2)

window_laps = [l for l in valid_laps if lap <= l <= lap + 3]
window_results = []
best_row = None

for future_lap in window_laps:
    future_row = car_laps[car_laps["lap_number"] == future_lap].iloc[0]
    sim_future = simulate_pit_effect(
        lap_number=int(future_row["lap_number"]),
        median_pace=float(future_row["median_pace"]),
        gap_to_front=float(future_row["gap_to_front_s"]),
        pit_loss_seconds=pit_loss,
        max_laps=max_laps,
    )
    sim_future["lap_number"] = int(sim_future["lap_number"])
    window_results.append(sim_future)

if window_results:
    window_df = pd.DataFrame(window_results)

    st.write("**Simulated pit effect per lap in this window:**")
    window_table = (
        window_df[
            ["lap_number", "no_pit_next_lap_time", "pit_next_lap_time", "pit_penalty_effect"]
        ]
        .rename(
            columns={
                "lap_number": "Lap",
                "no_pit_next_lap_time": "No-pit next lap (s)",
                "pit_next_lap_time": "Pit next lap (s)",
                "pit_penalty_effect": "Pit penalty effect (s)",
            }
        )
        .sort_values("Lap")
    )
    st.dataframe(window_table, hide_index=True)

    best_row = window_df.sort_values("pit_penalty_effect").iloc[0]
    msg = (
        f"Best pit lap in this window: **Lap {int(best_row['lap_number'])}** "
        f"(penalty effect: {best_row['pit_penalty_effect']:.2f} s)"
    )
    if best_row["pit_penalty_effect"] < 0:
        st.success("✅ " + msg)
    else:
        st.warning("⚠️ " + msg)
else:
    st.write("Not enough future laps available to explore pit window.")

st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown("---")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🗣 AI Race Commentary")

if best_row is not None:
    penalty = float(best_row["pit_penalty_effect"])
    best_pit_lap = int(best_row["lap_number"])
    gain_or_loss = "gain" if penalty < 0 else "lose"
    secs = abs(penalty)

    commentary = f"""
Car **{selected_car}** is currently running **P{int(row['position'])}**, with a gap of **{row['gap_to_leader_s']:.1f}s** to the leader and **{row['gap_to_front_s']:.1f}s** to the car ahead.

The driver’s **best lap** is **{best_lap:.2f}s**, with an average pace of **{avg_lap:.2f}s** and a consistency index (σ) of **{consistency:.2f}s**, indicating **{rating_label}**.

Using our strategy model and assuming a pit loss of **{pit_loss}s**, the optimal pit window in the next few laps appears at **lap {best_pit_lap}**, where the car is projected to **{gain_or_loss} ~{secs:.2f}s** relative to staying out.

This suggests that if conditions remain stable, planning a stop around lap **{best_pit_lap}** could improve overall race time for car **{selected_car}**.
"""
else:
    commentary = f"""
Car **{selected_car}** is currently running **P{int(row['position'])}** with a gap of **{row['gap_to_leader_s']:.1f}s** to the leader.
The model did not find enough future laps to evaluate a full pit window, but the current pace profile
(best lap **{best_lap:.2f}s**, average **{avg_lap:.2f}s**) indicates **{rating_label}**.
"""

st.write(commentary)
st.markdown('</div>', unsafe_allow_html=True)
st.markdown("---")

st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🏁 Race Overview (Top 5 Cars)")
show_overview = st.checkbox("Show race overview for top 5 cars")

if show_overview:
    final_lap = race_df["lap_number"].max()
    top5 = (
        race_df[race_df["lap_number"] == final_lap]
        .sort_values("position")
        .head(5)["car_id"]
        .tolist()
    )

    overview = race_df[race_df["car_id"].isin(top5)]

    fig2, ax2 = plt.subplots(figsize=(8, 4))
    for car in top5:
        cd = overview[overview["car_id"] == car]
        ax2.plot(cd["lap_number"], cd["position"], marker="o", label=car)

    ax2.invert_yaxis()
    ax2.set_xlabel("Lap")
    ax2.set_ylabel("Position")
    ax2.set_title("Position vs Lap – Top 5 Cars")
    ax2.grid(True)
    ax2.legend()
    st.pyplot(fig2)
st.markdown('</div>', unsafe_allow_html=True)
