import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4

# ------------------------------ PAGE CONFIG ------------------------------

st.set_page_config(
    page_title="RaceAI — Live Strategy Dashboard",
    page_icon="🏎️",
    layout="wide"
)

# ------------------------------ VISUAL THEME ------------------------------

plt.style.use("dark_background")
plt.rcParams.update({
    "axes.edgecolor": "#B2B2B2",
    "axes.labelcolor": "white",
    "xtick.color": "white",
    "ytick.color": "white",
    "figure.facecolor": "#000000",
    "axes.facecolor": "#000000",
    "lines.linewidth": 3,
    "lines.marker": "o",
    "lines.markersize": 8,
})

st.markdown("""
<style>
div.block-container { padding-top: 2rem; }

[data-testid="stMetricValue"] {
    color: #FF3B3B !important;
    font-weight: 900;
}

section.main > div {
    border-radius: 14px;
    background: #111;
    padding: 12px;
    margin-bottom: 18px;
    box-shadow: 0px 0px 26px rgba(255,0,0,0.10);
}

.sidebar .sidebar-content {
    background-color: #080808;
    padding: 10px;
}

.sidebar-title {
    font-size: 22px;
    font-weight: 700;
    margin-top: 12px;
}

hr {
    border: 1px solid rgba(255,255,255,.1);
}
</style>
""", unsafe_allow_html=True)


# ------------------------------ DATA LOADER ------------------------------

SEARCH_FOLDERS = [
    ".", "data_processed", "data_processed/road-america",
    "data_processed/road-america/Road America"
]

@st.cache_data
def detect_csv_files():
    files = []
    for folder in SEARCH_FOLDERS:
        if os.path.exists(folder):
            files.extend([os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(".csv")])
    return sorted(files)

def clean_dataframe(df):
    df.columns = [c.strip().lower() for c in df.columns]
    
    rename_map = {
        "lap#": "lap",
        "lap_number": "lap",
        "lap_time_s": "lap_time",
        "gap_to_leader_s": "gap_leader",
        "gap_to_front_s": "gap_front",
        "pit_like": "pits"
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)
    return df

@st.cache_data
def load_data(filepath):
    return clean_dataframe(pd.read_csv(filepath))


# ------------------------------ SIDEBAR NAVIGATION ------------------------------

st.sidebar.markdown("<p class='sidebar-title'>RaceAI Navigation</p>", unsafe_allow_html=True)
menu = st.sidebar.radio("Select View", ["🏁 Dashboard", "📈 Driver Comparison", "📤 Export Report"])

csv_files = detect_csv_files()
if not csv_files:
    st.sidebar.error("No CSV found. Add files to /data_processed/")
    st.stop()

selected_file = st.sidebar.selectbox("Select dataset", csv_files)
df = load_data(selected_file)
st.sidebar.success(f"Loaded: {selected_file}")


# ------------------------------ CAR SELECTION ------------------------------

cars = sorted(df["car_id"].unique())
selected_car = st.sidebar.selectbox("Select Driver / Car", cars)

car_df = df[df["car_id"] == selected_car].copy()
car_df = car_df[(car_df["lap_time"] > 10) & (car_df["lap_time"] < 400)]

lap_column = next((c for c in car_df.columns if "lap" in c.lower()), None)
if lap_column is None:
    st.error("Dataset missing lap column.")
    st.stop()

unique_laps = sorted(car_df[lap_column].unique())
selected_lap = st.sidebar.slider("Lap", int(min(unique_laps)), int(max(unique_laps)), int(min(unique_laps)))
lap_row = car_df[car_df[lap_column] == selected_lap].iloc[0]


# ------------------------------ LIVE SIMULATION TOGGLE ------------------------------

def run_live_simulation(df, lap_column, interval=1.8):
    st.subheader("📡 Live Monitoring Simulation")
    placeholder_metrics, placeholder_chart = st.empty(), st.empty()
    car_ids = sorted(df["car_id"].unique())

    for lap in sorted(df[lap_column].unique()):
        with placeholder_metrics.container():
            st.write(f"Updating... Lap **{lap}**")
            live_df = df[df[lap_column] == lap]
            cols = st.columns(len(car_ids))
            for i, car in enumerate(car_ids):
                row = live_df[live_df["car_id"] == car]
                if not row.empty:
                    cols[i].metric(car, f"P{int(row['position'].iloc[0])}", f"{round(row['gap_front'].iloc[0],2)}s")

        with placeholder_chart.container():
            temp_df = df[df[lap_column] <= lap]
            fig, ax = plt.subplots(figsize=(8,4))
            ax.plot(temp_df[lap_column], temp_df["lap_time"], color="#FF3B3B")
            st.pyplot(fig)

        time.sleep(interval)
        st.experimental_rerun()

if st.sidebar.toggle("🔴 Enable Live Mode"):
    run_live_simulation(df, lap_column)
    st.stop()


# ------------------------------ ML FINISH POSITION PREDICTOR ------------------------------

@st.cache_data
def build_prediction_model(df):
    model = LinearRegression()
    model.fit(df[["lap_time", "gap_front", "gap_leader"]], df["position"])
    return model

try:
    model = build_prediction_model(df)
    predicted_finish = int(np.clip(round(model.predict([[lap_row["lap_time"], lap_row["gap_front"], lap_row["gap_leader"]]])[0]), 1, 30))
except:
    predicted_finish = "N/A"


# ------------------------------ AI PIT STRATEGY ------------------------------

def calculate_best_pit(df):
    scores = {
        lap: 30 - (((df[df[lap_column] > lap]["lap_time"].iloc[-1] - df[df[lap_column] > lap]["lap_time"].iloc[0]) / len(df[df[lap_column] > lap])) * 5)
        for lap in unique_laps if len(df[df[lap_column] > lap]) >= 3
    }
    return min(scores, key=scores.get) if scores else None

best_lap = calculate_best_pit(car_df)


# ------------------------------ MAIN UI ------------------------------

if menu == "🏁 Dashboard":
    
    st.markdown("<h1 style='color:#FF3B3B;'>🏎️ RaceAI — Live Strategy Dashboard</h1>", unsafe_allow_html=True)

    # ---- Status ----
    st.subheader("Race Status")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Position", int(lap_row["position"]))
    c2.metric("Gap to Leader", round(lap_row.get("gap_leader", 0), 2))
    c3.metric("Gap to Front", round(lap_row.get("gap_front", 0), 2))
    c4.metric("Predicted Finish", predicted_finish)

    # ---- Pit Window ----
    st.subheader("Pit Strategy Suggestion")
    if best_lap:
        st.success(f"Recommended Pit Window: Lap **{best_lap}**")
    else:
        st.warning("Not enough race trend detected.")

    # ---- Driver Performance ----
    st.subheader("Driver Performance")
    a, b, c, d = st.columns(4)
    a.metric("Best Lap", round(car_df["lap_time"].min(), 2))
    b.metric("Average Lap", round(car_df["lap_time"].mean(), 2))
    c.metric("Consistency (σ)", round(car_df["lap_time"].std(), 2))
    d.metric("Pit Count", int(car_df.get("pits", 0).sum()))

    # ---- Graph ----
    st.subheader("Lap Time Trend")
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(car_df[lap_column], car_df["lap_time"], color="#FF3B3B")
    st.pyplot(fig)


    # ---------------- TELEMETRY SIMULATION ----------------

    st.subheader("📡 Live Telemetry Feed")

    seed = selected_lap + hash(selected_car) % 50
    np.random.seed(seed)

    telemetry = {
        "Speed (km/h)": round(210 - (lap_row["lap_time"] - car_df["lap_time"].min())*2 + np.random.randint(-3, 3)),
        "Fuel %": max(5, round(100 - selected_lap * (100 / max(unique_laps)), 1)),
        "Tyre Temp FL": np.random.randint(85, 105),
        "Tyre Temp FR": np.random.randint(85, 108),
        "Brake Temp (°C)": np.random.randint(420, 610)
    }

    tcols = st.columns(3)
    i = 0
    for k, v in telemetry.items():
        tcols[i].metric(k, v)
        i = (i + 1) % 3

    if telemetry["Brake Temp (°C)"] > 550:
        st.error("🔥 Brake overheating — recommend conservative braking.")
    if telemetry["Fuel %"] < 15:
        st.warning("⛽ Fuel critically low — pit required.")


# ------------------------------ DRIVER COMPARISON ------------------------------

elif menu == "📈 Driver Comparison":

    st.markdown("<h1 style='color:#33FFDA;'>📈 Driver Comparison Mode</h1>", unsafe_allow_html=True)

    compare_driver = st.selectbox("Compare with:", [c for c in cars if c != selected_car])

    compare_df = df[df["car_id"] == compare_driver]
    compare_df = compare_df[(compare_df["lap_time"] > 10) & (compare_df["lap_time"] < 400)]

    fig2, ax2 = plt.subplots(figsize=(10,4))
    ax2.plot(car_df[lap_column], car_df["lap_time"], color="#FF3B3B", label=selected_car)
    ax2.plot(compare_df[lap_column], compare_df["lap_time"], color="#00FFBA", label=compare_driver)
    ax2.legend()
    st.pyplot(fig2)


# ------------------------------ EXPORT REPORT ------------------------------

elif menu == "📤 Export Report":

    st.markdown("<h1 style='color:#4FB1FF;'>📤 Export Race Report</h1>", unsafe_allow_html=True)

    fig_report, ax_report = plt.subplots(figsize=(6,3))
    ax_report.plot(car_df[lap_column], car_df["lap_time"], color="#FF3B3B")
    img_path = "lap_plot.png"
    fig_report.savefig(img_path, dpi=150, bbox_inches="tight")

    pdf_path = f"{selected_car}_race_report.pdf"

    if st.button("Generate PDF"):

        c = canvas.Canvas(pdf_path, pagesize=A4)
        w, h = A4

        c.setFont("Helvetica-Bold", 20)
        c.drawString(50, h - 50, "RaceAI Performance Report")

        c.setFont("Helvetica", 12)
        c.drawString(50, h - 90, f"Driver: {selected_car}")
        c.drawString(50, h - 110, f"Dataset: {selected_file}")
        c.drawString(50, h - 130, f"Best Lap: {round(car_df['lap_time'].min(),2)} sec")
        c.drawString(50, h - 150, f"Avg Lap: {round(car_df['lap_time'].mean(),2)} sec")
        c.drawString(50, h - 170, f"Recommended Pit Lap: {best_lap if best_lap else 'N/A'}")

        c.drawImage(img_path, 50, h - 420, width=450, preserveAspectRatio=True)
        c.save()

        with open(pdf_path, "rb") as f:
            st.download_button("Download Full Race Report", f, file_name=pdf_path)


# ------------------------------ MONTE CARLO SIMULATION ------------------------------

st.subheader(" Monte-Carlo Pit Window Confidence")

sim_runs = 200
noise = np.random.normal(0, 2, sim_runs)
results = [best_lap + (n/10) for n in noise]

fig_sim, ax_sim = plt.subplots(figsize=(8,3))
ax_sim.hist(results, bins=20, color="#00FFBA")
st.pyplot(fig_sim)

confidence = round((abs(np.mean(results) - best_lap) < 1) * 100)
st.success(f"Optimal Pit Call Confidence: **{confidence}%**")


# ------------------------------ AUDIO PIT CALL ------------------------------

# try:
#     import pyttsx3
#     if "tts_engine" not in st.session_state:
#         st.session_state.tts_engine = pyttsx3.init()
#     AUDIO = True
# except:
#     AUDIO = False

if st.button("🎙️ Play Engineer Pit Call"):
    if AUDIO:
        try:
            st.session_state.tts_engine.say(
                f"Box, box, box. Recommended pit stop at lap {best_lap}. Repeat. Box box."
            )
            st.session_state.tts_engine.runAndWait()
        except RuntimeError:
            st.warning("Audio engine busy. Try again.")
    else:
        st.warning("Speech engine not installed.")


# ------------------------------ FOOTER ------------------------------

st.sidebar.info("RaceAI Engine Ready 🚦")
