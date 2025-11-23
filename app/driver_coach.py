import pandas as pd
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_processed" / "road-america"

def load_telemetry(session):
    telemetry_file = DATA_ROOT / f"{session}_telemetry.csv"
    return pd.read_csv(telemetry_file)

def compute_lap_trace(df, vehicle_id, lap_number):
    """Filters telemetry rows for a specific car and lap."""
    lap_df = df[(df["vehicle_id"] == vehicle_id) & (df["lap"] == lap_number)]
    
    # Normalize timeline to percentage of lap
    lap_df = lap_df.sort_values("timestamp")
    lap_df["lap_progress"] = np.linspace(0, 100, len(lap_df))
    return lap_df

def compare_laps(telemetry_df, car_id, best_lap, current_lap):
    """Returns two aligned traces: best vs selected."""
    trace_best = compute_lap_trace(telemetry_df, car_id, best_lap)
    trace_current = compute_lap_trace(telemetry_df, car_id, current_lap)

    # Align sample size for plotting consistency
    size = min(len(trace_best), len(trace_current))
    trace_best = trace_best.iloc[:size]
    trace_current = trace_current.iloc[:size]

    trace_best["speed_diff"] = trace_current["telemetry_value"].values - trace_best["telemetry_value"].values

    return trace_best, trace_current
