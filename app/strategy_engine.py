import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import InconsistentVersionWarning

# ---------------------------------------------------------
# Global setup
# ---------------------------------------------------------

# Suppress sklearn version mismatch warnings when loading models
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)

# Paths
BASE_DIR = Path(__file__).resolve().parent          # .../app
PROJECT_ROOT = BASE_DIR.parent                      # .../hack-the-track
MODELS_DIR = PROJECT_ROOT / "models"

STRATEGY_MODEL_PATH = MODELS_DIR / "strategy_model.pkl"
POSITION_MODEL_PATH = MODELS_DIR / "position_model.pkl"

# Load models once at import time
strategy_model = joblib.load(STRATEGY_MODEL_PATH)
try:
    position_model = joblib.load(POSITION_MODEL_PATH)
except Exception:
    # Fails gracefully if position model isn't present
    position_model = None


# ---------------------------------------------------------
# Lap time prediction + pit simulation
# ---------------------------------------------------------

def predict_next_lap(
    lap_number: int,
    median_pace: float,
    gap_to_front: float,
    pit_like: bool = False,
    max_laps: int = 20,
) -> float:
    """
    Predict next-lap time (seconds) given current race context.

    Parameters
    ----------
    lap_number : int
    median_pace : float
    gap_to_front : float
        Gap to car in front (seconds) – used as a relative pressure/proxy.
    pit_like : bool
        Whether this lap behaves like a pit/slow lap.
    max_laps : int
        Used for simple lap-progress normalization.

    Returns
    -------
    float
        Predicted next-lap time in seconds.
    """
    lap_progress = lap_number / max_laps if max_laps > 0 else 0.0

    sample = pd.DataFrame([{
        "lap_number": lap_number,
        "lap_progress": lap_progress,
        "median_pace": median_pace,
        "relative_gap": gap_to_front,
        "pit_like": int(bool(pit_like)),
    }])

    prediction = float(strategy_model.predict(sample)[0])
    return round(prediction, 2)


def simulate_pit_effect(
    lap_number: int,
    median_pace: float,
    gap_to_front: float,
    pit_loss_seconds: float = 30.0,
    max_laps: int = 20,
) -> dict:
    """
    Compare predicted next lap if we DON'T pit vs if we DO pit now.

    Returns a dict with:
        lap_number,
        no_pit_next_lap_time,
        pit_next_lap_time,
        pit_penalty_effect (positive = slower when pitting)
    """
    no_pit = predict_next_lap(
        lap_number=lap_number,
        median_pace=median_pace,
        gap_to_front=gap_to_front,
        pit_like=False,
        max_laps=max_laps,
    )

    with_pit = predict_next_lap(
        lap_number=lap_number,
        median_pace=median_pace,
        gap_to_front=gap_to_front + pit_loss_seconds,
        pit_like=True,
        max_laps=max_laps,
    )

    return {
        "lap_number": lap_number,
        "no_pit_next_lap_time": no_pit,
        "pit_next_lap_time": with_pit,
        "pit_penalty_effect": round(with_pit - no_pit, 2),
    }


def recommend_pit(sim_result: dict, threshold_gain: float = 1.5) -> str:
    """
    Turn numeric sim result into a human-readable strategy recommendation.
    """
    effect = sim_result["pit_penalty_effect"]
    lap = sim_result["lap_number"]

    if effect < -threshold_gain:
        return (
            f"Optimal pit window detected on lap {lap}. "
            f"Pitting is likely to GAIN ~{abs(effect):.2f} seconds."
        )
    elif -threshold_gain <= effect <= threshold_gain:
        return (
            f"Neutral window on lap {lap}. Strategy could go either way; "
            f"consider traffic, tires, and track evolution."
        )
    else:
        return (
            f"Not a good pit window on lap {lap}. "
            f"Pitting now may LOSE ~{effect:.2f} seconds."
        )


# ---------------------------------------------------------
# Strategy A vs B comparison
# ---------------------------------------------------------

def compare_strategies(
    lap_number: int,
    median_pace: float,
    gap_to_front: float,
    pit_loss_s: float,
    max_laps: int,
    strategyA_lap: int,
    strategyB_lap: int,
) -> dict:
    """
    Compare two simple pit strategies:
        Strategy A: pit on strategyA_lap
        Strategy B: pit on strategyB_lap

    Currently models only the next-lap effect; it's a simple but intuitive
    comparison for engineers during the race.
    """
    results = {}

    # Baseline: predicted next lap if we *don't* pit now
    baseline = predict_next_lap(
        lap_number=lap_number,
        median_pace=median_pace,
        gap_to_front=gap_to_front,
        pit_like=False,
        max_laps=max_laps,
    )

    for name, pit_lap in [("Strategy A", strategyA_lap), ("Strategy B", strategyB_lap)]:
        if lap_number == pit_lap:
            # If we pit this lap, add pit loss on top of baseline
            projected = baseline + pit_loss_s
        else:
            # If pit is in the future, just show baseline for now
            projected = baseline

        results[name] = {
            "pit_lap": int(pit_lap),
            "projected_time": round(projected, 2),
            "net_delta_vs_no_pit": round(projected - baseline, 2),
        }

    return results


# ---------------------------------------------------------
# Finish-position prediction model
# ---------------------------------------------------------

def predict_finish_position(
    lap_number: int,
    median_pace: float,
    gap_front: float,
    gap_leader: float,
) -> int | None:
    """
    Predict likely finishing position based on current context.

    Returns an integer race position (1 = P1) or None if the model
    is not available.
    """
    if position_model is None:
        return None

    sample = pd.DataFrame([{
        "lap_number": lap_number,
        "median_pace": median_pace,
        "gap_to_front_s": gap_front,
        "gap_to_leader_s": gap_leader,
    }])

    pred = int(position_model.predict(sample)[0])
    return pred


# ---------------------------------------------------------
# Global optimal pit window search
# ---------------------------------------------------------

def find_optimal_pit_window(
    car_laps: pd.DataFrame,
    current_lap: int,
    pit_loss_s: float,
    model_max_laps: int,
) -> dict | None:
    """
    Evaluate all FUTURE laps for the selected car and find the lap with
    minimal pit penalty (i.e., best time to pit).

    Returns a dict like simulate_pit_effect(), or None if not enough data.
    """
    future_laps = car_laps[car_laps["lap_number"] > current_lap]

    if future_laps.empty:
        return None

    results = []
    for _, row in future_laps.iterrows():
        sim = simulate_pit_effect(
            lap_number=int(row["lap_number"]),
            median_pace=float(row["median_pace"]),
            gap_to_front=float(row["gap_to_front_s"]),
            pit_loss_seconds=pit_loss_s,
            max_laps=model_max_laps,
        )
        results.append(sim)

    df = pd.DataFrame(results)
    best = df.loc[df["pit_penalty_effect"].idxmin()]  # smallest penalty = best lap
    return best.to_dict()


# ---------------------------------------------------------
# Self-test (only runs if you execute this file directly)
# ---------------------------------------------------------

if __name__ == "__main__":
    example_ctx = {
        "lap_number": 5,
        "median_pace": 92.5,
        "gap_to_front": 12.3,
        "pit_like": False,
    }

    pred = predict_next_lap(**example_ctx, max_laps=20)
    print(f"Sample next-lap prediction: {pred} s")

    sim = simulate_pit_effect(
        lap_number=2,
        median_pace=171.908,
        gap_to_front=0.057,
        pit_loss_seconds=30,
        max_laps=20,
    )
    print("Pit simulation:", sim)
    print("Recommendation:", recommend_pit(sim))

    if position_model is not None:
        finish = predict_finish_position(
            lap_number=10,
            median_pace=95.0,
            gap_front=3.2,
            gap_leader=25.0,
        )
        print("Predicted finish position:", finish)
    else:
        print("Position model not available; skipping finish prediction test.")
