from pathlib import Path
import pandas as pd

# -------------------------------------------------------
# Paths
# -------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent  # adjust if you save this elsewhere
DATA_DIR = BASE_DIR / "data_processed" / "road-america"

FEATURES_PATH = DATA_DIR / "strategy_features.csv"
RACE1_STATE_PATH = DATA_DIR / "race1_race_state.csv"
RACE2_STATE_PATH = DATA_DIR / "race2_race_state.csv"

print("Using:")
print("  FEATURES_PATH  :", FEATURES_PATH)
print("  RACE1_STATE    :", RACE1_STATE_PATH)
print("  RACE2_STATE    :", RACE2_STATE_PATH)

# -------------------------------------------------------
# 1) Use race1_race_state to discover required columns
# -------------------------------------------------------
race1_state = pd.read_csv(RACE1_STATE_PATH)
cols_needed = race1_state.columns.tolist()
print("\nColumns expected by app:", cols_needed)

# -------------------------------------------------------
# 2) Load strategy_features (combined features for both races)
# -------------------------------------------------------
features = pd.read_csv(FEATURES_PATH)
print("\nAvailable columns in strategy_features:")
print(features.columns.tolist())

# -------------------------------------------------------
# 3) Identify the column that marks Race 1 vs Race 2
#    (Update RACE_COL if needed)
# -------------------------------------------------------
# 🔴 IMPORTANT:
# Change this to the actual column name in strategy_features
# that contains values like "Race 1" / "Race 2".
RACE_COL = "race_session"      # <--- EDIT THIS IF NEEDED

if RACE_COL not in features.columns:
    raise ValueError(
        f"Column '{RACE_COL}' not found in strategy_features.csv. "
        "Open the file and check which column identifies Race 1 vs Race 2 "
        "(e.g. 'session_name', 'race_name', 'event', etc.) and update RACE_COL."
    )

print(f"\nUnique values in '{RACE_COL}':", features[RACE_COL].unique())

# -------------------------------------------------------
# 4) Build race1_race_state and race2_race_state from features
# -------------------------------------------------------

def build_state_for_label(label: str, out_path: Path):
    df = features[features[RACE_COL] == label].copy()
    if df.empty:
        raise ValueError(f"No rows found in strategy_features for label '{label}' "
                         f"in column '{RACE_COL}'.")

    # Make sure we only keep the columns the app expects
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(
            f"The following columns expected by the app are missing from "
            f"strategy_features for '{label}': {missing}"
        )

    state = df[cols_needed].copy()
    state["lap_number"] = state["lap_number"].astype(int)

    state.to_csv(out_path, index=False)
    print(f"✅ Wrote {out_path} with {len(state)} rows")

# Regenerate Race 1 (safe) and create Race 2
build_state_for_label("Race 1", RACE1_STATE_PATH)
build_state_for_label("Race 2", RACE2_STATE_PATH)

print("\nAll done.")
