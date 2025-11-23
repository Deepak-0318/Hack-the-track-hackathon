from pathlib import Path
import shutil

# ------------------------------------------
# Paths
# ------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data_processed" / "road-america"

SRC = DATA_DIR / "race1_race_state.csv"
DST = DATA_DIR / "race2_race_state.csv"

print("📁 Source:", SRC)
print("📁 Destination:", DST)

# ------------------------------------------
# Copy operation
# ------------------------------------------
if not SRC.exists():
    raise FileNotFoundError(f"❌ Missing source file: {SRC}")

shutil.copy(SRC, DST)
print("\n✅ Success!")
print("A Race 2 dataset has been created by cloning Race 1 data.\n")
print("👉 You can now restart Streamlit and select 'Race 2' in the app.")
