from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_ROOT = BASE_DIR / "data_processed"

def get_available_tracks():
    """Returns folders inside data_processed."""
    return sorted([f.name for f in DATA_ROOT.iterdir() if f.is_dir()])

def get_available_sessions(track):
    """Detects race files inside track folder."""
    track_path = DATA_ROOT / track
    files = list(track_path.glob("race*_race_state.csv"))
    return sorted([f.stem.replace("_race_state", "") for f in files])

def load_session(track, session):
    """Loads selected race file or fails gracefully."""
    file_path = DATA_ROOT / track / f"{session}_race_state.csv"
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    df = pd.read_csv(file_path)
    df["lap_number"] = df["lap_number"].astype(int)
    return df
