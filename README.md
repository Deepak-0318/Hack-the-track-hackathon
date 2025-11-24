🏁 RaceAI — Live Strategy & Telemetry System

RaceAI is an intelligent race engineering assistant built for real-time decision support, race analytics, and performance coaching.

Designed for racing teams, simulators, and hackathon innovation — this platform transforms raw lap data into actionable insights.

🚀 Features
Capability	Description
🧠 AI Pit Strategy Engine	Predicts optimal pit window using pace decay & race trend modeling
🎯 Driver Coaching Insights	Detects consistency, slow laps, and recommends target lap times
📈 Lap Time Trend Visualization	Real-time pace evolution, tyre degradation curve & race phases
⚡ Telemetry Model	Simulated speed, tyre temp, brake temp, fuel state alerts
🤖 Finish Position Prediction (ML)	Trained regression model predicts race finishing position
⚔️ Driver Comparison Mode	Compare pace, race lines & gap evolution across drivers
📊 Monte-Carlo Strategy Simulation	Confidence scoring for pit timing decisions
📄 Auto PDF Race Report Export	Export full data + charts with one click
🔴 Live Simulation Mode	Automatic lap playback with status updates


🧠 Tech Stack
Python
Streamlit (UI & dashboard)
Pandas / NumPy
Matplotlib
Scikit-Learn ML Models
ReportLab PDF Generator
pyttsx3 (Optional) for race engineer audio calls

📁 Folder Structure
📦 Hack-the-track-hackathon
 ┣ 📂 app
 ┃ ┗ app.py
 ┣ 📂 data_processed
 ┃ ┗ road-america/...
 ┣ README.md
 ┣ requirements.txt
 ┗ .streamlit/config.toml

▶ How to Run Locally
git clone https://github.com/Deepak-0318/Hack-the-track-hackathon
cd Hack-the-track-hackathon
pip install -r requirements.txt
streamlit run app/app.py

🔧 Dataset Format (Expected Columns)
car_id
lap_number / lap
lap_time_s
position
gap_to_leader_s
gap_to_front_s
pit_like


Auto-renaming handles variations (lap_number, lap#, etc.)



