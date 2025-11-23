TRD Live Strategist — UI assets and branding

This folder contains art assets and notes for the Streamlit UI.

Logo
- `static/logo.svg` is a simple vector placeholder used by the app.
- Replace with a higher-fidelity SVG/PNG if you have one. Recommended sizes: width 120-320px.

Styling
- The app injects a small CSS block at the top of `app.py` to apply a dark gradient background, card containers, and button styles.
- To change the primary colors, edit the CSS block in `app.py` (search for `.stApp` and `.stButton`).

Cards
- Major sections are wrapped in `.card` containers via `st.markdown('<div class="card">', unsafe_allow_html=True)` and closed after the section. This keeps the layout visually consistent.

Notes
- Streamlit class names can change across versions; if styling stops applying, update selectors or wrap sections with explicit HTML for precise targeting.
