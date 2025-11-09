# Spine Stenosis UI (Streamlit)

Thin Streamlit client for your HF FastAPI:
- Upload NIfTI/NPZ -> get severity, probabilities, Grad-CAM
- Optional Gemini narrative & QA
- Performance page (metrics.json)
- Explainability gallery (examples)

## Local dev
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
# Create .streamlit/secrets.toml (DO NOT COMMIT):
# GEMINI_API_KEY="..."
# API_BASE="https://<your>.hf.space"
# Optional: METRICS_URL, EXAMPLES_BASE_URL
streamlit run streamlit_app.py
