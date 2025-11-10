import json
from io import BytesIO
from pathlib import Path
import requests
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Explainability (Grad-CAM gallery)", page_icon="🧠", layout="wide")
st.title("🧠 Explainability : Lumbar & Cervical MRI ")

def load_gallery():
    base_url = st.secrets.get("EXAMPLES_BASE_URL", "").rstrip("/")
    if base_url:
        try:
            meta = requests.get(f"{base_url}/meta.json", timeout=15).json()
            items = [{"title": v or k, "url": f"{base_url}/{k}"} for k, v in meta.items()]
            return items, "Loaded remote examples"
        except Exception as e:
            st.warning(f"Remote examples unavailable: {e}")

    base = Path(__file__).resolve().parents[1] / "assets" / "examples"
    meta = {}
    if (base / "meta.json").exists():
        try: meta = json.loads((base / "meta.json").read_text())
        except: pass
    items = []
    for p in sorted(base.glob("*.png")):
        items.append({"title": meta.get(p.name, p.stem), "path": str(p)})
    return items, "Loaded local assets"

items, src = load_gallery()
st.caption(src)

if not items:
    st.info("Place PNGs in assets/examples/ and (optionally) add captions in meta.json.")
    st.stop()

cols = st.columns(3)
for i, it in enumerate(items):
    with cols[i % 3]:
        try:
            if "url" in it:
                img = Image.open(BytesIO(requests.get(it["url"], timeout=20).content))
            else:
                img = Image.open(it["path"])
            st.image(img, width="stretch", caption=it["title"])
        except Exception as e:
            st.warning(f"Failed to load {it.get('url') or it.get('path')}: {e}")
