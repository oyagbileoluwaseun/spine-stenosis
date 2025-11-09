# ------------------------------
# Spine Stenosis – Streamlit UI (client only)
# Calls your HF FastAPI (/infer) and adds:
# LLM narratives, QA/modality check, references, PDF export
# ------------------------------

import os, io, time, json, base64, datetime, requests
import streamlit as st
from PIL import Image                               # PIL Image class
from duckduckgo_search import DDGS
from reportlab.lib.pagesizes import A4
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas
from reportlab.lib.units import mm

# LLM (optional)
from agno.agent import Agent
from agno.models.google import Gemini
from agno.media import Image as AgnoImage  # expects: url | filepath | content (bytes)

# ---------- Config from Secrets / Env ----------
API_BASE = st.secrets.get("API_BASE", os.getenv("API_BASE", "")).rstrip("/")
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY", os.getenv("GEMINI_API_KEY", ""))

st.set_page_config(page_title="Spine Stenosis Severity", page_icon="🩻", layout="wide")
st.markdown("""
<style>
.block-container { padding-top: 1.5rem; }
.card { padding:.9rem 1.1rem; border-radius:10px; border:1px solid #e6e6e6; background:#fafafa; margin-bottom:.8rem; }
.disclaimer { background:#fff5e6; border:1px solid #ffd9a3; color:#664d00; }
h1,h2,h3 { margin-top:.2rem; }
hr { margin:.2rem 0 1rem 0; }
</style>
""", unsafe_allow_html=True)

# ---------- Helpers ----------
def post_infer(file_name: str, file_bytes: bytes, patient_id: str):
    """
    Call your HF /infer endpoint and surface backend errors verbosely.
    """
    if not API_BASE:
        raise RuntimeError("API_BASE is not configured. Set it in .streamlit/secrets.toml")

    files = {"file": (file_name, file_bytes, "application/octet-stream")}
    data = {"patient_id": patient_id}
    headers = {"accept": "application/json"}
    url = f"{API_BASE}/infer"

    try:
        r = requests.post(url, files=files, data=data, headers=headers, timeout=300)
    except requests.RequestException as e:
        raise RuntimeError(f"Could not reach backend at {url}\n{e}") from e

    # Bubble up FastAPI error payloads for easier debugging
    if r.status_code >= 400:
        try:
            detail = r.json()
        except Exception:
            detail = r.text
        raise requests.HTTPError(
            f"{r.status_code} error from {url}\nResponse:\n{detail}"
        )

    try:
        return r.json()
    except Exception as e:
        raise RuntimeError(f"Backend returned non-JSON response: {r.text[:1000]}") from e


@st.cache_resource(show_spinner=False)
def build_llm(api_key: str):
    if not api_key:
        return None
    return Agent(model=Gemini(id="gemini-2.0-flash", api_key=api_key), markdown=True)


def decode_b64_image(b64: str) -> Image.Image | None:
    try:
        return Image.open(io.BytesIO(base64.b64decode(b64)))
    except Exception:
        return None


def pil_to_agno_image(img: Image.Image) -> AgnoImage:
    """Convert PIL image -> Agno Image using raw PNG bytes (content)."""
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return AgnoImage(content=buf.getvalue(), mime_type="image/png")


def auto_refs(severity: str, region_hint: str = "spine", k: int = 3):
    query = f"{region_hint} {severity or 'spinal stenosis'} site:radiopaedia.org OR site:nih.gov OR site:nice.org.uk OR site:who.int"
    out = []
    with DDGS() as ddg:
        for r in ddg.text(query, max_results=k):
            out.append({"title": r.get("title", "Reference"),
                        "href": r.get("href", ""),
                        "snippet": r.get("body", "")})
    return out


def make_pdf(meta: dict, probs: dict, cutpoint, severity: str,
             clinician: str, patient: str, overlay: Image.Image | None) -> bytes:
    """Generate PDF report in memory."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4; m = 15*mm; y = H - m

    def line(txt, s=11, b=False):
        nonlocal y
        c.setFont("Helvetica-Bold" if b else "Helvetica", s)
        c.drawString(m, y, txt); y -= 14

    # Header
    line("Spine Stenosis – Severity Report", 16, True)
    line(f"Generated: {datetime.datetime.utcnow():%Y-%m-%d %H:%M:%S} UTC", 9)
    line(f"Patient ID: {meta.get('patient_id','')}", 10)
    c.line(m, y, W-m, y); y -= 12

    # Summary
    line("Prediction Summary", 13, True)
    line(f"Predicted Severity: {severity}", 11)
    line(f"Operating Cut-point: {cutpoint}", 11)
    if probs:
        line("Class Probabilities:", 11, True)
        for k, v in probs.items():
            line(f"  - {k}: {float(v):.4f}", 10)
    y -= 6; c.line(m, y, W-m, y); y -= 12

    # LLM text
    if clinician or patient:
        line("LLM Narrative", 13, True)
        if clinician:
            line("Clinician Summary:", 11, True)
            for ln in clinician.splitlines(): line(ln, 10)
        if patient:
            line("Patient-Friendly Explanation:", 11, True)
            for ln in patient.splitlines(): line(ln, 10)
        y -= 10; c.line(m, y, W-m, y); y -= 12

    # Evidence image
    if overlay is not None:
        line("Grad-CAM Evidence", 13, True)
        max_w, max_h = W - 2*m, 120*mm
        iw, ih = overlay.size
        s = min(max_w/iw, max_h/ih)
        y_img = max(y - ih*s, 40*mm)
        c.drawImage(ImageReader(overlay), m, y_img, width=iw*s, height=ih*s,
                    preserveAspectRatio=True, mask='auto')
        y = y_img - 12

    if y < 40*mm: c.showPage()
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(m, 20*mm, "Educational use only. Not for diagnostic decisions without qualified review.")
    c.save(); buf.seek(0)
    return buf.read()


def split_llm(full_md: str):
    if not full_md: return "", ""
    lower = full_md.lower()
    if "patient" in lower and "clinician" in lower:
        clin, pat, cur = [], [], None
        for p in full_md.splitlines():
            L = p.lower()
            if "clinician" in L: cur = "c"; continue
            if "patient" in L: cur = "p"; continue
            if cur == "c": clin.append(p)
            elif cur == "p": pat.append(p)
        return ("\n".join(clin).strip() or full_md, "\n".join(pat).strip() or full_md)
    return full_md, full_md


# ---------- Header & disclaimer ----------
left, right = st.columns([0.75, 0.25])
with left:
    st.title("🩻 Spine Stenosis – Severity & Evidence")
    st.caption("Prediction & Grad-CAM come from your HF FastAPI. UI adds narrative, QA, references, and PDF.")
with right:
    st.markdown('<div class="card disclaimer">⚠️ <b>Medical disclaimer:</b> Educational support only. Requires qualified clinical review.</div>',
                unsafe_allow_html=True)
st.divider()

# ---------- Upload & run ----------
st.subheader("Upload Scan")
st.write("Accepted for inference: **NIfTI (.nii/.nii.gz)** or **NPZ**.")
uploaded = st.file_uploader("Upload file", type=["nii", "gz", "npz"])
patient_id = st.text_input("Patient ID", value=f"case-{int(time.time())}")
run = st.button("Run Inference", type="primary", use_container_width=True)

agent = build_llm(GEMINI_API_KEY) if GEMINI_API_KEY else None

out = None
overlay_img = None
clinician_text, patient_text = "", ""

if run:
    if not uploaded:
        st.warning("Please upload a file first.")
        st.stop()

    with st.spinner("Running inference on server…"):
        try:
            out = post_infer(uploaded.name, uploaded.getvalue(), patient_id)
        except requests.HTTPError as e:
            st.error(f"Backend returned an error:\n\n{e}")
            st.stop()
        except Exception as e:
            st.exception(e)
            st.stop()

    # ---- Parse server response ----
    severity = out.get("severity", "unknown")
    probs = out.get("probabilities", {}) or {}
    meta = out.get("meta", {}) or {}
    cutpoint = meta.get("cutpoint") or meta.get("cutpoints") or out.get("cutpoint")
    b64 = (out.get("evidence") or {}).get("gradcam_resnet3d_png") or (out.get("evidence") or {}).get("gradcam_png")
    overlay_img = decode_b64_image(b64) if b64 else None

    st.header("Analysis Results")
    a, b, c = st.columns(3)
    a.metric("Predicted Severity", severity)
    b.metric("Operating Cut-point", f"{cutpoint:.3f}" if isinstance(cutpoint, (int, float)) else str(cutpoint))
    c.metric("Model Stack", "Small3DNet + ResNet3D")

    st.markdown("#### Class Probabilities")
    if probs:
        st.table({k: round(float(v), 4) for k, v in probs.items()}.items())
    else:
        st.write("No probabilities returned.")

    st.markdown("#### Grad-CAM Evidence")
    if overlay_img:
        st.image(overlay_img, width="stretch", caption="Grad-CAM overlay")
    else:
        st.info("No evidence image returned.")

    # ---- LLM narrative (optional) ----
    if agent:
        with st.spinner("Generating AI narrative…"):
            ptxt = ", ".join(f"{k}: {float(v):.3f}" for k, v in probs.items()) if probs else "n/a"
            prompt = f"""
Produce two sections:
1) Clinician Summary – concise bullets (findings, confidence, suggested next steps)
2) Patient-Friendly Explanation – 5–7 short sentences

Use only:
- severity = {severity}
- probabilities = {ptxt}
- cut-point = {cutpoint}
Mention overlay focus if provided. Add limitations and when manual review is required.
"""
            images_arg = [pil_to_agno_image(overlay_img)] if overlay_img else None
            try:
                resp = agent.run(prompt, images=images_arg)
                clinician_text, patient_text = split_llm(resp.content)
            except Exception as e:
                st.warning(f"LLM narrative failed: {e}")

        st.markdown("#### 🧾 Clinician Summary")
        st.markdown(clinician_text or "_Not available_")
        st.markdown("#### 👩‍⚕️ Patient-Friendly Explanation")
        st.markdown(patient_text or "_Not available_")

    # ---- Auto-references ----
    with st.expander("🔎 Auto-references (trusted medical sources)"):
        try:
            refs = auto_refs(severity, "spine", 3)
            for r in refs:
                st.markdown(f"- [{r['title']}]({r['href']})  \n  _{r['snippet']}_")
        except Exception as e:
            st.info(f"Could not fetch references: {e}")

    # ---- PDF ----
    pdf_bytes = make_pdf({"patient_id": patient_id}, probs, cutpoint, severity,
                         clinician_text, patient_text, overlay_img)
    st.subheader("📄 Download Case Report (PDF)")
    st.download_button("Download PDF", data=pdf_bytes,
                       file_name=f"{patient_id}_stenosis_report.pdf",
                       mime="application/pdf", use_container_width=True)

# ---------- QA / Modality check (UI-only) ----------
st.divider()
st.subheader("Optional: Image Quality & Modality Check (LLM – UI only)")
st.caption("Upload a JPG/PNG screenshot (NOT used for prediction). Assistant warns if non-spine.")
qa_file = st.file_uploader("Upload JPG/PNG for quality/modality review", type=["jpg", "jpeg", "png"], key="qa")
if qa_file and agent:
    qa_img = Image.open(io.BytesIO(qa_file.getvalue()))
    qprompt = "Identify modality and anatomical region. If not spine, warn clearly. Flag motion/noise/cropping/contrast issues and give 3 improvement tips."
    with st.spinner("Reviewing image quality…"):
        try:
            resp = agent.run(qprompt, images=[pil_to_agno_image(qa_img)])
            st.markdown(resp.content)
        except Exception as e:
            st.warning(f"QA failed: {e}")
elif qa_file and not agent:
    st.info("Set GEMINI_API_KEY in secrets to enable QA.")
