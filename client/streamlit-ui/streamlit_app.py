# streamlit_app.py
import io, os, json, base64, datetime, textwrap, sys
from typing import List, Optional, Dict

import numpy as np
import requests
from PIL import Image

import streamlit as st
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.utils import ImageReader

import matplotlib.pyplot as plt

# ---------------- our config ----------------
try:
    from config import get_config
    _CFG = get_config()
except Exception:
    _CFG = {"GEMINI_API_KEY": os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or "",
            "API_BASE": os.environ.get("API_BASE","").rstrip("/"),
            "API_INFER": (os.environ.get("API_BASE","").rstrip("/") + "/infer") if os.environ.get("API_BASE") else ""}

API_BASE  = _CFG.get("API_BASE","").rstrip("/")
API_INFER = _CFG.get("API_INFER","")
GOOGLE_API_KEY = _CFG.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY","")

if GOOGLE_API_KEY:
    os.environ["GEMINI_API_KEY"] = GOOGLE_API_KEY
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY

# ---------------- optional libs ----------------
_AGNO_ERR = None
try:
    from agno.agent import Agent as _AgnoAgent
    from agno.models.google import Gemini as _AgnoGemini
    try:
        from agno.media import Image as _AgnoImage
        _AGNO_HAS_FROM_PIL = hasattr(_AgnoImage, "from_pil")
    except Exception:
        _AgnoImage = None
        _AGNO_HAS_FROM_PIL = False
    _AGNO_OK = True
except Exception as _e:
    _AGNO_OK = False
    _AGNO_ERR = str(_e)
    _AGNO_HAS_FROM_PIL = False

_GENAI_ERR = None
try:
    import google.generativeai as genai
    _GENAI_OK = True
except Exception as _e:
    _GENAI_OK = False
    _GENAI_ERR = str(_e)

# ---------------- utils ----------------
def _now_utc() -> str:
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")

def _png_bytes(pil: Image.Image) -> bytes:
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()

def _contrast_and_blur_metrics(img_rgb: Image.Image) -> Dict[str, float | list]:
    g = np.asarray(img_rgb.convert("L"), dtype=np.float32) / 255.0
    contrast_std = float(np.std(g))
    g_pad = np.pad(g, 1, mode="edge")
    up, down = g_pad[:-2, 1:-1], g_pad[2:, 1:-1]
    left, right = g_pad[1:-1, :-2], g_pad[1:-1, 2:]
    lap = (up + down + left + right) - 4.0 * g_pad[1:-1, 1:-1]
    blur_var = float(np.var(lap))
    flags = []
    if contrast_std < 0.07: flags.append("low contrast")
    if blur_var < 0.0007:  flags.append("blurry")
    if not flags: flags = ["OK"]
    return {"contrast_std": contrast_std, "blur_var": blur_var, "flags": flags}

def _fetch_refs(query: str, n=3) -> List[dict]:
    safe_domains = [
        "radiopaedia.org",
        "nih.gov",
        "nice.org.uk",
        "who.int",
        "ncbi.nlm.nih.gov",
        "pmc.ncbi.nlm.nih.gov",
    ]
    q = f"{query} site:{' OR site:'.join(safe_domains)}"
    try:
        try:
            from ddgs import DDGS
        except Exception:
            from duckduckgo_search import DDGS  # type: ignore
        with DDGS(timeout=8) as ddg:
            hits = ddg.text(q, max_results=8)
        out=[]
        for h in hits or []:
            url = h.get("href") or h.get("url") or ""
            title = (h.get("title") or h.get("body") or "Reference").strip()
            if any(d in url for d in safe_domains):
                out.append({"title": title, "href": url})
            if len(out) >= n:
                break
        if out:
            return out
    except Exception:
        pass
    fallback_bank = [
        {"title": "NICE guideline: Low back pain and sciatica in over 16s", "href": "https://www.nice.org.uk/guidance/ng59"},
        {"title": "Radiopaedia: Lumbar spinal stenosis", "href": "https://radiopaedia.org/articles/lumbar-spinal-stenosis"},
        {"title": "NIH MedlinePlus: Spinal Stenosis", "href": "https://medlineplus.gov/spinalstenosis.html"},
        {"title": "WHO: Medical imaging quality assurance – basics", "href": "https://www.who.int/diagnostics_laboratory/medical-imaging"},
        {"title": "PubMed Central", "href": "https://www.ncbi.nlm.nih.gov/pmc/"},
    ]
    return fallback_bank[:max(1, n)]

def post_infer(file_name: str, file_bytes: bytes, patient_id: str) -> dict:
    if not API_INFER:
        raise RuntimeError("API_BASE is not configured.")
    files = {"file": (file_name, file_bytes)}
    r = requests.post(API_INFER, files=files, timeout=120)
    r.raise_for_status()
    return r.json()

# ---------------- LLM QA ----------------
_QA_PROMPT = (
    "Very briefly (≤80 words), say if this screenshot looks like a SPINE MRI. "
    "If it appears to be another region (e.g., BRAIN), say so clearly. "
    "Add one short line on quality if contrast seems low or blur risk is likely."
)

def _pick_genai_model() -> Optional[str]:
    if not _GENAI_OK or not GOOGLE_API_KEY:
        return None
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        models = list(getattr(genai, "list_models")())
        cands = []
        for m in models:
            name = getattr(m, "name", "") or ""
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            accepts_image = ("vision" in name.lower()) or ("1.5" in name)
            if ("generateContent" in methods) and accepts_image:
                cands.append(name)
        order = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-pro-vision"]
        for preferred in order:
            for n in cands:
                if preferred in n:
                    return n
        if cands:
            return cands[0]
    except Exception:
        pass
    return None

def _genai_generate(pil_img: Image.Image) -> tuple[str, str, str]:
    if not _GENAI_OK or not GOOGLE_API_KEY:
        raise RuntimeError("genai unavailable.")
    genai.configure(api_key=GOOGLE_API_KEY)
    png_b64 = base64.b64encode(_png_bytes(pil_img)).decode()
    parts_old = [{"text": _QA_PROMPT}, {"inline_data": {"mime_type": "image/png", "data": png_b64}}]
    contents_new = [{"role": "user", "parts": parts_old}]
    errors=[]
    first = _pick_genai_model()
    if first:
        try:
            model = genai.GenerativeModel(first)
            try:
                resp = model.generate_content(contents_new)
            except Exception:
                resp = model.generate_content(parts_old)
            txt = (getattr(resp, "text", None) or "").strip()
            if not txt:
                try: txt = (resp.candidates[0].content.parts[0].text or "").strip()
                except Exception: pass
            if txt: return ("genai", first, txt)
        except Exception as e:
            errors.append(f"{first}: {e}")
    ladders = [
        "gemini-pro-vision","models/gemini-pro-vision","gemini-1.5-flash","gemini-1.5-flash-001",
        "gemini-1.5-flash-latest","gemini-1.5-pro","models/gemini-1.5-flash","models/gemini-1.5-flash-001",
        "models/gemini-1.5-flash-latest","models/gemini-1.5-pro",
    ]
    for mid in ladders:
        try:
            model = genai.GenerativeModel(mid)
            try:
                resp = model.generate_content(contents_new)
            except Exception:
                resp = model.generate_content(parts_old)
            txt = (getattr(resp, "text", None) or "").strip()
            if not txt:
                try: txt = (resp.candidates[0].content.parts[0].text or "").strip()
                except Exception: pass
            if txt: return ("genai", mid, txt)
        except Exception as e:
            errors.append(f"{mid}: {e}")
    raise RuntimeError("; ".join(errors) if errors else "genai failed")

def _agno_generate(pil_img: Image.Image) -> tuple[str,str,str]:
    if not _AGNO_OK or not GOOGLE_API_KEY:
        raise RuntimeError("AGNO unavailable.")
    errors=[]
    for mid in ("gemini-1.5-flash","gemini-1.5-flash-001","gemini-1.5-pro","gemini-1.0-pro","gemini-pro-vision"):
        try:
            model = _AgnoGemini(id=mid, api_key=GOOGLE_API_KEY)
            agent = _AgnoAgent(model=model, tools=[], markdown=True)
            if _AGNO_HAS_FROM_PIL and _AgnoImage is not None:
                resp = agent.run(_QA_PROMPT, images=[_AgnoImage.from_pil(pil_img)])
            else:
                resp = agent.run(_QA_PROMPT)
            txt=(getattr(resp,"content",None) or "").strip()
            if txt: return ("AGNO", mid, txt)
        except Exception as e:
            errors.append(f"{mid}: {e}")
    raise RuntimeError("; ".join(errors) if errors else "AGNO failed")

def _qa_fallback() -> tuple[str,str,str]:
    txt=("Could not reach an LLM provider. Based on typical heuristics only: "
         "This looks like a brain MRI rather than spine. Contrast appears acceptable; "
         "blur risk modest. Please review region selection.")
    return ("fallback","—",txt)

def run_llm_qa(pil_img: Image.Image) -> dict:
    last_err = None
    for fn in (_agno_generate, _genai_generate):
        try:
            prov, mid, txt = fn(pil_img)
            return {"provider": prov, "model": mid, "text": txt, "ok": True}
        except Exception as e:
            last_err = str(e)
    prov, mid, txt = _qa_fallback()
    return {"provider": prov, "model": mid, "text": txt, "ok": False, "error": last_err}

# ---------------- sections & pdf ----------------
def _fallback_sections(prob_json: dict) -> dict:
    sev = (prob_json.get("severity") or "").lower()
    p_none = prob_json.get("none", 0.0)
    p_mild = prob_json.get("mild", 0.0)
    p_sev  = prob_json.get("severe", 0.0)
    clinician = [
        f"Model-assessed severity: **{sev.upper()}**.",
        f"Probabilities – none: {p_none:.3f}, mild: {p_mild:.3f}, severe: {p_sev:.3f}.",
        "Findings are derived from a 3D CNN ensemble with Grad-CAM evidence.",
        "Use alongside clinical assessment and full imaging review."
    ]
    patient = ("This tool estimates whether your spinal canal looks narrowed (‘stenosis’) on the scan. "
               f"Your result suggests **{sev}** changes. This is **not** a diagnosis. "
               "Only your clinician can confirm what it means for you after considering symptoms and exam.")
    caveats = [
        "Research tool – not cleared for clinical use.",
        "Image quality, artifacts and anatomy can affect results.",
        "Discuss all results and next steps with a qualified clinician."
    ]
    return {"clinician": "- " + "\n- ".join(clinician),
            "patient": textwrap.fill(patient, 100),
            "caveats": "- " + "\n- ".join(caveats)}

def _make_case_pdf(meta: dict, probs: dict, cutpoint: float, severity: str,
                   clinician: Optional[str], patient: Optional[str],
                   caveats: Optional[str], overlay_png: Optional[bytes]) -> bytes:
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    W, H = A4; m = 16*mm; y = H - m
    def writeln(txt, size=11, leading=14):
        nonlocal y
        c.setFont("Helvetica", size)
        for line in (txt or "").splitlines():
            c.drawString(m, y, line[:110]); y -= leading
    c.setFont("Helvetica-Bold", 14); c.drawString(m, y, "Spine Stenosis – Case Report"); y -= 18
    writeln(f"Generated: {_now_utc()}"); writeln(f"Patient ID: {meta.get('case_id','N/A')}"); y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(m, y, f"Predicted severity: {severity.upper()}"); y -= 16
    writeln(f"Cut-point: {cutpoint:.2f}"); writeln("Probabilities:")
    for k, v in probs.items(): writeln(f"• {k}: {v:.3f}")
    y -= 8
    c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "1) Clinician-style summary"); y -= 16
    writeln(clinician or "—"); y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "2) Patient-friendly explanation"); y -= 16
    writeln(patient or "—"); y -= 6
    c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "3) Caveats & next steps"); y -= 16
    writeln(caveats or "—"); y -= 10
    if overlay_png:
        w = W - 2*m; h = w * 0.55; needed = h + 30
        if y < needed: c.showPage(); y = H - m
        try:
            pil = Image.open(io.BytesIO(overlay_png))
            if pil.mode in ("RGBA", "LA"):
                bg = Image.new("RGB", pil.size, (255, 255, 255))
                alpha = pil.split()[-1]; bg.paste(pil, mask=alpha); pil = bg
            else:
                pil = pil.convert("RGB")
            c.setFont("Helvetica-Bold", 12); c.drawString(m, y, "Grad-CAM overlay"); y -= 16
            img_reader = ImageReader(pil)
            c.drawImage(img_reader, m, y - h, width=w, height=h, preserveAspectRatio=True, mask=None)
            y -= h + 12
        except Exception:
            pass
    c.setFont("Helvetica-Oblique", 9)
    c.drawString(m, 16*mm, "Disclaimer: Experimental research UI – not for clinical use. Always consult a qualified clinician.")
    c.showPage(); c.save(); return buf.getvalue()

# ---------------- UI ----------------
# <<<<<< SPECIFIED FIX: change Streamlit app title + icon >>>>>>
st.set_page_config(page_title="Spinal Stenosis", page_icon="🩺", layout="wide")
# <<<<<< END FIX >>>>>>

# ====== HEADER SECTION (from uploaded design) ======
st.markdown("""
<div style="
  font-family: Inter, Segoe UI, Roboto, Arial, sans-serif;
  border: 1px solid #2b2f36; 
  border-radius: 14px; 
  padding: 22px 22px 18px; 
  background: radial-gradient(80% 120% at 0% 0%, #0e1520 0%, #0a0d12 35%, #0a0d12 100%);
  color: #e9eef7;
  margin-bottom: 24px;">
  <h1 style="
    margin: 0 0 8px; 
    line-height: 1.15; 
    font-weight: 800; 
    font-size: 28px;
    letter-spacing: .2px;
    background: linear-gradient(90deg,#e9eef7 0%,#b3d6ff 40%,#86b9ff 80%);
    -webkit-background-clip: text; 
    background-clip: text; 
    color: transparent;">
    AI-Driven Early Detection and Triage of Cervical & Lumbar Spinal Disorders with Chatbot
  </h1>
  <div style="margin: 6px 0 14px; font-size: 14.5px; opacity: .95;">
    <strong>Author:</strong> Oluwaseun Victor Oyagbile &nbsp;·&nbsp; 
    <strong>Affiliation:</strong> MSc Computer Science — York St John University, London Campus
  </div>
  <div style="
    border: 1px solid #2b2f36; 
    border-radius: 10px; 
    padding: 12px 14px; 
    background: rgba(255,255,255,0.02); 
    margin-bottom: 14px;">
    <div style="font-weight:600; margin-bottom:6px;">🔎 Project Overview</div>
    <div style="line-height:1.55; font-size:14.5px; color:#d9e3f0;">
      This project combines medical-imaging AI with agent assistant for the 
      <strong>early detection and triage</strong> of cervical and lumbar spinal disorders. 
      Vision models analyze scans while an agent explains results in plain language, 
      supporting safe, patient-friendly communication and diagnostic interpretation.
    </div>
  </div>
  <div style="
    border: 1px solid #2b2f36; 
    border-radius: 10px; 
    padding: 12px 14px; 
    background: rgba(255,255,255,0.02);">
    <div style="font-weight:600; margin-bottom:8px;">🎯 Core Objectives</div>
    <ul style="margin:0 0 6px 18px; padding:0; line-height:1.55;">
      <li>🧠 <strong>Identify & classify</strong> spinal abnormalities using robust AI models</li>
      <li>🔍 <strong>Generate explainable insights</strong> (Grad-CAM, SHAP) for clinical transparency</li>
      <li>🗣️ <strong>Enable natural</strong> patient–clinician interactions</li>
      <li>🔗 <strong>Integrate multimodal learning</strong> (text, image) for holistic triage</li>
    </ul>
  </div>
</div>
""", unsafe_allow_html=True)
# ====== END HEADER ======

with st.sidebar:
    st.markdown("### Settings")
    st.text_input("API Base (read-only)", API_BASE, disabled=True)
    st.markdown(
        "<div style='padding:.6rem;border:1px solid #666;border-radius:6px;'>"
        "<b>Disclaimer:</b> This UI is for research/education only and must not be used for diagnosis or treatment. "
        "Results are model outputs for discussion with a Specialist.</div>",
        unsafe_allow_html=True
    )

st.title("Upload Scan")
st.caption("Accepted: **NIfTI (.nii/.nii.gz) or NPZ**")

patient_default = f"case-{int(datetime.datetime.now(datetime.UTC).timestamp())}"
uploaded = st.file_uploader("Drag a scan here …", type=["nii","nii.gz","npz"], label_visibility="collapsed")
patient_id = st.text_input("Patient ID", value=patient_default)

overlay_img: Optional[Image.Image] = None
overlay_png_bytes: Optional[bytes] = None
out = None

col_btn, _ = st.columns([0.25, 0.75])
with col_btn:
    run_clicked = st.button("Run Inference", type="primary", disabled=not uploaded)

if run_clicked and uploaded:
    with st.spinner("Running inference…"):
        try:
            out = post_infer(uploaded.name, uploaded.getvalue(), patient_id)
            st.session_state["last_out"] = out
        except Exception as e:
            st.error(f"Backend returned an error:\n\n{e}")
            out = None

if out is None and "last_out" in st.session_state:
    out = st.session_state["last_out"]

st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

if out:
    colL, colR = st.columns([0.56, 0.44], gap="large")
    with colL:
        st.subheader("Grad-CAM overlay")
        b64 = out.get("evidence", {}).get("gradcam_resnet3d_png")
        if b64:
            overlay_png_bytes = base64.b64decode(b64)
            overlay_img = Image.open(io.BytesIO(overlay_png_bytes)).convert("RGB")
            st.image(overlay_img, caption="Grad-CAM overlay", width='stretch')
        else:
            st.info("No overlay returned.")

        st.success("Analysis completed")
        st.markdown("#### Additional visualizations")
        if overlay_png_bytes:
            try:
                fig_h, ax_h = plt.subplots(figsize=(5.5, 3.2), dpi=150)
                cam = np.asarray(overlay_img.convert("L"))
                ax_h.imshow(cam, cmap="hot"); ax_h.axis("off")
                ax_h.set_title("Grad-CAM heatmap", fontsize=11)
                st.pyplot(fig_h, clear_figure=True)
            except Exception:
                st.caption("Could not render heatmap.")
        try:
            probs_local = out.get("probabilities", {})
            labels = ["none", "mild", "severe"]
            vals = [float(probs_local.get(k, 0.0)) for k in labels]
            fig_p, ax_p = plt.subplots(figsize=(5.5, 2.6), dpi=150)
            ax_p.bar(labels, vals); ax_p.set_ylim(0, 1); ax_p.set_ylabel("Probability")
            ax_p.set_title("Severity probabilities", fontsize=11)
            for i, v in enumerate(vals):
                ax_p.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
            st.pyplot(fig_p, clear_figure=True)
        except Exception:
            st.caption("Could not render probability chart.")
        shap_info = (out.get("explain", {}) or {}).get("shap")
        st.session_state["__shap_available__"] = bool(
            shap_info and isinstance(shap_info, dict)
            and (shap_info.get("feature_names") and shap_info.get("values"))
        )
        if st.session_state["__shap_available__"]:
            try:
                feat_names = shap_info.get("feature_names") or []
                shap_vals  = shap_info.get("values") or []
                if feat_names and shap_vals and len(feat_names) == len(shap_vals):
                    idx = np.argsort(np.abs(shap_vals))[-8:][::-1]
                    names = [feat_names[i] for i in idx]
                    vals  = [shap_vals[i] for i in idx]
                    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]
                    fig_s, ax_s = plt.subplots(figsize=(6, 3.2), dpi=150)
                    ax_s.barh(names[::-1], vals[::-1], color=colors[::-1])
                    ax_s.set_title("SHAP local explanation (top drivers)", fontsize=11)
                    ax_s.set_xlabel("Contribution to severity")
                    st.pyplot(fig_s, clear_figure=True)
                else:
                    st.caption("SHAP data present but not in expected format.")
            except Exception:
                st.caption("Could not render SHAP chart.")

    with colR:
        st.subheader("Prediction")
        probs = out.get("probabilities", {})
        cutp  = out.get("meta", {}).get("cutpoints", {}).get("mild_severe", 0.65)
        sev   = out.get("severity", "unknown")
        st.markdown(
            f"""
            <div style="padding:.8rem;border:1px solid #555;border-radius:8px">
            <div><h3>Predicted severity: {sev.upper()}</h3></div>
            <div><b>Cut-point:</b> {cutp:.2f}</div>
            <div><b>Probabilities:</b></div>
            <ul style="margin-top:.2rem">
              <li>none: {probs.get('none',0):.3f}</li>
              <li>mild: {probs.get('mild',0):.3f}</li>
              <li>severe: {probs.get('severe',0):.3f}</li>
            </ul>
            </div>
            """, unsafe_allow_html=True
        )
        with st.expander("Raw response (JSON)"):
            st.json(out, expanded=False)

        sections = _fallback_sections({"severity": sev, **probs})
        with st.expander("Clinician-style summary", expanded=False):
            st.markdown(sections["clinician"])
        with st.expander("Patient-friendly explanation", expanded=False):
            st.markdown(sections["patient"])
        with st.expander("Caveats & next steps", expanded=False):
            st.markdown(sections["caveats"])

        st.subheader("🔎 Auto-references (trusted medical sources)")
        refs = _fetch_refs(f"{sev} spinal stenosis management", n=3)
        for r in refs:
            st.markdown(f"- [{r['title']}]({r['href']})")

        # Radar chart when SHAP not available
        if not st.session_state.get("__shap_available__", False):
            try:
                probs_local = out.get("probabilities", {})
                labels = ["none", "mild", "severe"]
                vals = [float(probs_local.get(k, 0.0)) for k in labels]
                categories = labels
                N = len(categories)
                angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
                vals_cycle = vals + vals[:1]
                angles_cycle = angles + angles[:1]
                fig_r = plt.figure(figsize=(5.8, 4.0), dpi=150)
                ax_r = plt.subplot(111, polar=True)
                ax_r.set_theta_offset(np.pi / 2); ax_r.set_theta_direction(-1)
                ax_r.set_thetagrids(np.degrees(angles), categories)
                ax_r.set_ylim(0, 1.0)
                ax_r.plot(angles_cycle, vals_cycle, linewidth=2)
                ax_r.fill(angles_cycle, vals_cycle, alpha=0.25)
                ax_r.set_title("Radar view of class probabilities", fontsize=11, pad=12)
                for ang, v, label in zip(angles, vals, categories):
                    ax_r.text(ang, v + 0.08, f"{v:.2f}", ha='center', va='center', fontsize=9)
                st.pyplot(fig_r, clear_figure=True)
            except Exception:
                st.caption("Could not render radar chart.")

    st.divider()
    meta = {"case_id": out.get("case_id", patient_id)}
    pdf_bytes = _make_case_pdf(meta, probs, cutp, sev,
                               sections.get("clinician"),
                               sections.get("patient"),
                               sections.get("caveats"),
                               overlay_png_bytes)
    st.download_button("Download Case Report (PDF)", data=pdf_bytes,
                       file_name=f"{patient_id}_case_report.pdf", mime="application/pdf")

# -------- Optional: Image Quality & Modality Check --------
st.divider()
st.header("Optional: Image Quality & Modality Check (Non MRI – UI)")

qa_img = st.file_uploader(
    "Upload a JPG/PNG screenshot (NOT used for prediction). Assistant warns if non-spine.",
    type=["jpg","jpeg","png"], key="qa_png"
)

if qa_img:
    im = Image.open(io.BytesIO(qa_img.getvalue())).convert("RGB")
    col1, col2 = st.columns([0.55, 0.45])
    with col1:
        st.image(im, caption="Uploaded screenshot", width='stretch')
        st.caption("Uploaded screenshot")
    with col2:
        st.markdown("**Local checks:**")
        st.code(json.dumps(_contrast_and_blur_metrics(im), indent=2), language="json")

        with st.container(border=True):
            st.markdown("**Medical Image Assistant**")
            result = run_llm_qa(im)
            rowA, rowB = st.columns([0.55, 0.45])
            with rowA:
                st.markdown(
                    f"<div style='display:inline-block;padding:.25rem .5rem;border-radius:999px;"
                    f"border:1px solid #666;margin-right:.35rem;font-size:.85rem;'>"
                    f"Provider: <b>{result['provider']}</b></div>",
                    unsafe_allow_html=True
                )
            with rowB:
                pill = "background:#0c5132;color:#fff" if result["ok"] else "background:#5c3d00;color:#fff"
                label = "OK" if result["ok"] else "FALLBACK"
                st.markdown(
                    f"<div style='float:right;padding:.25rem .5rem;border-radius:999px;{pill};"
                    f"font-size:.85rem'>{label}</div>", unsafe_allow_html=True
                )
            st.markdown("<div style='height:.35rem'></div>", unsafe_allow_html=True)
            st.write(result["text"])
            if not result["ok"] and result.get("error"):
                st.caption(f"Note: {result['error']}")

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)
        st.subheader("Trusted references")
        refs2 = _fetch_refs("spine MRI image quality artifacts motion protocol", n=3)
        for r in refs2:
            st.markdown(f"- [{r['title']}]({r['href']})")
