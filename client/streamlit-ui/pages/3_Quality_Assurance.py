import io, os, json, base64, datetime, textwrap, sys
from typing import List, Optional, Dict

import numpy as np
from PIL import Image
import streamlit as st

# --------- minimal config for LLM keys ----------
GOOGLE_API_KEY = (
    os.environ.get("GEMINI_API_KEY")
    or os.environ.get("GOOGLE_API_KEY", "")
)

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

# ---------------- shared helpers ----------------
def _contrast_and_blur_metrics(img_rgb: Image.Image) -> Dict[str, float | list]:
    g = np.asarray(img_rgb.convert("L"), dtype=np.float32) / 255.0
    contrast_std = float(np.std(g))
    g_pad = np.pad(g, 1, mode="edge")
    up, down = g_pad[:-2, 1:-1], g_pad[2:, 1:-1]
    left, right = g_pad[1:-1, :-2], g_pad[1:-1, 2:]
    lap = (up + down + left + right) - 4.0 * g_pad[1:-1, 1:-1]
    blur_var = float(np.var(lap))
    flags = []
    if contrast_std < 0.07:
        flags.append("low contrast")
    if blur_var < 0.0007:
        flags.append("blurry")
    if not flags:
        flags = ["OK"]
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
        out = []
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
    png_bytes = io.BytesIO()
    pil_img.save(png_bytes, format="PNG")
    png_b64 = base64.b64encode(png_bytes.getvalue()).decode()
    parts_old = [{"text": _QA_PROMPT}, {"inline_data": {"mime_type": "image/png", "data": png_b64}}]
    contents_new = [{"role": "user", "parts": parts_old}]
    errors = []
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
                try:
                    txt = (resp.candidates[0].content.parts[0].text or "").strip()
                except Exception:
                    pass
            if txt:
                return ("genai", first, txt)
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
                try:
                    txt = (resp.candidates[0].content.parts[0].text or "").strip()
                except Exception:
                    pass
            if txt:
                return ("genai", mid, txt)
        except Exception as e:
            errors.append(f"{mid}: {e}")
    raise RuntimeError("; ".join(errors) if errors else "genai failed")

def _agno_generate(pil_img: Image.Image) -> tuple[str, str, str]:
    if not _AGNO_OK or not GOOGLE_API_KEY:
        raise RuntimeError("AGNO unavailable.")
    errors = []
    for mid in ("gemini-1.5-flash","gemini-1.5-flash-001","gemini-1.5-pro","gemini-1.0-pro","gemini-pro-vision"):
        try:
            model = _AgnoGemini(id=mid, api_key=GOOGLE_API_KEY)
            agent = _AgnoAgent(model=model, tools=[], markdown=True)
            if _AGNO_HAS_FROM_PIL and _AgnoImage is not None:
                resp = agent.run(_QA_PROMPT, images=[_AgnoImage.from_pil(pil_img)])
            else:
                resp = agent.run(_QA_PROMPT)
            txt = (getattr(resp, "content", None) or "").strip()
            if txt:
                return ("AGNO", mid, txt)
        except Exception as e:
            errors.append(f"{mid}: {e}")
    raise RuntimeError("; ".join(errors) if errors else "AGNO failed")

def _qa_fallback() -> tuple[str, str, str]:
    txt = (
        "Could not reach an LLM provider. Based on typical heuristics only: "
        "This looks like a brain MRI rather than spine. Contrast appears acceptable; "
        "blur risk modest. Please review region selection."
    )
    return ("fallback", "—", txt)

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

# ---------------- UI ----------------
st.title("🩺 Image Quality & Modality Check")

st.write(
    "Upload a **JPG/JPEG/PNG screenshot** (viewer capture, PACS screenshot, etc.). "
    "This page does **not** run the stenosis model analysis on your upload. It only checks your upload image "
    "type and comments on potential quality issues and references."
)

qa_img = st.file_uploader(
    "Upload a JPG/PNG screenshot (NOT used for prediction). Assistant warns if non-spine.",
    type=["jpg", "jpeg", "png"],
    key="qa_png",
)

if qa_img:
    im = Image.open(io.BytesIO(qa_img.getvalue())).convert("RGB")

    col1, col2 = st.columns([0.55, 0.45])

    with col1:
        st.image(im, caption="Uploaded screenshot", use_container_width=True)
        st.caption("Uploaded screenshot")

    with col2:
        st.markdown("**Local checks (contrast & blur):**")
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
                    unsafe_allow_html=True,
                )
            with rowB:
                pill = "background:#0c5132;color:#fff" if result["ok"] else "background:#5c3d00;color:#fff"
                label = "OK" if result["ok"] else "FALLBACK"
                st.markdown(
                    f"<div style='float:right;padding:.25rem .5rem;border-radius:999px;{pill};"
                    f"font-size:.85rem'>{label}</div>",
                    unsafe_allow_html=True,
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
else:
    st.info("Upload a spine imaging screenshot to run the quality & modality check.")
