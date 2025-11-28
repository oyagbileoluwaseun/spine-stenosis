import os
import io
import json
import tempfile
from typing import List, Dict, Optional

import streamlit as st

# =========================================================
# 1. UTILITIES TO LOAD KEYS & SECRETS
# =========================================================

def _load_local_secrets_file() -> Dict[str, str]:
    """
    Manual fallback: read .streamlit/secrets.toml relative to project root.
    Useful if st.secrets is not populated for some reason.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    secrets_path = os.path.join(base_dir, ".streamlit", "secrets.toml")
    data: Dict[str, str] = {}

    if not os.path.exists(secrets_path):
        return data

    try:
        # Python 3.11+: tomllib in stdlib; older: use toml package
        try:
            import tomllib as toml_lib  # type: ignore[attr-defined]
        except Exception:
            import toml as toml_lib  # type: ignore

        with open(secrets_path, "rb") as f:
            toml_data = toml_lib.load(f)

        for k, v in toml_data.items():
            if isinstance(v, (str, int, float, bool)):
                data[k] = str(v)
    except Exception:
        pass

    return data


def _get_chatbot_api_key() -> Optional[str]:
    """
    Resolve the chatbot API key in a robust order:

    1) st.secrets["GOOGLE_API_KEY_CHATBOT"] / ["GEMINI_API_KEY_CHATBOT"]
    2) st.secrets["GOOGLE_API_KEY"] / ["GEMINI_API_KEY"]
    3) .streamlit/secrets.toml (manually parsed)
    4) Environment variables with the same names
    5) Optional config.get_config() values
    """
    # 1) Streamlit secrets
    secrets_dict: Dict[str, str] = {}
    try:
        secrets_dict = dict(st.secrets)
    except Exception:
        secrets_dict = {}

    key = (
        secrets_dict.get("GOOGLE_API_KEY_CHATBOT")
        or secrets_dict.get("GEMINI_API_KEY_CHATBOT")
        or secrets_dict.get("GOOGLE_API_KEY")
        or secrets_dict.get("GEMINI_API_KEY")
    )

    # 2) Local secrets file
    if not key:
        file_secrets = _load_local_secrets_file()
        key = (
            file_secrets.get("GOOGLE_API_KEY_CHATBOT")
            or file_secrets.get("GEMINI_API_KEY_CHATBOT")
            or file_secrets.get("GOOGLE_API_KEY")
            or file_secrets.get("GEMINI_API_KEY")
        )

    # 3) Environment variables
    if not key:
        key = (
            os.environ.get("GOOGLE_API_KEY_CHATBOT")
            or os.environ.get("GEMINI_API_KEY_CHATBOT")
            or os.environ.get("GOOGLE_API_KEY")
            or os.environ.get("GEMINI_API_KEY")
        )

    # 4) Optional config.get_config()
    if not key:
        try:
            from config import get_config  # type: ignore

            cfg = get_config()
            key = (
                cfg.get("GEMINI_API_KEY_CHATBOT")
                or cfg.get("GOOGLE_API_KEY_CHATBOT")
                or cfg.get("GEMINI_API_KEY")
                or cfg.get("GOOGLE_API_KEY")
            )
        except Exception:
            pass

    return key


CHATBOT_API_KEY: Optional[str] = _get_chatbot_api_key()

# =========================================================
# 2. GEMINI CLIENT SETUP (CHATBOT-ONLY, DYNAMIC MODEL)
# =========================================================

_GENAI_OK: bool = False
_GENAI_ERR: Optional[str] = None
_CHATBOT_MODEL_ID: Optional[str] = None  # resolved from list_models()

try:
    import google.generativeai as genai  # type: ignore

    if CHATBOT_API_KEY:
        genai.configure(api_key=CHATBOT_API_KEY)

        # Discover a valid model dynamically
        try:
            models = list(genai.list_models())
            candidates: List[str] = []
            for m in models:
                name = getattr(m, "name", "") or ""
                methods = set(getattr(m, "supported_generation_methods", []) or [])
                if "generateContent" in methods:
                    candidates.append(name)

            preferred_keywords = ["flash", "pro"]
            chosen = None
            for kw in preferred_keywords:
                for n in candidates:
                    if kw in n:
                        chosen = n
                        break
                if chosen:
                    break
            if not chosen and candidates:
                chosen = candidates[0]

            if chosen:
                _CHATBOT_MODEL_ID = chosen
                _GENAI_OK = True
            else:
                _GENAI_OK = False
                _GENAI_ERR = "No model with generateContent found for this API key."

        except Exception as e:
            _GENAI_OK = False
            _GENAI_ERR = f"list_models failed: {e}"

    else:
        _GENAI_OK = False
        _GENAI_ERR = "No chatbot API key found in secrets or environment."

except Exception as _e:  # pragma: no cover
    _GENAI_OK = False
    _GENAI_ERR = str(_e)


def _pick_genai_model() -> Optional[str]:
    """Return the discovered chat model."""
    return _CHATBOT_MODEL_ID if _GENAI_OK else None


# =========================================================
# 3. OPTIONAL TEXT-TO-SPEECH (gTTS)
# =========================================================

_TTS_OK: bool = False
_TTS_ERR: Optional[str] = None

try:
    from gtts import gTTS  # type: ignore

    _TTS_OK = True
except Exception as _e:
    _TTS_OK = False
    _TTS_ERR = str(_e)


def synthesize_tts(text: str, lang: str = "en") -> Optional[bytes]:
    """
    Use gTTS to synthesise speech for the given text.
    Returns MP3 bytes or None if TTS unavailable / failed.
    """
    if not _TTS_OK:
        return None
    try:
        tts = gTTS(text=text, lang=lang)
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception as e:
        st.session_state["TTS_LAST_ERROR"] = str(e)
        return None


# =========================================================
# 4. OPTIONAL SPEECH-TO-TEXT (Whisper + mic)
# =========================================================

_STT_OK: bool = False
_STT_ERR: Optional[str] = None
_WHISPER_MODEL = None

try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore
    import whisper  # type: ignore

    # You can override with WHISPER_MODEL env var ("tiny", "base", "small", etc.)
    whisper_model_name = os.environ.get("WHISPER_MODEL", "base")
    _WHISPER_MODEL = whisper.load_model(whisper_model_name)
    _STT_OK = True
except Exception as _e:
    _STT_OK = False
    _STT_ERR = str(_e)
    _WHISPER_MODEL = None


def transcribe_audio_bytes(audio_bytes: bytes) -> Optional[str]:
    """
    Transcribe raw audio bytes using Whisper.
    Returns transcript text or None on failure.
    """
    if not _STT_OK or _WHISPER_MODEL is None:
        return None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        result = _WHISPER_MODEL.transcribe(tmp_path, language="en")
        text = (result.get("text") or "").strip()
        return text or None
    except Exception as e:
        st.session_state["STT_LAST_ERROR"] = str(e)
        return None


# =========================================================
# 5. LOAD CHATBOT CONTENT FROM JSON
# =========================================================

def _load_chatbot_content() -> Dict:
    """
    Load FAQ, fallback texts and generic sections from assets/chatbot_content.json.
    """
    default = {
        "app_name": "Spine Stenosis Assistant",
        "description": "",
        "faq": [],
        "fallback": {
            "no_llm_available": (
                "I cannot reach the assistant right now. "
                "I can only answer questions about how to use this research UI. "
                "For any personal medical questions, please contact a qualified clinician."
            ),
            "medical_disclaimer": (
                "I can help with questions about how to use this research UI, "
                "but I cannot provide medical advice, diagnose conditions or recommend treatments."
            ),
        },
        "fallback_sections": {},
    }

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(base_dir, "assets", "chatbot_content.json")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        default.update(data)
    except Exception:
        pass
    return default


CHAT_CFG = _load_chatbot_content()


def _faq_markdown() -> str:
    """Render FAQ list from JSON into markdown."""
    faq_items = CHAT_CFG.get("faq", []) or []
    if not faq_items:
        return "No FAQ has been configured yet."
    lines: List[str] = []
    for i, item in enumerate(faq_items, start=1):
        q = (item.get("question") or "").strip()
        a = (item.get("answer") or "").strip()
        if not q and not a:
            continue
        lines.append(f"**{i}. {q}**\n\n{a}\n")
    return "\n".join(lines)


_FAQ_MD = _faq_markdown()

_SUPPORT_SYSTEM_PROMPT = """
You are a helpful support assistant for the *Spine Stenosis Assistant* research UI.

Your responsibilities:
- Answer questions about how to use the application (uploading scans, understanding the interface, Grad-CAM, SHAP,
  model performance views, quality-assurance page, case reports and general app behaviour).
- You may provide high-level educational information about spinal stenosis, MRI and machine learning, but:
  - Do NOT give personalised medical advice.
  - Do NOT diagnose, recommend treatments or medications.
  - Always remind the user that this tool is experimental and not a diagnostic device.

If a question is clearly about personal health, diagnosis, prognosis or treatment, politely refuse and tell the user to
speak to a qualified clinician, then briefly re-focus on what the UI can and cannot do.
""".strip()


# =========================================================
# 6. CORE CHATBOT LOGIC
# =========================================================

def _chat_support_answer(question: str, history: List[Dict]) -> str:
    """
    Use Gemini (if available) to answer support questions about the app.
    history: list of {"role": "user"/"assistant", "content": "..."}.
    """
    # If no LLM available at all, return fallback + FAQ from JSON
    if not _GENAI_OK or not CHATBOT_API_KEY or not _pick_genai_model():
        fb = CHAT_CFG.get("fallback", {}) or {}
        base = fb.get("no_llm_available") or (
            "I cannot reach the assistant right now. "
            "Here is the built-in FAQ about this UI instead."
        )
        return base + "\n\n" + _FAQ_MD

    app_name = CHAT_CFG.get("app_name", "this UI")
    description = CHAT_CFG.get("description", "")

    # Build short conversation snippet (last few turns)
    def _fmt(msg: Dict) -> str:
        return f"{msg['role']}: {msg['content']}"

    convo_snippet = "\n".join(_fmt(m) for m in history[-6:])

    model_id = _pick_genai_model()
    model = genai.GenerativeModel(model_id)  # type: ignore[arg-type]

    prompt = (
        _SUPPORT_SYSTEM_PROMPT
        + f"\n\nApplication name: {app_name}\nDescription: {description}\n\n"
        + "Here is a FAQ about the application:\n"
        + _FAQ_MD.strip()
        + "\n\nConversation so far:\n"
        + (convo_snippet or "[no previous messages]")
        + "\n\nUser question:\n"
        + question.strip()
        + "\n\nAssistant answer (support for the UI only, no medical advice):"
    )

    try:
        resp = model.generate_content(prompt)
        answer = (getattr(resp, "text", None) or "").strip()
        if not answer and getattr(resp, "candidates", None):
            try:
                answer = resp.candidates[0].content.parts[0].text.strip()
            except Exception:
                pass
        if not answer:
            raise RuntimeError("Empty response from Gemini.")

        # Append short medical disclaimer from JSON
        disclaimer = (CHAT_CFG.get("fallback", {}) or {}).get("medical_disclaimer", "")
        if disclaimer:
            answer += "\n\n---\n" + disclaimer
        return answer

    except Exception as e:
        # Capture real error for diagnostics (but not shown to end users)
        st.session_state["CHATBOT_LAST_ERROR"] = str(e)

        fb = CHAT_CFG.get("fallback", {}) or {}
        base = fb.get("no_llm_available") or (
            "I’m unable to connect to the assistant right now. "
            "Here is the built-in FAQ about this UI instead."
        )
        return base + "\n\n" + _FAQ_MD


# =========================================================
# 7. STREAMLIT PAGE UI
# =========================================================

st.set_page_config(
    page_title="Chatbot – Spine Stenosis Assistant",
    page_icon="🤖",
    layout="wide",
)

st.title("🤖 Chatbot – Application Support")

st.markdown(
    "This chatbot answers questions about **how to use the Spine Stenosis Assistant UI** – "
    "for example, uploading scans, understanding severity labels, Grad-CAM overlays, "
    "model performance pages and the Quality Assurance tools.\n\n"
    "**It cannot provide medical advice, diagnose conditions or recommend treatments.**"
)

# --- Developer diagnostics (no secrets exposed) ---
with st.expander("🛠️ Developer diagnostics (not visible to end users by default)", expanded=False):
    st.write(
        {
            "CHATBOT_API_KEY_found": bool(CHATBOT_API_KEY),
            "GENAI_OK": _GENAI_OK,
            "GENAI_ERR": _GENAI_ERR,
            "CHATBOT_MODEL_ID": _CHATBOT_MODEL_ID,
            "CHATBOT_LAST_ERROR": st.session_state.get("CHATBOT_LAST_ERROR"),
            "TTS_OK": _TTS_OK,
            "TTS_ERR": _TTS_ERR,
            "TTS_LAST_ERROR": st.session_state.get("TTS_LAST_ERROR"),
            "STT_OK": _STT_OK,
            "STT_ERR": _STT_ERR,
            "STT_LAST_ERROR": st.session_state.get("STT_LAST_ERROR"),
        }
    )

cols = st.columns([0.7, 0.3])

with cols[1]:
    with st.expander("📄 View built-in FAQ", expanded=False):
        st.markdown(_FAQ_MD)

with cols[0]:
    # Initialise chat-related state
    if "chatbot_messages" not in st.session_state:
        st.session_state["chatbot_messages"] = []
    if "chatbot_last_answer" not in st.session_state:
        st.session_state["chatbot_last_answer"] = None
    if "stt_transcript" not in st.session_state:
        st.session_state["stt_transcript"] = None

    messages: List[Dict] = st.session_state["chatbot_messages"]

    # ---------- Voice input (mic + Whisper) ----------
    # ==========================
# Voice input (beta) – Whisper + ffmpeg check
# ==========================
with st.expander("🎙️ Conversational Chatbot Support (Voice input)", expanded=True):
    st.write(
        "Record a short question with your microphone. "
        "The audio is transcribed for you (rest assured, no data leaves your machine)."
    )

    # Initialise STT diagnostics in session_state
    st.session_state.setdefault("__stt_ok__", False)
    st.session_state.setdefault("__stt_last_error__", None)
    st.session_state.setdefault("__voice_transcript__", "")

    # Lazy import microphone component
    try:
        from streamlit_mic_recorder import mic_recorder  # type: ignore
        mic_available = True
        mic_err = None
    except Exception as e:
        mic_available = False
        mic_err = str(e)

    if not mic_available:
        st.info(
            "Microphone component not available (`streamlit-mic-recorder`). "
            "Install it in your virtualenv with:\n\n"
            "`pip install streamlit-mic-recorder`"
        )
        if mic_err:
            st.caption(f"Details: {mic_err}")
    else:
        # Record audio
        audio = mic_recorder(
            start_prompt="🎙️ Start recording",
            stop_prompt="⏹ Stop",
            key="voice_mic_recorder",
        )

        raw_bytes = None
        if audio is not None:
            # Different versions can return dict or raw bytes
            if isinstance(audio, dict) and "bytes" in audio:
                raw_bytes = audio["bytes"]
            else:
                raw_bytes = audio

            # Let user listen back to what was recorded
            st.audio(raw_bytes, format="audio/wav")

        # Transcribe button
        if raw_bytes is not None and st.button("Transcribe recording", key="btn_transcribe_voice"):
            import os
            import tempfile
            import shutil
            import subprocess

            # ---- ffmpeg sanity check (this is where WinError 2 usually comes from) ----
            try:
                ff_path = shutil.which("ffmpeg")
                if not ff_path:
                    raise RuntimeError(
                        "ffmpeg executable not found on PATH for this Python process. "
                        "From the SAME terminal where you run `streamlit run`, "
                        "ensure `ffmpeg -version` prints correctly."
                    )

                # Try running it once so any path issues show up here
                proc = subprocess.run(
                    [ff_path, "-version"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
            except Exception as e:
                st.session_state["__stt_ok__"] = False
                st.session_state["__stt_last_error__"] = f"ffmpeg check failed: {e}"
                st.error(
                    "Could not transcribe this audio because ffmpeg is not reachable "
                    "inside Python. See developer diagnostics for details."
                )
            else:
                # ---- Actual Whisper transcription ----
                try:
                    import whisper  # type: ignore
                except Exception as e:
                    st.session_state["__stt_ok__"] = False
                    st.session_state["__stt_last_error__"] = f"Whisper import failed: {e}"
                    st.error("Could not transcribe this audio. See developer diagnostics for details.")
                else:
                    tmp_path = None
                    try:
                        # Save recording to a temporary WAV file
                        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                            tmp.write(raw_bytes)
                            tmp_path = tmp.name

                        # Cache the Whisper model so we don't reload every time
                        model = st.session_state.get("__whisper_model__")
                        if model is None:
                            model = whisper.load_model("tiny")
                            st.session_state["__whisper_model__"] = model

                        # Run transcription – fp16=False so CPU-only is fine
                        result = model.transcribe(tmp_path, fp16=False, language="en")
                        text = (result.get("text") or "").strip()

                        if not text:
                            st.session_state["__stt_ok__"] = False
                            st.session_state["__stt_last_error__"] = "Transcription result was empty."
                            st.error(
                                "Could not transcribe this audio – the model returned an empty transcript."
                            )
                        else:
                            st.session_state["__stt_ok__"] = True
                            st.session_state["__stt_last_error__"] = None
                            st.session_state["__voice_transcript__"] = text
                            st.success("Transcription completed. You can copy or tweak the text below.")
                    except Exception as e:
                        st.session_state["__stt_ok__"] = False
                        st.session_state["__stt_last_error__"] = repr(e)
                        st.error("Could not transcribe this audio. See developer diagnostics for details.")
                    finally:
                        if tmp_path is not None:
                            try:
                                os.remove(tmp_path)
                            except OSError:
                                pass

    # Show transcript area if we have anything
    if st.session_state.get("__voice_transcript__"):
        st.text_area(
            "Transcript (you can edit this, then paste into the chat box below):",
            value=st.session_state["__voice_transcript__"],
            key="voice_transcript_editor",
            height=100,
        )



    # ---------- Show previous messages ----------
    for msg in messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ---------- Option to send voice transcript ----------
    pending_voice = st.session_state.get("stt_transcript")
    voice_send_clicked = False
    if pending_voice:
        with st.container():
            st.info(f"Voice transcript ready: {pending_voice}")
            voice_send_clicked = st.button("Send voice transcript as question", key="send_voice_q")

    user_text: Optional[str] = None
    if voice_send_clicked and pending_voice:
        user_text = pending_voice
        st.session_state["stt_transcript"] = None
    else:
        user_text = st.chat_input("Ask a question about this application…")

    # ---------- Handle new user message ----------
    if user_text:
        # 1) Add user message
        user_msg = {"role": "user", "content": user_text}
        messages.append(user_msg)
        with st.chat_message("user"):
            st.markdown(user_text)

        # 2) Assistant reply
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer = _chat_support_answer(user_text, messages[:-1])
                st.markdown(answer)

        # 3) Store assistant reply
        messages.append({"role": "assistant", "content": answer})
        st.session_state["chatbot_last_answer"] = answer

    # ---------- TTS playback for last assistant reply ----------
    last_answer = st.session_state.get("chatbot_last_answer")

    if last_answer:
        st.markdown("---")
        st.markdown("#### 🔊 Listen to the last reply")

        col_btn, col_audio = st.columns([0.25, 0.75])

        with col_btn:
            play_clicked = st.button(
                "Play last reply",
                key="play_tts_button",
                disabled=not _TTS_OK,
                help=None if _TTS_OK else "Text-to-speech library (gTTS) not available.",
            )

        with col_audio:
            if play_clicked:
                if not _TTS_OK:
                    st.caption("Text-to-speech is not enabled on this deployment (missing gTTS).")
                else:
                    with st.spinner("Generating audio…"):
                        audio_bytes = synthesize_tts(last_answer)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")
                        else:
                            st.caption(
                                "Could not generate audio for this reply. "
                                "Check TTS diagnostics in the developer panel."
                            )
