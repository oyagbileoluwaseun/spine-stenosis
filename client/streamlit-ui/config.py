# config.py
import os
from functools import lru_cache

try:
    import streamlit as st
    _HAS_ST = True
except Exception:
    _HAS_ST = False

@lru_cache(maxsize=1)
def get_config():
    """
    Centralised config loader:
      1) Environment variables
      2) Streamlit secrets (if running in Streamlit)
    """
    env = os.environ
    secrets = getattr(st, "secrets", {}) if _HAS_ST else {}

    gemini_api_key = env.get("GEMINI_API_KEY") or secrets.get("GEMINI_API_KEY", "")
    api_base = (secrets.get("API_BASE", env.get("API_BASE", "")) or "").rstrip("/")

    return {
        "GEMINI_API_KEY": gemini_api_key,
        "API_BASE": api_base,
        "API_INFER": f"{api_base}/infer" if api_base else "",
    }
