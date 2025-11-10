import os, json, requests
from pathlib import Path
import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide")
st.title("📊 Model Performance")

# -------- Load metrics.json --------
def load_metrics():
    url = st.secrets.get("METRICS_URL", "")
    if url:
        try:
            r = requests.get(url, timeout=20); r.raise_for_status()
            return r.json(), f"Loaded metrics from {url}"
        except Exception as e:
            st.warning(f"Could not load METRICS_URL: {e}")

    local = Path(__file__).resolve().parents[1] / "assets" / "metrics.json"
    if local.exists():
        return json.loads(local.read_text()), "Loaded metrics from model assest"

    return None, "No metrics source found."

metrics, src = load_metrics()
st.caption(src)




if not metrics:
    st.info("Provide metrics via secrets(METRICS_URL), assets/metrics.json, or upload.")
    st.stop()

mean = metrics.get("mean", {})
res  = pd.DataFrame(metrics.get("results", []))

# Summary cards
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Dice (mean)", f"{mean.get('dice', 0):.3f}")
c2.metric("Surface Dice@1mm", f"{mean.get('sds_1mm', 0):.3f}")
c3.metric("ASSD (mm)", f"{mean.get('assd_mm', 0):.3f}")
c4.metric("HD95 (mm)", f"{mean.get('hd95_mm', 0):.3f}")
c5.metric("Precision", f"{mean.get('prec', 0):.3f}")
c6.metric("Recall", f"{mean.get('recall', 0):.3f}")

st.divider()
st.subheader("Per-case metrics")
st.dataframe(res, use_container_width=True, height=360)

# ECDF helpers
def ecdf(y):
    y = np.asarray(y); y = y[~np.isnan(y)]
    x = np.sort(y); f = np.arange(1, len(x)+1)/len(x)
    return x, f

row1 = st.columns(2)
if not res.empty and "dice" in res.columns:
    x, f = ecdf(res["dice"].values)
    row1[0].plotly_chart(px.line(x=x, y=f, labels={"x":"dice","y":"F(x)"}, title="ECDF: Dice"), width="stretch")
if not res.empty and "sds_1mm" in res.columns:
    x, f = ecdf(res["sds_1mm"].values)
    row1[1].plotly_chart(px.line(x=x, y=f, labels={"x":"sds_1mm","y":"F(x)"}, title="ECDF: Surface Dice @1mm"),
                         width="stretch")

# Volume agreement + Bland–Altman
row2 = st.columns(2)
trend = None
try:
    import statsmodels.api as sm  
    trend = "ols"
except Exception:
    trend = None

if {"vol_gt_ml","vol_pred_ml"}.issubset(res.columns):
    df = res[["vol_gt_ml","vol_pred_ml"]].copy()
    row2[0].plotly_chart(px.scatter(df, x="vol_gt_ml", y="vol_pred_ml",
                                    trendline=trend, title="Volume agreement (GT vs Pred)"),
                         width="stretch")
    m = (df["vol_gt_ml"] + df["vol_pred_ml"]) / 2
    d = df["vol_pred_ml"] - df["vol_gt_ml"]
    ba = pd.DataFrame({"Mean": m, "Diff": d})
    row2[1].plotly_chart(px.scatter(ba, x="Mean", y="Diff", title="Bland–Altman: Volumes"),
                         width="stretch")

st.divider()
st.subheader("Optional static figures (assets/figs)")
figs_dir = Path(__file__).resolve().parents[1] / "assets" / "figs"
names = [
    "fig_violin_metrics.png",
    "fig_heatmap_slice_dice.png",
    "fig_scatter_volumes.png",
    "fig_ecdf_dice_sds.png",
    "fig_bland_altman_vol.png",
    "fig_binlabel_pr_logreg_vs_xgb.png",
]
cols = st.columns(2); shown = 0
for i, name in enumerate(names):
    p = figs_dir / name
    if p.exists():
        with cols[i % 2]:
            st.image(str(p), caption=name, width="stretch")
        shown += 1
if shown == 0:
    st.caption("No static figures found (optional).")
