from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .io_schemas import InferResponse
from .runtime import load_models, bytes_to_vol, prob_and_cam, cam_overlay_base64
import torch, numpy as np
from pathlib import Path

app = FastAPI(title="Stenosis Severity API", version="1.2",
              description="Severity + Grad-CAM overlay (PNG)")

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_small, _resnet3d, _cfg = load_models(Path(__file__).resolve().parents[1] / "manifest.json", _device)

@app.get("/health")
def health():
    return {"status": "ok", "device": str(_device), "models": {"small3dnet": True, "resnet3d": True}}

@app.get("/version")
def version():
    return {
        "cutpoints": _cfg.get("cutpoints", {"none_mild":0.35,"mild_severe":0.65}),
        "small3dnet_ckpt": _cfg["small3dnet_ckpt"],
        "resnet3d_ckpt": _cfg["resnet3d_ckpt"],
        "xgb_path": _cfg.get("xgb_bundle"),
        "meta_logreg": _cfg.get("meta_logreg"),
    }

@app.post("/infer", response_model=InferResponse)
async def infer(file: UploadFile = File(...)):
    try:
        data = await file.read()
        vol = bytes_to_vol(data, filename=file.filename)  # <— pass filename so we detect .nii/.npz
    except Exception as e:
        # Return 400 with a clear message instead of 500
        raise HTTPException(status_code=400, detail=f"Failed to parse upload: {e}")

    x = torch.from_numpy(vol[None, None, ...]).to(_device)

    p_small = torch.softmax(_small(x), 1)[0, 1].item()
    p_res, cam = prob_and_cam(x, _resnet3d, _device)
    p_meta = 0.5 * (p_small + p_res)

    cuts = _cfg.get("cutpoints", {"none_mild":0.35,"mild_severe":0.65})
    if p_meta < cuts["none_mild"]:
        sev = "none"
    elif p_meta < cuts["mild_severe"]:
        sev = "mild"
    else:
        sev = "severe"

    overlay_b64 = cam_overlay_base64(vol, cam)

    return {
        "case_id": file.filename or "upload",
        "severity": sev,
        "probabilities": {"none": 1.0 - p_meta, "mild": p_meta, "severe": max(0.0, p_meta - cuts["mild_severe"])},
        "level1": {"p_small3dnet": p_small, "p_resnet3d": p_res, "p_xgboost": None},
        "meta": {"p_severity": p_meta, "cutpoints": cuts, "version": "stage5_3d.meta.v1"},
        "evidence": {"gradcam_resnet3d_png": overlay_b64},
    }
