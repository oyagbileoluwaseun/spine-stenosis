import io
import os
import json
import base64
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from monai.networks.nets import resnet18 as monai_resnet18

import nibabel as nib  # <— ensure nibabel is in requirements.txt

# ---- Small3DNet (lightweight 3D CNN) ----
class Small3DNet(torch.nn.Module):
    def __init__(self, in_ch=1, base=16, n_classes=2):
        super().__init__()
        def cbr(c_in, c_out, s=1):
            return torch.nn.Sequential(
                torch.nn.Conv3d(c_in, c_out, 3, s, 1, bias=False),
                torch.nn.BatchNorm3d(c_out),
                torch.nn.ReLU(inplace=True)
            )
        self.body = torch.nn.Sequential(
            cbr(in_ch, base), cbr(base, base),
            cbr(base*1, base*2, 2), cbr(base*2, base*2),
            cbr(base*2, base*4, 2), cbr(base*4, base*4),
            cbr(base*4, base*8, 2), cbr(base*8, base*8),
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(base*8, n_classes)
        )
    def forward(self, x):
        return self.head(self.body(x))

# ---------- Utilities ----------
def _zscore(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    return (a - a.mean()) / (a.std() + 1e-6)

def _safe_load_npz_or_npy_bytes(b: bytes) -> np.ndarray:
    """
    Accept .npz (any key; prefer 'crop') or raw .npy.
    """
    bio = io.BytesIO(b)
    try:
        z = np.load(bio, allow_pickle=False)
        if isinstance(z, np.lib.npyio.NpzFile):
            arr = z["crop"] if "crop" in z.files else z[z.files[0]]
        else:
            arr = z  # .npy array
    except Exception as e:
        raise ValueError(f"Not NPZ/NPY bytes: {e}")
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume, got {arr.shape}")
    return _zscore(arr)

def _safe_load_nii_bytes(b: bytes, suffix: str = ".nii.gz") -> np.ndarray:
    """
    Write to a temporary file that nibabel can read (.nii/.nii.gz),
    then clean up. Returns (D,H,W) float32 z-scored.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tf:
            tf.write(b)
            tf.flush()
            tmp_path = tf.name

        img = nib.load(tmp_path)   # works for nii and nii.gz
        data = np.asanyarray(img.get_fdata(dtype=np.float32))
        data = np.squeeze(data)
        # MONAI/nibabel tend to return (H,W,D); we want (D,H,W)
        if data.ndim != 3:
            raise ValueError(f"NIfTI must be 3D, got shape {data.shape}")
        # Heuristic: if last axis is much larger, assume HWD and move to DHW
        # Commonly it is (H,W,D) – make it (D,H,W)
        data = np.moveaxis(data, -1, 0)
        return _zscore(data)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def bytes_to_vol(b: bytes, filename: str | None = None) -> np.ndarray:
    """
    Detect by filename (preferred) or by content, then return (D,H,W) float32.
    Supports: .npz, .npy, .nii, .nii.gz
    """
    ext = (filename or "").lower()
    if ext.endswith(".npz") or ext.endswith(".npy"):
        return _safe_load_npz_or_npy_bytes(b)
    if ext.endswith(".nii") or ext.endswith(".nii.gz"):
        # Use the exact suffix to help nibabel (correct gzip handling)
        suffix = ".nii.gz" if ext.endswith(".nii.gz") else ".nii"
        return _safe_load_nii_bytes(b, suffix=suffix)

    # Fallback: try NPZ/NPY first, then NIfTI
    try:
        return _safe_load_npz_or_npy_bytes(b)
    except Exception:
        return _safe_load_nii_bytes(b, suffix=".nii.gz")

def _mip2d(vol3d: np.ndarray, axis: int = 0) -> np.ndarray:
    v = vol3d.max(axis=axis)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return (v * 255.0).astype(np.uint8)

def _heatmap_overlay(bg_u8: np.ndarray, cam2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    c = cam2d - cam2d.min()
    c = c / (c.max() + 1e-8)
    from matplotlib import cm
    rgb = (cm.jet(c)[..., :3] * 255.0).astype(np.uint8)
    heat = Image.fromarray(rgb).convert("RGBA")
    bg = Image.fromarray(bg_u8).convert("L").convert("RGBA")
    a = (c * (alpha * 255)).astype(np.uint8)
    heat.putalpha(Image.fromarray(a))
    return Image.alpha_composite(bg, heat).convert("RGB")

def cam_overlay_base64(vol3d: np.ndarray, cam3d: np.ndarray) -> str:
    axis = int(np.argmax(vol3d.shape))  # pick the largest axis for MIP
    bg2d  = _mip2d(vol3d, axis=axis)
    cam2d = cam3d.max(axis=axis)
    over = _heatmap_overlay(bg2d, cam2d, alpha=0.55).resize((512, 512))
    buf = io.BytesIO()
    over.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- Model loading ----------
def _pick_resnet_target_layer(model):
    try:
        return model.layer4[-1].conv2
    except Exception:
        return getattr(model, 'layer4', model)

def load_models(manifest_path: Path, device):
    cfg = json.loads(Path(manifest_path).read_text())

    small = Small3DNet().to(device).eval()
    small.load_state_dict(
        torch.load(manifest_path.parent / cfg["small3dnet_ckpt"], map_location=device),
        strict=False
    )

    resnet3d = monai_resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device).eval()
    resnet3d.load_state_dict(
        torch.load(manifest_path.parent / cfg["resnet3d_ckpt"], map_location=device),
        strict=False
    )

    return small, resnet3d, cfg

# ---------- Grad-CAM ----------
@torch.inference_mode()
def prob_and_cam(x: torch.Tensor, resnet3d: torch.nn.Module, device) -> tuple[float, np.ndarray]:
    feats = {}
    target_layer = resnet3d.layer4[-1].conv2
    handle = target_layer.register_forward_hook(lambda m, i, o: feats.setdefault("f", o))
    logits = resnet3d(x)
    handle.remove()

    p = torch.softmax(logits, 1)[0, 1].item()
    fmap = feats["f"][0]                            # (C,d,h,w)
    w = resnet3d.fc.weight[1]                       # (C,)
    C = min(fmap.shape[0], w.shape[0])
    cam = (fmap[:C] * w[:C].view(C, 1, 1, 1)).sum(0).clamp(min=0)  # (d,h,w)
    cam = F.interpolate(cam[None, None, ...], size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
    return p, cam.detach().cpu().numpy()
