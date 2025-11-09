
import io, json, base64, tempfile
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from monai.networks.nets import resnet18 as monai_resnet18

import nibabel as nib  # NEW: NIfTI loader

# ---- Small3DNet (lightweight 3D CNN) ----
class Small3DNet(torch.nn.Module):
    def __init__(self, in_ch=1, base=16, n_classes=2):
        super().__init__()
        def cbr(c_in, c_out, s=1):
            return torch.nn.Sequential(
                torch.nn.Conv3d(c_in, c_out, 3, s, 1, bias=False),
                torch.nn.BatchNorm3d(c_out),
                torch.nn.ReLU(inplace=True))
        self.body = torch.nn.Sequential(
            cbr(in_ch, base), cbr(base, base),
            cbr(base, base*2, 2), cbr(base*2, base*2),
            cbr(base*2, base*4, 2), cbr(base*4, base*4),
            cbr(base*4, base*8, 2), cbr(base*8, base*8),
        )
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool3d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(base*8, 2)
        )
    def forward(self, x):
        return self.head(self.body(x))

# ---------- Utilities ----------
def _zscore(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    m = float(a.mean()); s = float(a.std())
    if s == 0.0: s = 1.0
    return (a - m) / s

# -------- SAFE VOLUME LOADING (NPZ / NIfTI) --------
def _sniff_format(b: bytes) -> str:
    """Return 'npz', 'nii_gz', 'nii', or 'unknown'."""
    if len(b) >= 4 and b[:4] == b'PK\x03\x04':
        return "npz"
    if len(b) >= 2 and b[:2] == b'\x1f\x8b':
        return "nii_gz"
    if len(b) >= 348 and b[344:348] in (b"n+1\x00", b"ni1\x00"):
        return "nii"
    return "unknown"

def _load_nifti_bytes(b: bytes) -> np.ndarray:
    """Load .nii/.nii.gz -> float32 (D,H,W), z-scored."""
    # Try direct BytesIO first; fallback to temp file for robustness.
    try:
        img = nib.load(io.BytesIO(b))
    except Exception:
        suffix = ".nii.gz" if _sniff_format(b) == "nii_gz" else ".nii"
        with tempfile.NamedTemporaryFile(suffix=suffix) as f:
            f.write(b); f.flush()
            img = nib.load(f.name)

    arr = np.asanyarray(img.get_fdata())   # (X,Y,Z[,T])
    if arr.ndim == 4:
        arr = arr[..., 0]
    # Convert (X,Y,Z) -> (D,H,W) = (Z,Y,X)
    arr = np.moveaxis(arr, [0, 1, 2], [2, 1, 0])
    return _zscore(arr.astype(np.float32))

def _load_npz_bytes_strict(b: bytes) -> np.ndarray:
    """Load NPZ safely (no pickle). Expect numeric 3D/4D array."""
    bio = io.BytesIO(b)
    try:
        z = np.load(bio, allow_pickle=False)
    except ValueError as e:
        raise ValueError(
            "Rejected NPZ containing pickled objects. "
            "Upload .nii/.nii.gz or a clean NPZ saved with numeric arrays only "
            "(e.g., np.savez_compressed('vol.npz', crop=your_3d_array))."
        ) from e

    if len(z.files) == 0:
        raise ValueError("Empty NPZ archive.")
    key = "crop" if "crop" in z.files else z.files[0]
    arr = z[key]

    if arr.dtype.kind not in "fiu":
        raise ValueError(f"NPZ array must be numeric, got dtype={arr.dtype}.")
    arr = np.squeeze(arr)
    if arr.ndim not in (3, 4):
        raise ValueError(f"NPZ array must be 3D/4D, got shape {arr.shape}.")
    if arr.ndim == 4:
        arr = arr[..., 0]
    # If your NPZ is (Z,Y,X) already, no swap; otherwise adapt here.
    # Keep as-is, most of your earlier NPZ crops were (D,H,W).
    return _zscore(arr.astype(np.float32))

def load_volume_bytes(b: bytes) -> np.ndarray:
    """
    Safe entry: accepts NIfTI (.nii/.nii.gz) and clean NPZ (no pickle).
    Returns float32 (D,H,W), z-scored.
    """
    fmt = _sniff_format(b)
    if fmt in ("nii", "nii_gz"):
        return _load_nifti_bytes(b)
    if fmt == "npz":
        return _load_npz_bytes_strict(b)
    # Unknown: try NIfTI first, then fail with message
    try:
        return _load_nifti_bytes(b)
    except Exception:
        raise ValueError("Unsupported file type. Upload .nii/.nii.gz or a clean numeric .npz.")

# Backwards-compatible name used by main.py
def bytes_to_vol(b: bytes) -> np.ndarray:
    return load_volume_bytes(b)

# ---------- Visualization helpers ----------
def _mip2d(vol3d: np.ndarray, axis: int = 0) -> np.ndarray:
    """Maximum intensity projection for background grayscale."""
    v = vol3d.max(axis=axis)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return (v * 255.0).astype(np.uint8)

def _heatmap_overlay(bg_u8: np.ndarray, cam2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Colorize CAM and alpha-blend on grayscale background (PIL)."""
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
    axis = int(np.argmax(vol3d.shape))
    bg2d  = _mip2d(vol3d, axis=axis)
    cam2d = cam3d.max(axis=axis)
    over = _heatmap_overlay(bg2d, cam2d, alpha=0.55).resize((512, 512))
    buf = io.BytesIO(); over.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- Model loading ----------
def _pick_resnet_target_layer(model):
    cand = ["layer4.1.conv2", "layer4.0.conv2", "layer4[-1].conv2"]
    for name in cand:
        try:
            m = model
            for part in name.replace('[', '.').replace(']', '').split('.'):
                if part == '': continue
                m = m[int(part)] if part.isdigit() else getattr(m, part)
            return m
        except Exception:
            pass
    return getattr(model, 'layer4', model)

def load_models(manifest_path: Path, device):
    cfg = json.loads(Path(manifest_path).read_text())
    small = Small3DNet().to(device).eval()
    small.load_state_dict(torch.load(manifest_path.parent / cfg["small3dnet_ckpt"], map_location=device), strict=False)

    resnet3d = monai_resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device).eval()
    resnet3d.load_state_dict(torch.load(manifest_path.parent / cfg["resnet3d_ckpt"], map_location=device), strict=False)

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
