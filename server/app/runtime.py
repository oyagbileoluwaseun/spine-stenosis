
import io, json, base64
from pathlib import Path
import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from monai.networks.nets import resnet18 as monai_resnet18

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
    return (a - a.mean()) / (a.std() + 1e-6)

def _safe_load_npz_bytes(b: bytes) -> np.ndarray:
    # Accept .npz with any key, or a raw .npy
    bio = io.BytesIO(b)
    try:
        z = np.load(bio)
        if isinstance(z, np.lib.npyio.NpzFile):
            # Prefer 'crop' if present
            if "crop" in z.files:
                arr = z["crop"]
            else:
                # first array
                arr = z[z.files[0]]
        else:
            arr = z
    except Exception:
        bio.seek(0)
        arr = np.load(bio, allow_pickle=False)
    # ensure shape [D,H,W]
    arr = np.squeeze(arr)
    assert arr.ndim == 3, f"Expected 3D crop (D,H,W), got {arr.shape}"
    return _zscore(arr)

def _mip2d(vol3d: np.ndarray, axis: int = 0) -> np.ndarray:
    """Maximum intensity projection for background grayscale."""
    # Normalize to [0,1] for viewing
    v = vol3d.max(axis=axis)
    v = (v - v.min()) / (v.max() - v.min() + 1e-8)
    return (v * 255.0).astype(np.uint8)

def _heatmap_overlay(bg_u8: np.ndarray, cam2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    """Colorize CAM and alpha-blend on grayscale background (PIL)."""
    # Normalize CAM → [0,1]
    c = cam2d - cam2d.min()
    c = c / (c.max() + 1e-8)
    # Matplotlib colormap (import here to avoid heavy import at module load)
    from matplotlib import cm
    rgb = (cm.jet(c)[..., :3] * 255.0).astype(np.uint8)  # (H,W,3)
    heat = Image.fromarray(rgb)
    bg = Image.fromarray(bg_u8).convert("L").convert("RGBA")
    heat = heat.convert("RGBA")
    # Set alpha channel from normalized cam
    a = (c * (alpha * 255)).astype(np.uint8)
    heat.putalpha(Image.fromarray(a))
    out = Image.alpha_composite(bg, heat).convert("RGB")
    return out

def cam_overlay_base64(vol3d: np.ndarray, cam3d: np.ndarray) -> str:
    # Choose axis with largest spatial size for nicer MIP
    axis = int(np.argmax(vol3d.shape))
    bg2d  = _mip2d(vol3d, axis=axis)
    cam2d = cam3d.max(axis=axis)
    over = _heatmap_overlay(bg2d, cam2d, alpha=0.55)
    over = over.resize((512, 512))
    buf = io.BytesIO()
    over.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()

# ---------- Model loading ----------
def _pick_resnet_target_layer(model):
    # try common last convs for 3D ResNet-18
    cand = ["layer4.1.conv2", "layer4.0.conv2", "layer4[-1].conv2"]
    for name in cand:
        try:
            # support dotted or indexed
            m = model
            for part in name.replace('[', '.').replace(']', '').split('.'):
                if part == '':
                    continue
                if part.isdigit():
                    m = m[int(part)]
                else:
                    m = getattr(m, part)
            return m
        except Exception:
            pass
    # fallback: last layer
    return getattr(model, 'layer4', model)

def load_models(manifest_path: Path, device):
    cfg = json.loads(Path(manifest_path).read_text())
    small = Small3DNet().to(device).eval()
    small.load_state_dict(torch.load(manifest_path.parent / cfg["small3dnet_ckpt"], map_location=device), strict=False)

    resnet3d = monai_resnet18(spatial_dims=3, n_input_channels=1, num_classes=2).to(device).eval()
    resnet3d.load_state_dict(torch.load(manifest_path.parent / cfg["resnet3d_ckpt"], map_location=device), strict=False)

    return small, resnet3d, cfg

def bytes_to_vol(b: bytes) -> np.ndarray:
    return _safe_load_npz_bytes(b)

# ---------- Grad-CAM (class-weighted last conv feature map) ----------
@torch.inference_mode()
def prob_and_cam(x: torch.Tensor, resnet3d: torch.nn.Module, device) -> tuple[float, np.ndarray]:
    feats = {}
    # grab last conv in layer4
    target_layer = resnet3d.layer4[-1].conv2
    handle = target_layer.register_forward_hook(lambda m, i, o: feats.setdefault("f", o))
    logits = resnet3d(x)
    handle.remove()

    p = torch.softmax(logits, 1)[0, 1].item()
    fmap = feats["f"][0]                            # (C,d,h,w)
    # Use class 1 (severity) weights from fc
    w = resnet3d.fc.weight[1]                       # (C,)
    C = min(fmap.shape[0], w.shape[0])
    cam = (fmap[:C] * w[:C].view(C, 1, 1, 1)).sum(0).clamp(min=0)  # (d,h,w)
    cam = F.interpolate(cam[None, None, ...], size=x.shape[2:], mode="trilinear", align_corners=False)[0, 0]
    return p, cam.detach().cpu().numpy()

