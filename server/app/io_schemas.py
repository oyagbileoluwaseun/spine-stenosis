
from pydantic import BaseModel
from typing import Optional, Dict

class VersionInfo(BaseModel):
    cutpoints: dict
    small3dnet_ckpt: str
    resnet3d_ckpt: str
    xgb_path: Optional[str] = None
    meta_logreg: Optional[str] = None

class Evidence(BaseModel):
    gradcam_resnet3d_png: Optional[str] = None

class Level1Scores(BaseModel):
    p_small3dnet: Optional[float]
    p_resnet3d: Optional[float]
    p_xgboost: Optional[float] = None

class MetaInfo(BaseModel):
    p_severity: float
    cutpoints: Dict[str, float]
    version: str = "stage5_3d.meta.v1"

class InferResponse(BaseModel):
    case_id: str
    severity: str
    probabilities: Dict[str, float]
    level1: Level1Scores
    meta: MetaInfo
    evidence: Evidence
