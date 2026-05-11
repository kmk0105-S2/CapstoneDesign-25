#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Cascade Inference Wrapper (ResNet18 기반)
#
# 의존 파일:
#   - ct_binary_resnet18.py   → CT binary 모델 구조 + load_model()
#   - us_subtype_resnet18.py  → US subtype 모델 구조 + load_model()
#
# 실행 순서:
#   1) ct_binary_resnet18.py  로 학습 → ct_binary_models/*.pth 생성
#   2) us_subtype_resnet18.py 로 학습 → us_subtype_models/*.pth 생성
#   3) 이 파일을 단독으로 실행 → cascade inference 수행
# ============================================================

import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import vit_b_16

# ============================================================
# 0) 공통 설정
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# pretrained weight 경로 (모델 구조 재현에만 필요, 실제 가중치는 .pth에서 로드)
LOCAL_RESNET_WEIGHTS = Path("/tf/models/pretrained/resnet18-f37072fd.pth")
LOCAL_VIT_WEIGHTS    = Path("/tf/models/pretrained/vit_b_16-c867db91.pth")

# CT binary 모델 저장 디렉토리 (ct_binary_resnet18.py 의 OUTPUT_DIR 과 일치)
CT_MODEL_DIR = Path("./ct_binary_models")

# US subtype 모델 저장 디렉토리 (us_subtype_resnet18.py 의 OUTPUT_DIR 과 일치)
US_MODEL_DIR = Path("./us_subtype_models")

# US subtype 클래스 정의 (us_subtype_resnet18.py 의 SUBTYPE_ID_MAP / CLASS_NAMES 와 일치)
SUBTYPE_NAME_MAP = {
    0: "clear_cell",
    1: "endometrioid",
    2: "etc",
    3: "mucinous",
    4: "serous",
}
NUM_CLASSES = len(SUBTYPE_NAME_MAP)


# ============================================================
# 1) 공통 유틸
# ============================================================
def load_local_state_dict(weight_path):
    weight_path = Path(weight_path)
    assert weight_path.exists(), f"Weight file not found: {weight_path}"
    ckpt = torch.load(weight_path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]
    return ckpt


# ============================================================
# 2) 모델 구조 정의
#    ct_binary_resnet18.py / us_subtype_resnet18.py 와 완전히 동일
# ============================================================

# ── 공통 Attention Pooling Wrapper ───────────────────────────
class PatientSliceAttentionClassifier(nn.Module):
    def __init__(self, encoder, feat_dim, hidden_dim=256, dropout=0.3, num_classes=1):
        super().__init__()
        self.encoder = encoder

        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x    = x.view(B * S, C, H, W)
        feat = self.encoder(x)
        feat = feat.view(B, S, -1)

        attn_score  = self.attn(feat)
        attn_weight = torch.softmax(attn_score, dim=1)

        pooled = (feat * attn_weight).sum(dim=1)
        logits = self.classifier(pooled)
        return logits


# ── CT Hybrid (ResNet18 + Transformer), output=1 ─────────────
class CTCNNTransformerHybridPatient(nn.Module):
    def __init__(
        self,
        d_model=256, nhead=8, num_layers=2, dim_feedforward=512,
        dropout=0.3, load_pretrained=False,
        resnet_weight_path=LOCAL_RESNET_WEIGHTS,
    ):
        super().__init__()
        backbone = models.resnet18(weights=None)
        if load_pretrained and Path(resnet_weight_path).exists():
            backbone.load_state_dict(load_local_state_dict(resnet_weight_path), strict=True)
        feat_dim    = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.proj      = nn.Linear(feat_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, 1),
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x    = x.view(B * S, C, H, W)
        feat = self.encoder(x)
        feat = feat.view(B, S, -1)
        feat = self.proj(feat)

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        return self.classifier(tokens[:, 0, :])


# ── US Hybrid (ResNet18 + Transformer), output=NUM_CLASSES ───
class USCNNTransformerHybridPatient(nn.Module):
    def __init__(
        self,
        d_model=256, nhead=8, num_layers=2, dim_feedforward=512,
        dropout=0.3, load_pretrained=False,
        resnet_weight_path=LOCAL_RESNET_WEIGHTS,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()
        backbone = models.resnet18(weights=None)
        if load_pretrained and Path(resnet_weight_path).exists():
            backbone.load_state_dict(load_local_state_dict(resnet_weight_path), strict=True)
        feat_dim    = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.proj      = nn.Linear(feat_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 16, d_model))

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=0.1, batch_first=True, activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256), nn.ReLU(), nn.Dropout(dropout), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x    = x.view(B * S, C, H, W)
        feat = self.encoder(x)
        feat = feat.view(B, S, -1)
        feat = self.proj(feat)

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]
        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        return self.classifier(tokens[:, 0, :])


# ============================================================
# 3) CT binary 모델 로드
#    ct_binary_resnet18.py 의 load_model() 와 동일 구조
# ============================================================
def _build_ct_resnet18():
    encoder    = models.resnet18(weights=None)
    feat_dim   = encoder.fc.in_features   # 512
    encoder.fc = nn.Identity()
    return PatientSliceAttentionClassifier(
        encoder=encoder, feat_dim=feat_dim, hidden_dim=256, dropout=0.1, num_classes=1,
    )

def _build_ct_vit():
    encoder       = vit_b_16(weights=None)
    feat_dim      = encoder.heads.head.in_features   # 768
    encoder.heads = nn.Identity()
    return PatientSliceAttentionClassifier(
        encoder=encoder, feat_dim=feat_dim, hidden_dim=512, dropout=0.3, num_classes=1,
    )

def _build_ct_hybrid():
    return CTCNNTransformerHybridPatient(load_pretrained=False)

def load_ct_model(model_type: str, model_path: str) -> nn.Module:
    """
    CT binary 모델을 로드합니다.

    Args:
        model_type : "resnet" | "vit" | "hybrid"
        model_path : ct_binary_resnet18.py 가 저장한 .pth 경로
                     예) "./ct_binary_models/ResNet18_fold1_best.pth"
    """
    builders = {"resnet": _build_ct_resnet18, "vit": _build_ct_vit, "hybrid": _build_ct_hybrid}
    if model_type not in builders:
        raise ValueError(f"Unknown CT model_type: {model_type}. Choose from {list(builders)}")

    model = builders[model_type]()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"[CT ] Loaded {model_type} from {model_path}")
    return model


# ============================================================
# 4) US subtype 모델 로드
#    us_subtype_resnet18.py 의 load_model() 와 동일 구조
# ============================================================
def _build_us_resnet18():
    encoder    = models.resnet18(weights=None)
    feat_dim   = encoder.fc.in_features   # 512
    encoder.fc = nn.Identity()
    return PatientSliceAttentionClassifier(
        encoder=encoder, feat_dim=feat_dim, hidden_dim=256, dropout=0.1, num_classes=NUM_CLASSES,
    )

def _build_us_vit():
    encoder       = vit_b_16(weights=None)
    feat_dim      = encoder.heads.head.in_features   # 768
    encoder.heads = nn.Identity()
    return PatientSliceAttentionClassifier(
        encoder=encoder, feat_dim=feat_dim, hidden_dim=512, dropout=0.3, num_classes=NUM_CLASSES,
    )

def _build_us_hybrid():
    return USCNNTransformerHybridPatient(load_pretrained=False, num_classes=NUM_CLASSES)

def load_us_model(model_type: str, model_path: str) -> nn.Module:
    """
    US subtype 모델을 로드합니다.

    Args:
        model_type : "resnet" | "vit" | "hybrid"
        model_path : us_subtype_resnet18.py 가 저장한 .pth 경로
                     예) "./us_subtype_models/ResNet18_US_Subtype_fold1_best.pth"
    """
    builders = {"resnet": _build_us_resnet18, "vit": _build_us_vit, "hybrid": _build_us_hybrid}
    if model_type not in builders:
        raise ValueError(f"Unknown US model_type: {model_type}. Choose from {list(builders)}")

    model = builders[model_type]()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device).eval()
    print(f"[US ] Loaded {model_type} from {model_path}")
    return model


# ============================================================
# 5) CT binary inference  (1명 기준)
# ============================================================
@torch.no_grad()
def predict_ct_malignancy_prob(ct_model: nn.Module, x_ct, ct_threshold=0.5):
    """
    Args:
        x_ct : np.ndarray 또는 torch.Tensor, shape [S,C,H,W] 또는 [1,S,C,H,W]
    Returns:
        dict: mal_logit, mal_prob, mal_pred
    """
    ct_model.eval()

    if isinstance(x_ct, np.ndarray):
        x_ct = torch.from_numpy(x_ct).float()
    if x_ct.dim() == 4:
        x_ct = x_ct.unsqueeze(0)   # [1, S, C, H, W]
    elif x_ct.dim() != 5:
        raise ValueError(f"x_ct shape must be [S,C,H,W] or [1,S,C,H,W], got {tuple(x_ct.shape)}")

    x_ct  = x_ct.to(device, non_blocking=True)
    logit = ct_model(x_ct)
    prob  = torch.sigmoid(logit).view(-1)[0].item()

    return {
        "mal_logit": float(logit.view(-1)[0].item()),
        "mal_prob":  float(prob),
        "mal_pred":  int(prob >= ct_threshold),   # 0=benign, 1=malignant
    }


# ============================================================
# 6) US subtype inference  (1명 기준)
# ============================================================
@torch.no_grad()
def predict_us_subtype(us_model: nn.Module, x_us):
    """
    Args:
        x_us : np.ndarray 또는 torch.Tensor, shape [S,C,H,W] 또는 [1,S,C,H,W]
    Returns:
        dict: sub_logits, sub_probs, sub_pred_idx, sub_pred_name
    """
    us_model.eval()

    if isinstance(x_us, np.ndarray):
        x_us = torch.from_numpy(x_us).float()
    if x_us.dim() == 4:
        x_us = x_us.unsqueeze(0)   # [1, S, C, H, W]
    elif x_us.dim() != 5:
        raise ValueError(f"x_us shape must be [S,C,H,W] or [1,S,C,H,W], got {tuple(x_us.shape)}")

    x_us   = x_us.to(device, non_blocking=True)
    logits = us_model(x_us)
    probs  = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())

    return {
        "sub_logits":    logits.squeeze(0).detach().cpu().numpy(),
        "sub_probs":     probs.squeeze(0).detach().cpu().numpy(),
        "sub_pred_idx":  pred_idx,
        "sub_pred_name": SUBTYPE_NAME_MAP.get(pred_idx, f"class_{pred_idx}"),
    }


# ============================================================
# 7) Cascade inference  (1명 기준)
#    CT가 benign → 종료
#    CT가 malignant → US subtype 추가 예측
# ============================================================
@torch.no_grad()
def cascade_infer_one_patient(
    ct_model: nn.Module,
    us_model: nn.Module,
    x_ct,
    x_us,
    ct_threshold=0.5,
):
    """
    Args:
        ct_model     : load_ct_model() 로 로드한 모델
        us_model     : load_us_model() 로 로드한 모델
        x_ct         : shape [S,C,H,W] 또는 [1,S,C,H,W]
        x_us         : shape [S,C,H,W] 또는 [1,S,C,H,W]
        ct_threshold : malignant 판단 임계값 (기본 0.5)
    Returns:
        dict:
            mal_logit, mal_prob, mal_pred,
            final_label,
            final_subtype_idx, final_subtype_name,  ← benign이면 None
            sub_probs                                ← benign이면 None
    """
    ct_out   = predict_ct_malignancy_prob(ct_model, x_ct, ct_threshold)
    mal_pred = ct_out["mal_pred"]

    result = {
        "mal_logit":          ct_out["mal_logit"],
        "mal_prob":           ct_out["mal_prob"],
        "mal_pred":           mal_pred,
        "final_label":        None,
        "final_subtype_idx":  None,
        "final_subtype_name": None,
        "sub_probs":          None,
    }

    if mal_pred == 0:
        result["final_label"] = "benign"
        return result

    us_out = predict_us_subtype(us_model, x_us)
    result["final_label"]        = "malignant"
    result["final_subtype_idx"]  = us_out["sub_pred_idx"]
    result["final_subtype_name"] = us_out["sub_pred_name"]
    result["sub_probs"]          = us_out["sub_probs"]

    return result


# ============================================================
# 8) 여러 환자 batch inference
# ============================================================
def build_common_patient_ids(ct_dict: dict, us_dict: dict):
    ct_ids  = set(ct_dict.keys())
    us_ids  = set(us_dict.keys())
    common  = sorted(ct_ids & us_ids)
    only_ct = sorted(ct_ids - us_ids)
    only_us = sorted(us_ids - ct_ids)

    print(f"Common patients : {len(common)}")
    print(f"Only CT         : {len(only_ct)}")
    print(f"Only US         : {len(only_us)}")

    return common, only_ct, only_us


def run_cascade_for_patients(
    ct_model: nn.Module,
    us_model: nn.Module,
    ct_inputs_by_pid: dict,   # {patient_id: x_ct}
    us_inputs_by_pid: dict,   # {patient_id: x_us}
    ct_threshold=0.5,
):
    """
    Args:
        ct_inputs_by_pid : {patient_id: x_ct}  x_ct shape [S,C,H,W]
        us_inputs_by_pid : {patient_id: x_us}  x_us shape [S,C,H,W]
    Returns:
        df_result   : pd.DataFrame  (환자별 최종 결과)
        common_ids  : CT·US 둘 다 있는 환자 ID 리스트
        only_ct     : CT만 있는 환자 ID 리스트
        only_us     : US만 있는 환자 ID 리스트
    """
    common_ids, only_ct, only_us = build_common_patient_ids(ct_inputs_by_pid, us_inputs_by_pid)

    rows = []
    for pid in common_ids:
        out = cascade_infer_one_patient(
            ct_model=ct_model,
            us_model=us_model,
            x_ct=ct_inputs_by_pid[pid],
            x_us=us_inputs_by_pid[pid],
            ct_threshold=ct_threshold,
        )

        row = {
            "patient_id":         pid,
            "mal_prob":           out["mal_prob"],
            "mal_pred":           out["mal_pred"],
            "final_label":        out["final_label"],
            "final_subtype_idx":  out["final_subtype_idx"],
            "final_subtype_name": out["final_subtype_name"],
        }

        if out["sub_probs"] is not None:
            for k, v in enumerate(out["sub_probs"]):
                row[f"sub_prob_{SUBTYPE_NAME_MAP.get(k, k)}"] = float(v)

        rows.append(row)

    df_result = pd.DataFrame(rows)
    return df_result, common_ids, only_ct, only_us


# ============================================================
# 9) 실행 예시
# ============================================================
if __name__ == "__main__":

    # ── 모델 로드 ────────────────────────────────────────────
    # fold 1 기준 예시. fold 2~5는 경로만 바꿔서 동일하게 사용.

    ct_model = load_ct_model(
        model_type="resnet",
        model_path=str(CT_MODEL_DIR / "ResNet18_fold1_best.pth"),
    )

    us_model = load_us_model(
        model_type="resnet",
        model_path=str(US_MODEL_DIR / "ResNet18_US_Subtype_fold1_best.pth"),
    )

    # ── 1명 inference 예시 ───────────────────────────────────
    # x_ct, x_us 는 각 학습 코드의 Dataset.__getitem__ 이 반환하는 텐서와 동일한 shape
    #   x_ct : [S, 3, 224, 224]  (CT 전처리 완료)
    #   x_us : [S, 3, 224, 224]  (US 전처리 완료)

    # 아래는 shape 확인용 더미 데이터 (실제 사용 시 제거)
    x_ct_dummy = torch.zeros(8, 3, 224, 224)    # CT  슬라이스 8장
    x_us_dummy = torch.zeros(4, 3, 224, 224)    # US  뷰 4장

    out = cascade_infer_one_patient(
        ct_model=ct_model,
        us_model=us_model,
        x_ct=x_ct_dummy,
        x_us=x_us_dummy,
        ct_threshold=0.5,
    )
    print("\n[Single patient result]")
    for k, v in out.items():
        print(f"  {k}: {v}")

    # ── 여러 환자 inference 예시 ─────────────────────────────
    # ct_inputs_by_pid = {"P001": x_ct_1, "P002": x_ct_2, ...}
    # us_inputs_by_pid = {"P001": x_us_1, "P002": x_us_2, ...}

    ct_inputs_by_pid = {"P001": x_ct_dummy, "P002": x_ct_dummy}
    us_inputs_by_pid = {"P001": x_us_dummy, "P002": x_us_dummy}

    df_result, common_ids, only_ct, only_us = run_cascade_for_patients(
        ct_model=ct_model,
        us_model=us_model,
        ct_inputs_by_pid=ct_inputs_by_pid,
        us_inputs_by_pid=us_inputs_by_pid,
        ct_threshold=0.5,
    )

    print("\n[Batch result]")
    print(df_result.to_string())
    df_result.to_csv("cascade_inference_results.csv", index=False)
    print("\nSaved: cascade_inference_results.csv")
