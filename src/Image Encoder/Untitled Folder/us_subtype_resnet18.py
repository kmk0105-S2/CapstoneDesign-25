#!/usr/bin/env python
# coding: utf-8

# ============================================================
# Ultrasound subtype-only 5-class CV training script
# - ResNet50 → ResNet18 변경
# - CT binary 코드 스타일 / 함수명 최대한 유지
# - US 전처리(3채널 생성, CLAHE, edge, mask crop) 없음
# - 입력 NPZ가 이미 전처리된 US 이미지를 담고 있다고 가정
# - fold 파일은 patient-level split 이 완료되어 있다고 가정
# ============================================================

from torchvision import models
from torchvision.models import vit_b_16

print("=" * 60)

try:
    m1 = models.resnet18(weights=None)
    print("ResNet18 build success")
except Exception as e:
    print("Fail", e)

try:
    m2 = vit_b_16(weights=None)
    print("Vit-B/16 build success")
except Exception as e:
    print("Fail", e)

print("=" * 60)

# ============================================================
# 1) 환경 설정 + import
# ============================================================
import os
import gc
import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, recall_score, balanced_accuracy_score
)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from torchvision.models import vit_b_16, ViT_B_16_Weights, ResNet18_Weights

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

torch.backends.cudnn.benchmark = True


# ============================================================
# 2) 경로 / 하이퍼파라미터 설정
# ============================================================
print(f"현재 작업 경로: {os.getcwd()}")

DATA_ROOT  = Path("/tf/nasw/dataset001/preprocessed/us_npz")
FOLDS      = [1, 2, 3, 4, 5]

OUTPUT_DIR = Path("./us_subtype_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE     = 224
BATCH_SIZE   = 8
NUM_WORKERS  = 2
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 7

# 로컬 pretrained weight 경로
# ResNet18 공식 weight 파일명으로 변경 (torchvision 기준)
LOCAL_RESNET_WEIGHTS = Path("/tf/models/pretrained/resnet18-f37072fd.pth")
LOCAL_VIT_WEIGHTS    = Path("/tf/models/pretrained/vit_b_16-c867db91.pth")

# ------------------------------------------------------------
# 데이터셋 key / subtype remap 설정
# ------------------------------------------------------------
IMAGE_KEY      = "images"
LABEL_KEY      = "labels"
SUBTYPE_KEY    = "tumor_types"
PATIENT_ID_KEY = "patient_ids"

SUBTYPE_ID_MAP = {
    0: 0,   # clear_cell
    1: 1,   # endometrioid
    3: 2,   # etc
    5: 3,   # mucinous
    7: 4,   # serous
}

CLASS_NAMES = [
    "clear_cell",
    "endometrioid",
    "etc",
    "mucinous",
    "serous",
]
NUM_CLASSES = len(SUBTYPE_ID_MAP)
MAX_VIEWS   = None  # None이면 npz에 들어있는 view 수 그대로 사용

print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
print("FOLDS:", FOLDS)
print("LOCAL_RESNET_WEIGHTS exists:", LOCAL_RESNET_WEIGHTS.exists())
print("LOCAL_VIT_WEIGHTS exists:",    LOCAL_VIT_WEIGHTS.exists())
print("NUM_CLASSES:", NUM_CLASSES)
print("SUBTYPE_ID_MAP:", SUBTYPE_ID_MAP)


# ============================================================
# 3) 유틸 함수
# ============================================================
def inspect_fold_npz(fold_idx):
    train_npz_path = DATA_ROOT / f"fold_{fold_idx}_train.npz"
    val_npz_path   = DATA_ROOT / f"fold_{fold_idx}_val.npz"

    print("TRAIN:", train_npz_path, train_npz_path.exists())
    print("VAL:",   val_npz_path,   val_npz_path.exists())

    train_npz = np.load(train_npz_path, allow_pickle=False, mmap_mode="r")
    val_npz   = np.load(val_npz_path,   allow_pickle=False, mmap_mode="r")

    for k in train_npz.files:
        print(f"[train] {k}: shape={train_npz[k].shape}, dtype={train_npz[k].dtype}")
    for k in val_npz.files:
        print(f"[val]   {k}: shape={val_npz[k].shape}, dtype={val_npz[k].dtype}")

    if PATIENT_ID_KEY in train_npz.files:
        print(f"[train] patient ids unique={len(np.unique(train_npz[PATIENT_ID_KEY]))}")
    if PATIENT_ID_KEY in val_npz.files:
        print(f"[val]   patient ids unique={len(np.unique(val_npz[PATIENT_ID_KEY]))}")


def load_local_state_dict(weight_path):
    weight_path = Path(weight_path)
    assert weight_path.exists(), f"Weight file not found: {weight_path}"

    ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    return ckpt


def compute_multiclass_metrics(y_true, y_pred):
    y_true = np.array(y_true).astype(int)
    y_pred = np.array(y_pred).astype(int)

    if len(y_true) == 0:
        return {
            "acc":          np.nan,
            "macro_f1":     np.nan,
            "macro_recall": np.nan,
            "balanced_acc": np.nan,
            "cm":           None,
        }

    metrics = {
        "acc":          accuracy_score(y_true, y_pred),
        "macro_f1":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "cm":           confusion_matrix(y_true, y_pred),
    }
    return metrics


def prepare_train_mode(model):
    model.train()

    if not hasattr(model, "encoder"):
        return

    for name, module in model.encoder.named_modules():
        if name.startswith("layer4"):
            continue
        if isinstance(module, nn.BatchNorm2d):
            module.eval()


def normalize_imagenet_tensor(x):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
    std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)
    return (x - mean) / std


def ensure_float01(img):
    img = np.asarray(img)
    if img.dtype == np.uint8:
        return img.astype(np.float32) / 255.0

    img  = img.astype(np.float32)
    vmax = float(img.max()) if img.size > 0 else 0.0
    vmin = float(img.min()) if img.size > 0 else 0.0

    if vmin >= 0.0 and vmax <= 1.0:
        return img
    if vmin >= 0.0 and vmax <= 255.0:
        return img / 255.0

    denom = max(vmax - vmin, 1e-6)
    return (img - vmin) / denom


# ============================================================
# 4) Dataset 정의
# ============================================================
class OvarianUSNPZDataset(Dataset):
    def __init__(self, npz_path):
        super().__init__()
        self.npz = np.load(npz_path, allow_pickle=False, mmap_mode="r")

        self.images      = self.npz[IMAGE_KEY]
        self.labels      = self.npz[LABEL_KEY].astype(np.float32) if LABEL_KEY in self.npz.files else None
        self.tumor_types = self.npz[SUBTYPE_KEY].astype(np.int64)
        self.patient_ids = self.npz[PATIENT_ID_KEY] if PATIENT_ID_KEY in self.npz.files else None

        self.keep_indices     = []
        self.remapped_subtypes = []

        for i, t in enumerate(self.tumor_types.astype(int)):
            if t not in SUBTYPE_ID_MAP:
                continue
            if self.labels is not None:
                if int(self.labels[i]) != 1:
                    continue
            self.keep_indices.append(i)
            self.remapped_subtypes.append(SUBTYPE_ID_MAP[t])

        self.keep_indices      = np.asarray(self.keep_indices,     dtype=np.int64)
        self.remapped_subtypes = np.asarray(self.remapped_subtypes, dtype=np.int64)

        print(f"Loaded {npz_path.name}: total={len(self.tumor_types)}, kept={len(self.keep_indices)}")

    def __len__(self):
        return len(self.keep_indices)

    def _get_views(self, idx):
        raw_idx = int(self.keep_indices[idx])
        imgs    = np.asarray(self.images[raw_idx])

        if imgs.ndim == 2:
            imgs = imgs[None, ..., None]
        elif imgs.ndim == 3:
            if imgs.shape[-1] in [1, 3]:
                imgs = imgs[None, ...]
            else:
                imgs = imgs[..., None]
        elif imgs.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported image ndim: {imgs.ndim}, shape={imgs.shape}")

        if imgs.shape[-1] == 1:
            imgs = np.repeat(imgs, 3, axis=-1)
        elif imgs.shape[-1] != 3:
            raise ValueError(f"Expected channel=1 or 3, got shape={imgs.shape}")

        if MAX_VIEWS is not None and len(imgs) > MAX_VIEWS:
            idxs = np.linspace(0, len(imgs) - 1, MAX_VIEWS).round().astype(int)
            imgs = imgs[idxs]

        return imgs

    def __getitem__(self, idx):
        imgs = self._get_views(idx)
        y    = int(self.remapped_subtypes[idx])

        view_tensors = []
        for view in imgs:
            x = ensure_float01(view)
            x = torch.from_numpy(x).float().permute(2, 0, 1).contiguous()
            view_tensors.append(x)

        x = torch.stack(view_tensors, dim=0)  # [S, 3, H, W]
        x = normalize_imagenet_tensor(x)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


# ============================================================
# 5) fold별 DataLoader 생성
# ============================================================
def make_fold_loaders(fold_idx):
    train_npz_path = DATA_ROOT / f"fold_{fold_idx}_train.npz"
    val_npz_path   = DATA_ROOT / f"fold_{fold_idx}_val.npz"

    assert train_npz_path.exists(), f"Missing: {train_npz_path}"
    assert val_npz_path.exists(),   f"Missing: {val_npz_path}"

    train_dataset = OvarianUSNPZDataset(train_npz_path)
    val_dataset   = OvarianUSNPZDataset(val_npz_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    subtype_np         = train_dataset.remapped_subtypes.astype(int)
    counts             = np.bincount(subtype_np, minlength=NUM_CLASSES).astype(np.float32)
    counts             = np.maximum(counts, 1.0)
    class_weight_value = 1.0 / np.sqrt(counts)
    class_weight_value = class_weight_value / class_weight_value.mean()
    class_weight       = torch.tensor(class_weight_value, device=device, dtype=torch.float32)

    print(f"[FOLD {fold_idx}] subtype counts={counts.tolist()}, class_weight={class_weight.cpu().numpy().round(4).tolist()}")
    print(f"[FOLD {fold_idx}] patient-level split assumed from fold_{fold_idx}_train/val.npz")

    return train_loader, val_loader, class_weight


# ============================================================
# 6) train / eval 함수
# ============================================================
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))


def train_one_epoch(model, loader, criterion, optimizer, device):
    prepare_train_mode(model)
    running_loss = 0.0
    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().ravel())
        y_pred.extend(preds.detach().cpu().numpy().ravel())

    metric_vals = compute_multiclass_metrics(y_true, y_pred)

    return {
        "loss":         running_loss / len(loader.dataset),
        "acc":          metric_vals["acc"],
        "macro_f1":     metric_vals["macro_f1"],
        "macro_recall": metric_vals["macro_recall"],
        "balanced_acc": metric_vals["balanced_acc"],
        "cm":           metric_vals["cm"],
    }


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)

        preds = torch.argmax(logits, dim=1)
        y_true.extend(labels.detach().cpu().numpy().ravel())
        y_pred.extend(preds.detach().cpu().numpy().ravel())

    metric_vals = compute_multiclass_metrics(y_true, y_pred)

    return {
        "loss":         running_loss / len(loader.dataset),
        "acc":          metric_vals["acc"],
        "macro_f1":     metric_vals["macro_f1"],
        "macro_recall": metric_vals["macro_recall"],
        "balanced_acc": metric_vals["balanced_acc"],
        "cm":           metric_vals["cm"],
    }


@torch.no_grad()
def eval_confusion(model, loader, device):
    model.eval()
    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        preds  = torch.argmax(logits, dim=1)

        y_true.extend(labels.cpu().numpy().ravel())
        y_pred.extend(preds.cpu().numpy().ravel())

    metric_vals = compute_multiclass_metrics(y_true, y_pred)

    print("\n-- Final Evaluation --")
    print(f"Accuracy      : {metric_vals['acc']:.4f}")
    print(f"Macro-F1      : {metric_vals['macro_f1']:.4f}")
    print(f"Macro-Recall  : {metric_vals['macro_recall']:.4f}")
    print(f"Balanced Acc  : {metric_vals['balanced_acc']:.4f}")
    print(metric_vals["cm"])
    print("\nClassification report:")
    print(classification_report(
        np.array(y_true).astype(int),
        np.array(y_pred).astype(int),
        target_names=CLASS_NAMES,
        digits=4,
        zero_division=0
    ))

    return metric_vals


# ============================================================
# 7) 모델 정의
# ============================================================
class PatientSliceAttentionClassifier(nn.Module):
    def __init__(self, encoder, feat_dim, hidden_dim=256, dropout=0.3, num_classes=NUM_CLASSES):
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
        x = x.view(B * S, C, H, W)

        feat = self.encoder(x)          # (B*S, feat_dim)
        feat = feat.view(B, S, -1)      # (B, S, feat_dim)

        attn_score  = self.attn(feat)           # (B, S, 1)
        attn_weight = torch.softmax(attn_score, dim=1)

        pooled = (feat * attn_weight).sum(dim=1)  # (B, feat_dim)
        logits = self.classifier(pooled)           # (B, num_classes)
        return logits


# ============================================================
# 8) build 함수들
# ============================================================

# ---------- ResNet18 ----------
def build_resnet18_patient_model(
    load_pretrained=True,
    freeze_backbone=True,
    unfreeze_layer4=True,
    weight_path=LOCAL_RESNET_WEIGHTS
):
    """
    ResNet50 → ResNet18 변경 포인트:
      - models.resnet18()
      - feat_dim: 512  (ResNet50은 2048)
      - hidden_dim: 256 으로 축소
    """
    encoder = models.resnet18(weights=None)

    if load_pretrained:
        state_dict = load_local_state_dict(weight_path)
        encoder.load_state_dict(state_dict, strict=True)

    feat_dim   = encoder.fc.in_features   # 512
    encoder.fc = nn.Identity()

    if freeze_backbone:
        for p in encoder.parameters():
            p.requires_grad = False
        if unfreeze_layer4:
            for p in encoder.layer4.parameters():
                p.requires_grad = True

    model = PatientSliceAttentionClassifier(
        encoder=encoder,
        feat_dim=feat_dim,
        hidden_dim=256,   # feat_dim=512 에 맞게 조정
        dropout=0.1,
        num_classes=NUM_CLASSES,
    )

    return model


# ---------- ViT-B/16 ----------
def build_vit_patient_model(
    load_pretrained=True,
    weight_path=LOCAL_VIT_WEIGHTS
):
    """ViT는 변경 없음 (feat_dim=768 유지)."""
    encoder = vit_b_16(weights=None)

    if load_pretrained:
        state_dict = load_local_state_dict(weight_path)
        encoder.load_state_dict(state_dict, strict=True)

    feat_dim      = encoder.heads.head.in_features   # 768
    encoder.heads = nn.Identity()

    model = PatientSliceAttentionClassifier(
        encoder=encoder,
        feat_dim=feat_dim,
        hidden_dim=512,
        dropout=0.3,
        num_classes=NUM_CLASSES,
    )
    return model


# ---------- ResNet18 + Transformer Hybrid ----------
class CNNTransformerHybridPatient(nn.Module):
    """
    ResNet50 → ResNet18 변경 포인트:
      - backbone: models.resnet18()
      - feat_dim 512 → d_model=256 으로 project
      - dim_feedforward=512 (d_model×2)
      - classifier hidden: 512 → 256
    """
    def __init__(
        self,
        d_model=256,
        nhead=8,
        num_layers=2,
        dim_feedforward=512,
        dropout=0.3,
        load_pretrained=True,
        freeze_backbone=True,
        unfreeze_layer4=True,
        resnet_weight_path=LOCAL_RESNET_WEIGHTS,
        num_classes=NUM_CLASSES,
    ):
        super().__init__()

        # ── backbone ──────────────────────────────────────────────
        backbone = models.resnet18(weights=None)
        if load_pretrained:
            state_dict = load_local_state_dict(resnet_weight_path)
            backbone.load_state_dict(state_dict, strict=True)

        feat_dim    = backbone.fc.in_features   # 512
        backbone.fc = nn.Identity()
        self.encoder = backbone

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_layer4:
                for p in self.encoder.layer4.parameters():
                    p.requires_grad = True

        # ── projection + Transformer ──────────────────────────────
        self.proj      = nn.Linear(feat_dim, d_model)   # 512 → 256
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 16, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

        # ── classifier ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feat = self.encoder(x)          # (B*S, 512)
        feat = feat.view(B, S, -1)      # (B, S, 512)
        feat = self.proj(feat)          # (B, S, d_model)

        cls    = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        tokens  = self.transformer(tokens)
        tokens  = self.norm(tokens)

        cls_out = tokens[:, 0, :]
        logits  = self.classifier(cls_out)
        return logits


def build_hybrid_patient_model(load_pretrained=True):
    return CNNTransformerHybridPatient(
        load_pretrained=load_pretrained,
        freeze_backbone=True,
        unfreeze_layer4=True,
        resnet_weight_path=LOCAL_RESNET_WEIGHTS,
        num_classes=NUM_CLASSES,
    )


# ============================================================
# 9) 공통 학습 함수
# ============================================================
def fit_model(
    model,
    model_name,
    fold_idx,
    train_loader,
    val_loader,
    device,
    num_epochs=30,
    lr=1e-4,
    weight_decay=1e-4,
    pos_weight=None,   # CT binary 코드와 이름 호환 유지. 실제로는 class_weight.
    patience=7,
    min_delta=0.001,
):
    model     = model.to(device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_score         = -np.inf
    best_path          = OUTPUT_DIR / f"{model_name}_fold{fold_idx}_best.pth"
    history            = []
    early_stop_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics   = eval_one_epoch(model, val_loader, criterion, device)

        row = {
            "fold":               fold_idx,
            "epoch":              epoch,
            "train_loss":         train_metrics["loss"],
            "train_acc":          train_metrics["acc"],
            "train_macro_f1":     train_metrics["macro_f1"],
            "train_macro_recall": train_metrics["macro_recall"],
            "train_balanced_acc": train_metrics["balanced_acc"],
            "val_loss":           val_metrics["loss"],
            "val_acc":            val_metrics["acc"],
            "val_macro_f1":       val_metrics["macro_f1"],
            "val_macro_recall":   val_metrics["macro_recall"],
            "val_balanced_acc":   val_metrics["balanced_acc"],
        }
        history.append(row)

        print(
            f"[{model_name}][Fold {fold_idx}][Epoch {epoch:02d}] "
            f"train loss={row['train_loss']:.4f}, acc={row['train_acc']:.4f}, "
            f"macro_f1={row['train_macro_f1']:.4f}, macro_recall={row['train_macro_recall']:.4f}, "
            f"balanced_acc={row['train_balanced_acc']:.4f} | "
            f"val loss={row['val_loss']:.4f}, acc={row['val_acc']:.4f}, "
            f"macro_f1={row['val_macro_f1']:.4f}, macro_recall={row['val_macro_recall']:.4f}, "
            f"balanced_acc={row['val_balanced_acc']:.4f}"
        )

        score = row["val_macro_f1"]
        if score > best_score + min_delta:
            best_score         = score
            early_stop_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> best saved by val_macro_f1: {best_path}")
        else:
            early_stop_counter += 1
            print(f"  -> no improvement ({early_stop_counter}/{patience})")

        if early_stop_counter >= patience:
            print(f"  -> early stopping triggered at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    return model, history_df, best_path


# ============================================================
# 10) 모델 로드 함수
# ============================================================
def load_model(model_type, model_path, device):
    if model_type == "resnet":
        model = build_resnet18_patient_model()   # resnet18로 변경
    elif model_type == "vit":
        model = build_vit_patient_model()
    elif model_type == "hybrid":
        model = build_hybrid_patient_model()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ============================================================
# 11) CV 실행 함수
# ============================================================
def run_cv_for_model(model_name, model_type, build_fn):
    fold_results   = []
    fold_histories = []

    for fold_idx in FOLDS:
        print("\n" + "=" * 80)
        print(f"Running {model_name} | Fold {fold_idx}")
        print("=" * 80)

        train_loader, val_loader, class_weight = make_fold_loaders(fold_idx)
        model = build_fn()

        _, history_df, best_path = fit_model(
            model=model,
            model_name=model_name,
            fold_idx=fold_idx,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=NUM_EPOCHS,
            lr=LR,
            weight_decay=WEIGHT_DECAY,
            pos_weight=class_weight,
            patience=PATIENCE,
        )

        best_model    = load_model(model_type, best_path, device)
        final_metrics = eval_confusion(best_model, val_loader, device)

        print(f"\n[{model_name}][Fold {fold_idx}] Final Metrics")
        print(f"Accuracy      : {final_metrics['acc']:.4f}")
        print(f"Macro-F1      : {final_metrics['macro_f1']:.4f}")
        print(f"Macro-Recall  : {final_metrics['macro_recall']:.4f}")
        print(f"Balanced Acc  : {final_metrics['balanced_acc']:.4f}")
        print("Confusion Matrix:")
        print(final_metrics["cm"])

        fold_results.append({
            "model":            model_name,
            "fold":             fold_idx,
            "acc":              final_metrics["acc"],
            "macro_f1":         final_metrics["macro_f1"],
            "macro_recall":     final_metrics["macro_recall"],
            "balanced_acc":     final_metrics["balanced_acc"],
            "Confusion Matrix": final_metrics["cm"],
        })

        history_df["model"] = model_name
        fold_histories.append(history_df)

        del model, best_model, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df      = pd.DataFrame(fold_results)
    history_df_all = pd.concat(fold_histories, ignore_index=True)
    return result_df, history_df_all


# ============================================================
# 12) 실행
# ============================================================
if __name__ == "__main__":

    # ── ResNet18 ──────────────────────────────────────────────
    print(">>> Training ResNet18...")
    resnet_cv_df, resnet_history_df = run_cv_for_model(
        model_name="ResNet18_US_Subtype",
        model_type="resnet",
        build_fn=lambda: build_resnet18_patient_model(load_pretrained=True),
    )
    print(resnet_cv_df)

    # ── ViT-B/16 ──────────────────────────────────────────────
    print(">>> Training ViT-B/16...")
    vit_cv_df, vit_history_df = run_cv_for_model(
        model_name="ViT-B16_US_Subtype",
        model_type="vit",
        build_fn=lambda: build_vit_patient_model(load_pretrained=True),
    )
    print(vit_cv_df)

    # ── Hybrid (ResNet18 + Transformer) ───────────────────────
    print(">>> Training Hybrid (ResNet18 + Transformer)...")
    hybrid_cv_df, hybrid_history_df = run_cv_for_model(
        model_name="Hybrid_US_Subtype",
        model_type="hybrid",
        build_fn=lambda: build_hybrid_patient_model(load_pretrained=True),
    )
    print(hybrid_cv_df)

    # ── 5-fold 평균 ± 표준편차 요약 ───────────────────────────
    all_cv_df  = pd.concat([resnet_cv_df, vit_cv_df, hybrid_cv_df], ignore_index=True)
    summary_df = (
        all_cv_df.groupby("model")[["acc", "macro_f1", "macro_recall", "balanced_acc"]]
        .agg(["mean", "std"])
    )
    print("\n===== 5-Fold Summary =====")
    print(summary_df)
