#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[2]:


# 1) 환경 설정 + import
import os
import gc
import random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, f1_score, roc_auc_score, recall_score
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


# In[3]:


print(f"현재 작업 경로: {os.getcwd()}")

test_relative = '/tf/nasw/dataset001/preprocessed/npz/fold_1_train.npz'
print(f"상대경로: {os.path.exists(test_relative)}")


# In[4]:


# 2. 경로 설정
DATA_ROOT = Path("/tf/nasw/dataset001/preprocessed/npz_v3_full_image")

FOLDS = [1, 2, 3, 4, 5]

OUTPUT_DIR = Path("./ct_binary_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = 224
BATCH_SIZE = 8
NUM_WORKERS = 2
NUM_EPOCHS = 30
LR = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE = 7

print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
print("FOLDS:", FOLDS)


# In[5]:


def inspect_fold_npz(fold_idx):
    train_npz_path = DATA_ROOT / f"fold_{fold_idx}_train.npz"
    val_npz_path   = DATA_ROOT / f"fold_{fold_idx}_val.npz"

    # ※ 원본 코드의 오타 수정: .exits() -> .exists()
    print("TRAIN:", train_npz_path, train_npz_path.exists())
    print("VAL:",   val_npz_path,   val_npz_path.exists())

    train_npz = np.load(train_npz_path, allow_pickle=False, mmap_mode="r")
    val_npz   = np.load(val_npz_path,   allow_pickle=False, mmap_mode="r")

    for k in train_npz.files:
        print(f"[train] {k}: shape={train_npz[k].shape}, dtype={train_npz[k].dtype}")
    for k in val_npz.files:
        print(f"[val]   {k}: shape={val_npz[k].shape}, dtype={val_npz[k].dtype}")


# In[6]:


# 로컬 pretrained weight 경로
# ResNet18 공식 weight 파일명으로 변경 (torchvision 기준)
LOCAL_RESNET_WEIGHTS = Path("/tf/models/pretrained/resnet18-f37072fd.pth")
LOCAL_VIT_WEIGHTS    = Path("/tf/models/pretrained/vit_b_16-c867db91.pth")

print("LOCAL_RESNET_WEIGHTS exists:", LOCAL_RESNET_WEIGHTS.exists())
print("LOCAL_VIT_WEIGHTS exists:",    LOCAL_VIT_WEIGHTS.exists())


# In[7]:


def load_local_state_dict(weight_path):
    weight_path = Path(weight_path)
    assert weight_path.exists(), f"Weight file not found: {weight_path}"

    ckpt = torch.load(weight_path, map_location="cpu")

    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    return ckpt


# In[8]:


# 4. Dataset 정의
class OvarianCTNPZDataset(Dataset):
    def __init__(self, npz_path):
        super().__init__()
        self.npz = np.load(npz_path, allow_pickle=False, mmap_mode="r")

        self.images = self.npz["images"]
        self.labels = self.npz["labels"].astype(np.float32)

        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        self.tumor_types = self.npz["tumor_types"] if "tumor_types" in self.npz.files else None
        self.recurrences = self.npz["recurrences"] if "recurrences" in self.npz.files else None

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.images[idx]
        y = self.labels[idx]

        x = torch.from_numpy(x).float().permute(0, 3, 1, 2).contiguous()

        x = x / 255.0
        x = (x - self.mean) / self.std

        y = torch.tensor([y], dtype=torch.float32)

        return x, y


# In[9]:


# fold별 DataLoader 생성
def make_fold_loaders(fold_idx):
    train_npz_path = DATA_ROOT / f"fold_{fold_idx}_train.npz"
    val_npz_path   = DATA_ROOT / f"fold_{fold_idx}_val.npz"

    assert train_npz_path.exists(), f"Missing: {train_npz_path}"
    assert val_npz_path.exists(),   f"Missing: {val_npz_path}"

    train_dataset = OvarianCTNPZDataset(train_npz_path)
    val_dataset   = OvarianCTNPZDataset(val_npz_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    labels_np = train_dataset.labels.astype(int)
    neg = (labels_np == 0).sum()
    pos = (labels_np == 1).sum()

    pos_weight_value = neg / max(pos, 1)
    pos_weight = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)

    print(f"[FOLD {fold_idx}] Train benign(0): {neg}, malignant(1): {pos}, pos_weight={pos_weight.item():.4f}")

    return train_loader, val_loader, pos_weight


# In[10]:


# 6. train / eval 함수

scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.array(y_true).astype(int)
    y_prob = np.array(y_prob).astype(float)
    y_pred = (y_prob > threshold).astype(int)

    metrics = {
        "acc":    accuracy_score(y_true, y_pred),
        "auc":    roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "f1":     f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "cm":     confusion_matrix(y_true, y_pred)
    }
    return metrics


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    y_true, y_prob = [], []

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

        probs = torch.sigmoid(logits)
        y_true.extend(labels.detach().cpu().numpy().ravel())
        y_prob.extend(probs.detach().cpu().numpy().ravel())

    metric_vals = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    metrics = {
        "loss":   running_loss / len(loader.dataset),
        "acc":    metric_vals["acc"],
        "auc":    metric_vals["auc"],
        "f1":     metric_vals["f1"],
        "recall": metric_vals["recall"],
        "cm":     metric_vals["cm"]
    }
    return metrics


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    y_true, y_prob = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            logits = model(imgs)
            loss   = criterion(logits, labels)

        running_loss += loss.item() * imgs.size(0)

        probs = torch.sigmoid(logits)
        y_true.extend(labels.detach().cpu().numpy().ravel())
        y_prob.extend(probs.detach().cpu().numpy().ravel())

    metric_vals = compute_binary_metrics(y_true, y_prob, threshold=0.5)

    metrics = {
        "loss":   running_loss / len(loader.dataset),
        "auc":    metric_vals["auc"],
        "acc":    metric_vals["acc"],
        "f1":     metric_vals["f1"],
        "recall": metric_vals["recall"],
        "cm":     metric_vals["cm"]
    }
    return metrics


@torch.no_grad()
def eval_confusion(model, loader, device, threshold=0.5):
    model.eval()
    y_true, y_prob = [], []

    for imgs, labels in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        probs  = torch.sigmoid(logits)

        y_true.extend(labels.cpu().numpy().ravel())
        y_prob.extend(probs.cpu().numpy().ravel())

    metric_vals = compute_binary_metrics(y_true, y_prob, threshold=threshold)

    # ※ 원본 코드의 오타 수정: \m -> \n
    print(f"\n-- Final Evaluation (threshold={threshold}) --")
    print(f"Accuracy: {metric_vals['acc']:.4f}")
    print(f"ROC-AUC : {metric_vals['auc']:.4f}")
    print(f"F1-score : {metric_vals['f1']:.4f}")
    print(f"Recall   : {metric_vals['recall']:.4f}")
    print(metric_vals["cm"])

    print("\nClassification report:")
    y_pred = (np.array(y_prob) > threshold).astype(int)
    print(classification_report(np.array(y_true).astype(int), y_pred, digits=4, zero_division=0))

    return metric_vals


# In[11]:


# Slice encoder + patient-level attention pooling wrapper
# (ResNet18 / ViT 모두 동일하게 사용)

class PatientSliceAttentionClassifier(nn.Module):
    def __init__(self, encoder, feat_dim, hidden_dim=256, dropout=0.3):
        super().__init__()
        self.encoder = encoder

        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1)
        )

        self.classifier = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feat = self.encoder(x)          # (B*S, feat_dim)
        feat = feat.view(B, S, -1)      # (B, S, feat_dim)

        attn_score  = self.attn(feat)           # (B, S, 1)
        attn_weight = torch.softmax(attn_score, dim=1)

        pooled = (feat * attn_weight).sum(dim=1)  # (B, feat_dim)
        logits = self.classifier(pooled)           # (B, 1)

        return logits


# In[12]:


# 7. 모델 정의

# ---------- ResNet18 ----------
def build_resnet18_patient_model(
    load_pretrained=True,
    freeze_backbone=True,
    unfreeze_layer4=True,
    weight_path=LOCAL_RESNET_WEIGHTS
):
    """
    ResNet50 → ResNet18 변경 포인트:
      - models.resnet18()  (layer 구성: layer1~layer4, BasicBlock 기반)
      - feat_dim: ResNet18의 fc.in_features = 512  (ResNet50은 2048)
      - hidden_dim을 256으로 축소 (파라미터 수 균형)
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
        hidden_dim=256,   # ResNet18 feat_dim=512 에 맞게 조정
        dropout=0.1
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
        dropout=0.3
    )
    return model


# ---------- ResNet18 + Transformer Hybrid ----------
class CNNTransformerHybridPatient(nn.Module):
    """
    ResNet50 → ResNet18 변경 포인트:
      - backbone: models.resnet18()
      - feat_dim 512  → d_model=256 으로 project (ResNet50 때는 2048→512)
      - nhead=8 유지하되 d_model=256이므로 head당 32-dim
      - dim_feedforward=512 (d_model×2)
      - pos_embed 슬라이스 수(8) 유지
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
        resnet_weight_path=LOCAL_RESNET_WEIGHTS
    ):
        super().__init__()

        # ── backbone ──────────────────────────────────────────────
        backbone = models.resnet18(weights=None)
        if load_pretrained:
            state_dict = load_local_state_dict(resnet_weight_path)
            backbone.load_state_dict(state_dict, strict=True)

        feat_dim   = backbone.fc.in_features   # 512
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
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

        # ── classifier ────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feat = self.encoder(x)          # (B*S, 512)
        feat = feat.view(B, S, -1)      # (B, S, 512)
        feat = self.proj(feat)          # (B, S, d_model)

        cls    = self.cls_token.expand(B, -1, -1)          # (B, 1, d_model)
        tokens = torch.cat([cls, feat], dim=1)              # (B, 1+S, d_model)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        tokens  = self.transformer(tokens)
        tokens  = self.norm(tokens)

        cls_out = tokens[:, 0, :]        # (B, d_model)
        logits  = self.classifier(cls_out)

        return logits


def build_hybrid_patient_model(load_pretrained=True):
    return CNNTransformerHybridPatient(
        load_pretrained=load_pretrained,
        freeze_backbone=True,
        unfreeze_layer4=True,
        resnet_weight_path=LOCAL_RESNET_WEIGHTS
    )


# In[13]:


# 8. 공통 학습 함수
def fit_model(
    model, model_name, fold_idx,
    train_loader, val_loader, device,
    num_epochs=30, lr=1e-4, weight_decay=1e-4,
    pos_weight=None, patience=7, min_delta=0.001
):
    model     = model.to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay
    )

    best_auc          = -1
    best_val_loss     = np.inf
    best_path         = OUTPUT_DIR / f"{model_name}_fold{fold_idx}_best.pth"
    history           = []
    early_stop_counter = 0

    for epoch in range(1, num_epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics   = eval_one_epoch(model, val_loader, criterion, device)

        row = {
            "fold":         fold_idx,
            "epoch":        epoch,
            "train_loss":   train_metrics["loss"],
            "train_acc":    train_metrics["acc"],
            "train_f1":     train_metrics["f1"],
            "train_recall": train_metrics["recall"],
            "train_auc":    train_metrics["auc"],
            "val_loss":     val_metrics["loss"],
            "val_acc":      val_metrics["acc"],
            "val_f1":       val_metrics["f1"],
            "val_auc":      val_metrics["auc"],
            "val_recall":   val_metrics["recall"],
        }
        history.append(row)

        print(
            f"[{model_name}][Fold {fold_idx}][Epoch {epoch:02d}] "
            f"train loss={row['train_loss']:.4f}, acc={row['train_acc']:.4f}, auc={row['train_auc']:.4f}, "
            f"f1={row['train_f1']:.4f}, recall={row['train_recall']:.4f} | "
            f"val loss={row['val_loss']:.4f}, acc={row['val_acc']:.4f}, auc={row['val_auc']:.4f}, "
            f"f1={row['val_f1']:.4f}, recall={row['val_recall']:.4f}"
        )

        if row["val_auc"] > best_auc:
            best_auc = row["val_auc"]

        if row["val_loss"] < best_val_loss - min_delta:
            best_val_loss      = row["val_loss"]
            early_stop_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> best saved by val_loss: {best_path}")
        else:
            early_stop_counter += 1
            print(f"  -> no improvement ({early_stop_counter}/{patience})")

        if early_stop_counter >= patience:
            print(f"  -> early stopping triggered at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    return model, history_df, best_path


# In[14]:


# 모델 로드 함수
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


# In[15]:


def run_cv_for_model(model_name, model_type, build_fn):
    fold_results  = []
    fold_histories = []

    for fold_idx in FOLDS:
        print("\n" + "=" * 80)
        print(f"Running {model_name} | Fold {fold_idx}")
        print("=" * 80)

        train_loader, val_loader, pos_weight = make_fold_loaders(fold_idx)

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
            pos_weight=pos_weight,
            patience=PATIENCE
        )

        best_model    = load_model(model_type, best_path, device)
        final_metrics = eval_confusion(best_model, val_loader, device, threshold=0.5)

        print(f"\n[{model_name}][Fold {fold_idx}] Final Metrics")
        print(f"Accuracy: {final_metrics['acc']:.4f}")
        print(f"ROC-AUC : {final_metrics['auc']:.4f}")
        print(f"F1      : {final_metrics['f1']:.4f}")
        print(f"Recall  : {final_metrics['recall']:.4f}")
        print("Confusion Matrix:")
        print(final_metrics["cm"])

        fold_results.append({
            "model":            model_name,
            "fold":             fold_idx,
            "acc":              final_metrics["acc"],
            "ROC-AUC":          final_metrics["auc"],
            "F1-score":         final_metrics["f1"],
            "Recall":           final_metrics["recall"],
            "Confusion Matrix": final_metrics["cm"]
        })

        history_df["model"] = model_name
        fold_histories.append(history_df)

        del model, best_model, train_loader, val_loader
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result_df       = pd.DataFrame(fold_results)
    history_df_all  = pd.concat(fold_histories, ignore_index=True)
    return result_df, history_df_all


# In[16]:


# 9. ResNet18 학습
print(">>> Training ResNet18...")
resnet_cv_df, resnet_history_df = run_cv_for_model(
    model_name="ResNet18",
    model_type="resnet",
    build_fn=lambda: build_resnet18_patient_model(load_pretrained=True)
)

display(resnet_cv_df)


# In[17]:


# 10. ViT-B/16 학습
print(">>> Training ViT-B/16...")
vit_cv_df, vit_history_df = run_cv_for_model(
    model_name="ViT-B16",
    model_type="vit",
    build_fn=lambda: build_vit_patient_model(load_pretrained=True)
)

display(vit_cv_df)


# In[18]:


# 11. Hybrid (ResNet18 + Transformer) 학습
print(">>> Training Hybrid (ResNet18 + Transformer)...")
hybrid_cv_df, hybrid_history_df = run_cv_for_model(
    model_name="Hybrid",
    model_type="hybrid",
    build_fn=lambda: build_hybrid_patient_model(load_pretrained=True)
)

display(hybrid_cv_df)


# In[19]:


# 13. 5-fold 평균 ± 표준편차 요약
all_cv_df = pd.concat([resnet_cv_df, vit_cv_df, hybrid_cv_df], ignore_index=True)

summary_df = (
    all_cv_df.groupby("model")[["ROC-AUC", "F1-score", "Recall"]]
    .agg(["mean", "std"])
)

display(summary_df)


# In[20]:


import matplotlib.pyplot as plt

def plot_one_model_by_fold(history_df, model_name):
    folds = sorted(history_df["fold"].unique())

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    for fold in folds:
        df_fold = history_df[history_df["fold"] == fold].sort_values("epoch")
        plt.plot(df_fold["epoch"], df_fold["train_loss"], linestyle="--", alpha=0.7, label=f"Fold{fold} Train")
        plt.plot(df_fold["epoch"], df_fold["val_loss"],   alpha=0.7,      label=f"Fold{fold} Val")
    plt.title(f"{model_name} - Loss by Fold")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(fontsize=8)

    plt.subplot(1, 2, 2)
    for fold in folds:
        df_fold = history_df[history_df["fold"] == fold].sort_values("epoch")
        plt.plot(df_fold["epoch"], df_fold["val_auc"], alpha=0.8, label=f"Fold{fold}")
    plt.title(f"{model_name} - Val ROC-AUC by Fold")
    plt.xlabel("Epoch")
    plt.ylabel("AUC")
    plt.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


# In[21]:


plot_one_model_by_fold(resnet_history_df, "ResNet18")
plot_one_model_by_fold(vit_history_df,    "ViT-B16")
plot_one_model_by_fold(hybrid_history_df, "Hybrid (ResNet18 + Transformer)")
