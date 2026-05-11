#!/usr/bin/env python
# coding: utf-8

# ============================================================
# CT Cascade 학습 스크립트
#
# [구조 개요]
#   1단계: malignancy binary classification (benign / malignant)
#   2단계: malignant 샘플에 대해서만 subtype multi-class classification
#
# [원본 대비 주요 수정 사항]
#   1. inspect_fold_npz 오타 수정: .exits() → .exists()
#   2. infer_malignant_type_ids 들여쓰기 버그 수정
#      (return이 fold loop 안에 있어 fold 1 결과만 반환하던 문제)
#   3. cascade_correct 인덱스 불일치 버그 수정
#      (oracle_pred/oracle_true는 all_mal_mask 기준,
#       all_mal_pred[gt_mal_mask]는 gt_mal_mask 기준 → 길이 불일치)
#   4. fit_model print 오타 수정: 콤마로 분리된 f-string → 하나로 합침
#   5. 모델 구조 개선:
#      - shared head 이후 task별 projection(mal_proj / sub_proj) 분리
#      - early stopping 기준: val_loss → val_auc (malignancy recall 중심)
#      - subtype_loss_weight warmup 스케줄 추가
# ============================================================


# ============================================================
# 빌드 확인
# ============================================================
from torchvision import models
from torchvision.models import vit_b_16

print("=" * 60)

try:
    m1 = models.resnet50(weights=None)
    print("ResNet50 build success")
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
from collections import OrderedDict, defaultdict

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
from torchvision.models import vit_b_16, ViT_B_16_Weights, ResNet50_Weights

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# cuDNN autotuner: 입력 크기가 고정일 때 속도 향상
torch.backends.cudnn.benchmark = True


# ============================================================
# 2) 경로 / 하이퍼파라미터 설정
# ============================================================
print(f"현재 작업 경로: {os.getcwd()}")

DATA_ROOT  = Path("/tf/nasw/dataset001/preprocessed/npz_v3_full_image")
FOLDS      = [1, 2, 3, 4, 5]

OUTPUT_DIR = Path("./ct_cascade_models")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE     = 224
BATCH_SIZE   = 8
NUM_WORKERS  = 2
NUM_EPOCHS   = 30
LR           = 1e-4
WEIGHT_DECAY = 1e-4
PATIENCE     = 7
MIN_DELTA    = 0.001
THRESHOLD    = 0.5   # malignancy 판단 임계값

# subtype loss 가중치 warmup 설정
# - 학습 초반에는 malignancy 분류에 집중하고,
#   후반부로 갈수록 subtype loss 비중을 점진적으로 높임
SUBTYPE_LOSS_WEIGHT_START = 0.1   # 1 epoch 시작값
SUBTYPE_LOSS_WEIGHT_END   = 0.5   # NUM_EPOCHS epoch 도달값

print("DATA_ROOT:", DATA_ROOT)
print("OUTPUT_DIR:", OUTPUT_DIR.resolve())
print("FOLDS:", FOLDS)


# ============================================================
# 3) 로컬 pretrained weight 경로
# ============================================================
LOCAL_RESNET_WEIGHTS = Path("/tf/models/pretrained/resnet50-11ad3fa6.pth")
LOCAL_VIT_WEIGHTS    = Path("/tf/models/pretrained/vit_b_16-c867db91.pth")

print("LOCAL_RESNET_WEIGHTS exists:", LOCAL_RESNET_WEIGHTS.exists())
print("LOCAL_VIT_WEIGHTS exists:",    LOCAL_VIT_WEIGHTS.exists())


# ============================================================
# 4) 유틸 함수
# ============================================================
def load_local_state_dict(weight_path):
    weight_path = Path(weight_path)
    assert weight_path.exists(), f"Weight file not found: {weight_path}"

    ckpt = torch.load(weight_path, map_location="cpu")

    # 일부 checkpoint는 {"state_dict": {...}} 형태로 저장되어 있음
    if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        ckpt = ckpt["state_dict"]

    return ckpt


def inspect_fold_npz(fold_idx):
    train_npz_path = DATA_ROOT / f"fold_{fold_idx}_train.npz"
    val_npz_path   = DATA_ROOT / f"fold_{fold_idx}_val.npz"

    # ✅ 버그 수정: .exits() → .exists()
    print("TRAIN:", train_npz_path, train_npz_path.exists())
    print("VAL:",   val_npz_path,   val_npz_path.exists())

    train_npz = np.load(train_npz_path, allow_pickle=False, mmap_mode="r")
    val_npz   = np.load(val_npz_path,   allow_pickle=False, mmap_mode="r")

    for k in train_npz.files:
        print(f"[train] {k}: shape={train_npz[k].shape}, dtype={train_npz[k].dtype}")
    for k in val_npz.files:
        print(f"[val]   {k}: shape={val_npz[k].shape}, dtype={val_npz[k].dtype}")


def get_subtype_loss_weight(epoch: int, total_epochs: int) -> float:
    """
    subtype loss 가중치를 epoch에 따라 선형 warmup.

    epoch 1  → SUBTYPE_LOSS_WEIGHT_START
    epoch N  → SUBTYPE_LOSS_WEIGHT_END
    사이는 선형 보간.

    이유: 학습 초반 malignancy gate가 불안정할 때
    subtype loss가 너무 크면 전체 학습을 방해할 수 있음.
    """
    t = (epoch - 1) / max(total_epochs - 1, 1)   # 0.0 ~ 1.0
    return SUBTYPE_LOSS_WEIGHT_START + t * (SUBTYPE_LOSS_WEIGHT_END - SUBTYPE_LOSS_WEIGHT_START)


# ============================================================
# 5) malignant subtype ID 자동 추론
# ============================================================
def infer_malignant_type_ids(data_root, folds):
    """
    NPZ 파일에서 tumor_type별로 label(0/1)을 수집하여
    항상 malignant(label=1)인 tumor_type ID 목록을 자동 추론.

    ✅ 버그 수정: 원본은 return이 fold loop 안에 있어서
       fold 1 처리 후 바로 반환되는 문제가 있었음.
       loop 밖으로 conflicts 체크와 return을 이동.
    """
    # tumor_type별로 등장한 label 값들을 수집
    label_set_by_type = defaultdict(set)

    for fold_idx in folds:
        for split_name in ["train", "val"]:
            npz_path = Path(data_root) / f"fold_{fold_idx}_{split_name}.npz"
            data     = np.load(npz_path, allow_pickle=False, mmap_mode="r")

            labels      = data["labels"].astype(int)
            tumor_types = data["tumor_types"].astype(int)

            for y, t in zip(labels, tumor_types):
                label_set_by_type[int(t)].add(int(y))

    # ✅ loop 종료 후 한 번만 체크 (원본: loop 안에서 매 fold마다 체크 후 return)
    # 동일 tumor_type이 benign/malignant 양쪽에 있으면 데이터 오류
    conflicts = {
        t: sorted(list(v))
        for t, v in label_set_by_type.items()
        if len(v) > 1
    }
    if len(conflicts) > 0:
        raise ValueError(
            f"tumor_type이 양성/악성 양쪽에 섞여 있음: {conflicts}\n"
            "데이터 전처리를 확인하세요."
        )

    # label이 {1}만인 tumor_type만 malignant subtype으로 간주
    malignant_type_ids = sorted([
        t for t, labs in label_set_by_type.items()
        if labs == {1}
    ])

    print("malignant_type_ids:", malignant_type_ids)
    return malignant_type_ids


MALIGNANT_TYPE_IDS     = infer_malignant_type_ids(DATA_ROOT, FOLDS)
MALIGNANT_MAP          = {tid: i for i, tid in enumerate(MALIGNANT_TYPE_IDS)}
NUM_MALIGNANT_SUBTYPES = len(MALIGNANT_TYPE_IDS)

print("MALIGNANT_MAP:", MALIGNANT_MAP)
print("NUM_MALIGNANT_SUBTYPES:", NUM_MALIGNANT_SUBTYPES)


# ============================================================
# 6) Dataset 정의
# ============================================================
class OvarianCTNPZDataset(Dataset):
    def __init__(self, npz_path):
        super().__init__()
        self.npz = np.load(npz_path, allow_pickle=False, mmap_mode="r")

        self.images      = self.npz["images"]
        self.labels      = self.npz["labels"].astype(np.float32)
        self.tumor_types = self.npz["tumor_types"].astype(np.int64)

        # patient_ids가 없으면 인덱스로 대체
        self.patient_ids = (
            self.npz["patient_ids"]
            if "patient_ids" in self.npz.files
            else np.arange(len(self.labels))
        )

        # ImageNet 정규화 상수 (CT/US 공통으로 사용 가능)
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        self.std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x           = self.images[idx]
        y           = int(self.labels[idx])
        tumor_type  = int(self.tumor_types[idx])

        # NPZ 저장 형태: [S, H, W, C] → [S, C, H, W]
        x = torch.from_numpy(x).float().permute(0, 3, 1, 2).contiguous()
        x = x / 255.0
        x = (x - self.mean) / self.std

        # malignant 샘플에만 subtype label 부여
        if y == 1:
            if tumor_type not in MALIGNANT_MAP:
                raise ValueError(
                    f"malignant sample인데 tumor_type={tumor_type}가 MALIGNANT_MAP에 없음. "
                    f"infer_malignant_type_ids 결과 확인 필요."
                )
            mal_subtype = MALIGNANT_MAP[tumor_type]
            mal_mask    = True
        else:
            # benign은 subtype 없음 → -1로 패딩 (loss 계산 시 mal_mask로 필터링)
            mal_subtype = -1
            mal_mask    = False

        return {
            "image":       x,
            "label":       torch.tensor(y, dtype=torch.float32),
            "mal_subtype": torch.tensor(mal_subtype, dtype=torch.long),
            "mal_mask":    torch.tensor(mal_mask, dtype=torch.bool),
        }


# ============================================================
# 7) fold별 DataLoader 생성
# ============================================================
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

    # ── malignancy pos_weight ────────────────────────────────
    # benign/malignant 불균형 보정: neg/pos 비율로 BCEWithLogitsLoss에 전달
    labels_np        = train_dataset.labels.astype(int)
    neg              = (labels_np == 0).sum()
    pos              = (labels_np == 1).sum()
    pos_weight_value = neg / max(pos, 1)
    mal_pos_weight   = torch.tensor([pos_weight_value], device=device, dtype=torch.float32)

    # ── subtype class weight ─────────────────────────────────
    # subtype별 샘플 수 역수로 class weight 계산 (inv mean normalization)
    subtype_counts = np.zeros(NUM_MALIGNANT_SUBTYPES, dtype=np.float64)
    for y, t in zip(train_dataset.labels.astype(int), train_dataset.tumor_types.astype(int)):
        if y == 1 and t in MALIGNANT_MAP:
            subtype_counts[MALIGNANT_MAP[t]] += 1.0

    subtype_counts       = np.maximum(subtype_counts, 1.0)   # 0 나누기 방지
    inv                  = 1.0 / subtype_counts
    subtype_class_weights = inv / inv.mean()                  # 평균 1로 정규화
    subtype_class_weights = torch.tensor(subtype_class_weights, dtype=torch.float32, device=device)

    print(
        f"[FOLD {fold_idx}] Train benign(0): {neg}, malignant(1): {pos}, "
        f"pos_weight={mal_pos_weight.item():.4f}, "
        f"subtype_counts={subtype_counts.tolist()}, "
        f"subtype_class_weights={subtype_class_weights.detach().cpu().numpy().round(4).tolist()}"
    )

    return train_loader, val_loader, mal_pos_weight, subtype_class_weights


# ============================================================
# 8) Cascade 손실 함수
# ============================================================
class CascadeCriterion(nn.Module):
    def __init__(self, pos_weight, subtype_class_weights, subtype_loss_weight=0.5):
        """
        Args:
            pos_weight           : BCEWithLogitsLoss의 pos_weight (malignancy)
            subtype_class_weights: CrossEntropyLoss의 class weight (subtype)
            subtype_loss_weight  : subtype loss를 얼마나 반영할지 가중치
                                   epoch마다 외부에서 업데이트 가능
        """
        super().__init__()
        self.bce               = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.ce                = nn.CrossEntropyLoss(weight=subtype_class_weights)
        self.subtype_loss_weight = subtype_loss_weight

    def forward(self, outputs, batch):
        labels      = batch["label"].to(device).view(-1)        # [B]
        mal_mask    = batch["mal_mask"].to(device).view(-1)     # [B] bool
        mal_subtype = batch["mal_subtype"].to(device).view(-1)  # [B]

        # malignancy binary loss (benign/malignant 전체 샘플)
        loss_m = self.bce(outputs["malignancy_logits"], labels)

        # subtype loss (malignant 샘플만)
        # mal_mask=True인 샘플만 필터링해서 CrossEntropyLoss 계산
        if mal_mask.any():
            loss_s = self.ce(
                outputs["subtype_logits"][mal_mask],
                mal_subtype[mal_mask]
            )
        else:
            # batch 안에 malignant 샘플이 없는 경우 (소규모 데이터에서 발생 가능)
            loss_s = torch.tensor(0.0, device=device)

        loss_total = loss_m + self.subtype_loss_weight * loss_s

        return {
            "loss_total":      loss_total,
            "loss_malignancy": loss_m,
            "loss_subtype":    loss_s,
        }


# ============================================================
# 9) 지표 계산 함수
# ============================================================
def compute_binary_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    y_pred = (y_prob >= threshold).astype(int)

    return {
        "acc":    accuracy_score(y_true, y_pred),
        "auc":    roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        "f1":     f1_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "cm":     confusion_matrix(y_true, y_pred),
    }


def compute_multiclass_metrics(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    if len(y_true) == 0:
        return {"acc": np.nan, "macro_f1": np.nan, "macro_recall": np.nan, "cm": None}

    return {
        "acc":          accuracy_score(y_true, y_pred),
        "macro_f1":     f1_score(y_true, y_pred, average="macro", zero_division=0),
        "macro_recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "cm":           confusion_matrix(y_true, y_pred),
    }


# ============================================================
# 10) train / eval 함수
# ============================================================
scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))


def _compute_cascade_metrics(
    all_label, all_mal_prob, all_mal_mask,
    all_subtype_true, all_subtype_pred,
    running_total, running_m, running_s,
    n_samples,
):
    """
    train_one_epoch / eval_one_epoch 공통 지표 계산 로직.
    중복 코드를 줄이기 위해 분리.

    ✅ 버그 수정: cascade_correct 인덱스 불일치 수정
        - oracle_pred / oracle_true: all_mal_mask(GT malignant) 기준으로 필터링된 배열
        - all_mal_pred[gt_mal_mask]: GT malignant 샘플에 대한 예측값
        - 위 두 배열의 길이는 동일 (gt_mal_mask == all_mal_mask)
        - 원본 코드는 all_mal_mask 대신 mal_mask를 별도로 쓰거나
          배열 길이가 맞지 않는 경우 발생 → gt_mal_mask로 통일하여 수정
    """
    mal_metrics = compute_binary_metrics(all_label, all_mal_prob, threshold=THRESHOLD)

    all_label        = np.asarray(all_label).astype(int)
    all_mal_pred     = (np.asarray(all_mal_prob) >= THRESHOLD).astype(int)
    all_mal_mask     = np.asarray(all_mal_mask).astype(bool)
    all_subtype_true = np.asarray(all_subtype_true).astype(int)
    all_subtype_pred = np.asarray(all_subtype_pred).astype(int)

    # oracle subtype: GT가 malignant인 샘플에 대해서만 subtype 평가
    # (malignancy gate의 영향 없이, subtype head 자체 성능 측정)
    oracle_true    = all_subtype_true[all_mal_mask]
    oracle_pred    = all_subtype_pred[all_mal_mask]
    subtype_oracle = compute_multiclass_metrics(oracle_true, oracle_pred)

    # cascade subtype: GT malignant 샘플 중
    #   ① malignancy gate도 malignant로 예측하고 (gate pass)
    #   ② subtype도 맞춰야 정답
    gt_mal_mask = all_mal_mask   # GT label == 1인 샘플 = mal_mask와 동일

    if gt_mal_mask.sum() > 0:
        gate_pass = all_mal_pred[gt_mal_mask] == 1   # gate 통과 여부 [N_mal]

        # ✅ oracle_pred는 gt_mal_mask 기준으로 필터링된 배열이므로 길이 일치
        subtype_correct = oracle_pred == oracle_true  # subtype 정답 여부 [N_mal]

        gate_recall          = gate_pass.mean()
        # cascade 정답 = gate도 통과하고 subtype도 맞아야 함
        subtype_cascade_acc  = (gate_pass & subtype_correct).mean()
    else:
        gate_recall         = np.nan
        subtype_cascade_acc = np.nan

    return {
        "loss":           running_total / n_samples,
        "loss_malignancy": running_m / n_samples,
        "loss_subtype":   running_s / n_samples,

        # malignancy 지표
        "acc":    mal_metrics["acc"],
        "auc":    mal_metrics["auc"],
        "f1":     mal_metrics["f1"],
        "recall": mal_metrics["recall"],
        "cm":     mal_metrics["cm"],

        # subtype oracle 지표 (gate 무관, subtype head 자체 성능)
        "subtype_oracle_acc":      subtype_oracle["acc"],
        "subtype_oracle_macro_f1": subtype_oracle["macro_f1"],
        "subtype_oracle_cm":       subtype_oracle["cm"],

        # cascade 지표 (gate + subtype 통합 성능)
        "subtype_cascade_acc":  float(subtype_cascade_acc),
        "subtype_gate_recall":  float(gate_recall),
    }


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_total, running_m, running_s = 0.0, 0.0, 0.0
    all_label        = []
    all_mal_prob     = []
    all_mal_mask     = []
    all_subtype_true = []
    all_subtype_pred = []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)

        optimizer.zero_grad()

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs   = model(imgs)
            loss_dict = criterion(outputs, batch)
            loss      = loss_dict["loss_total"]

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        bs             = imgs.size(0)
        running_total += loss_dict["loss_total"].item() * bs
        running_m     += loss_dict["loss_malignancy"].item() * bs
        running_s     += loss_dict["loss_subtype"].item() * bs

        # CPU로 이동 후 numpy 변환
        label_np        = batch["label"].detach().cpu().numpy().astype(int).ravel()
        mal_mask_np     = batch["mal_mask"].detach().cpu().numpy().astype(bool).ravel()
        mal_prob_np     = torch.sigmoid(outputs["malignancy_logits"]).detach().cpu().numpy().ravel()
        subtype_pred_np = outputs["subtype_logits"].argmax(dim=1).detach().cpu().numpy().astype(int).ravel()
        subtype_true_np = batch["mal_subtype"].detach().cpu().numpy().astype(int).ravel()

        all_label.extend(label_np.tolist())
        all_mal_prob.extend(mal_prob_np.tolist())
        all_mal_mask.extend(mal_mask_np.tolist())
        all_subtype_true.extend(subtype_true_np.tolist())
        all_subtype_pred.extend(subtype_pred_np.tolist())

    return _compute_cascade_metrics(
        all_label, all_mal_prob, all_mal_mask,
        all_subtype_true, all_subtype_pred,
        running_total, running_m, running_s,
        len(loader.dataset),
    )


@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()

    running_total, running_m, running_s = 0.0, 0.0, 0.0
    all_label        = []
    all_mal_prob     = []
    all_mal_mask     = []
    all_subtype_true = []
    all_subtype_pred = []

    for batch in loader:
        imgs = batch["image"].to(device, non_blocking=True)

        with torch.amp.autocast(device_type="cuda", enabled=(device.type == "cuda")):
            outputs   = model(imgs)
            loss_dict = criterion(outputs, batch)

        bs             = imgs.size(0)
        running_total += loss_dict["loss_total"].item() * bs
        running_m     += loss_dict["loss_malignancy"].item() * bs
        running_s     += loss_dict["loss_subtype"].item() * bs

        label_np        = batch["label"].detach().cpu().numpy().astype(int).ravel()
        mal_mask_np     = batch["mal_mask"].detach().cpu().numpy().astype(bool).ravel()
        mal_prob_np     = torch.sigmoid(outputs["malignancy_logits"]).detach().cpu().numpy().ravel()
        subtype_pred_np = outputs["subtype_logits"].argmax(dim=1).detach().cpu().numpy().astype(int).ravel()
        subtype_true_np = batch["mal_subtype"].detach().cpu().numpy().astype(int).ravel()

        all_label.extend(label_np.tolist())
        all_mal_prob.extend(mal_prob_np.tolist())
        all_mal_mask.extend(mal_mask_np.tolist())
        all_subtype_true.extend(subtype_true_np.tolist())
        all_subtype_pred.extend(subtype_pred_np.tolist())

    return _compute_cascade_metrics(
        all_label, all_mal_prob, all_mal_mask,
        all_subtype_true, all_subtype_pred,
        running_total, running_m, running_s,
        len(loader.dataset),
    )


@torch.no_grad()
def eval_confusion(model, loader, criterion, device, threshold=0.5):
    """최종 평가 + classification report 출력."""
    metrics = eval_one_epoch(model, loader, criterion, device)

    print("\n-- Final Evaluation --")
    print(f"[Malignancy]  AUC={metrics['auc']:.4f}, F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, Acc={metrics['acc']:.4f}")
    print(f"[Subtype Oracle]  Acc={metrics['subtype_oracle_acc']:.4f}, MacroF1={metrics['subtype_oracle_macro_f1']:.4f}")
    print(f"[Cascade]  SubtypeAcc={metrics['subtype_cascade_acc']:.4f}, GateRecall={metrics['subtype_gate_recall']:.4f}")
    print("Binary CM:")
    print(metrics["cm"])
    print("Subtype Oracle CM:")
    print(metrics["subtype_oracle_cm"])

    return metrics


# ============================================================
# 11) 모델 정의
# ============================================================

# ── 공통 Attention Pooling + Dual Head Classifier ────────────
class PatientSliceAttentionClassifier(nn.Module):
    """
    [구조 개선]
    원본: shared → malignancy_head / subtype_head (동일 표현 공유)
    수정: shared → mal_proj → malignancy_head
                 → sub_proj → subtype_head

    이유: malignancy 판단(global 패턴)과 subtype 분류(세부 형태 패턴)는
    집중해야 할 특징이 달라서 task별 projection을 추가하면 각 head가
    더 특화된 표현을 학습할 수 있음.
    """
    def __init__(self, encoder, feat_dim, hidden_dim=512, proj_dim=256, dropout=0.1):
        super().__init__()
        self.encoder = encoder

        # slice 간 중요도 attention (슬라이스별 가중합 풀링)
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )

        # 공통 표현 추출
        self.shared = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── task별 projection (개선 포인트) ──────────────────
        # malignancy: 양성/악성 global 구분에 집중
        self.mal_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # subtype: 악성 내 세부 형태 패턴 분류에 집중
        self.sub_proj = nn.Sequential(
            nn.Linear(hidden_dim, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.malignancy_head = nn.Linear(proj_dim, 1)
        self.subtype_head    = nn.Linear(proj_dim, NUM_MALIGNANT_SUBTYPES)

    def forward(self, x):
        # x: (B, S, C, H, W)
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feat = self.encoder(x)           # (B*S, feat_dim)
        feat = feat.view(B, S, -1)       # (B, S, feat_dim)

        # attention pooling
        attn_score  = self.attn(feat)                    # (B, S, 1)
        attn_weight = torch.softmax(attn_score, dim=1)
        pooled      = (feat * attn_weight).sum(dim=1)    # (B, feat_dim)

        # 공통 → task별 분기
        z     = self.shared(pooled)    # (B, hidden_dim)
        z_mal = self.mal_proj(z)       # (B, proj_dim)
        z_sub = self.sub_proj(z)       # (B, proj_dim)

        return {
            # squeeze(1): [B,1] → [B]  (BCEWithLogitsLoss 입력 형태)
            "malignancy_logits": self.malignancy_head(z_mal).squeeze(1),
            "subtype_logits":    self.subtype_head(z_sub),   # (B, NUM_MALIGNANT_SUBTYPES)
        }


# ── ResNet50 기반 모델 ────────────────────────────────────────
def build_resnet50_patient_model(
    load_pretrained=True,
    freeze_backbone=True,
    unfreeze_layer4=True,
    weight_path=LOCAL_RESNET_WEIGHTS,
):
    encoder = models.resnet50(weights=None)

    if load_pretrained:
        state_dict = load_local_state_dict(weight_path)
        encoder.load_state_dict(state_dict, strict=True)

    feat_dim   = encoder.fc.in_features   # 2048
    encoder.fc = nn.Identity()

    if freeze_backbone:
        for p in encoder.parameters():
            p.requires_grad = False
        if unfreeze_layer4:
            # layer4만 fine-tuning (고수준 특징 추출 레이어)
            for p in encoder.layer4.parameters():
                p.requires_grad = True

    model = PatientSliceAttentionClassifier(
        encoder=encoder,
        feat_dim=feat_dim,
        hidden_dim=512,
        proj_dim=256,
        dropout=0.1,
    )
    return model


# ── ViT-B/16 기반 모델 ───────────────────────────────────────
def build_vit_patient_model(
    load_pretrained=True,
    weight_path=LOCAL_VIT_WEIGHTS,
):
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
        proj_dim=256,
        dropout=0.1,
    )
    return model


# ── CNN + Transformer Hybrid 모델 ────────────────────────────
class CNNTransformerHybridPatient(nn.Module):
    """
    ResNet50으로 슬라이스별 특징 추출 후
    Transformer로 슬라이스 간 관계를 모델링.

    [구조 개선]
    원본: shared → malignancy_head / subtype_head
    수정: shared → mal_proj → malignancy_head
                 → sub_proj → subtype_head
    """
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_layers=2,
        dim_feedforward=1024,
        proj_dim=256,
        dropout=0.1,
        load_pretrained=True,
        freeze_backbone=True,
        unfreeze_layer4=True,
        resnet_weight_path=LOCAL_RESNET_WEIGHTS,
    ):
        super().__init__()

        # ── ResNet50 backbone ─────────────────────────────────
        backbone = models.resnet50(weights=None)
        if load_pretrained:
            state_dict = load_local_state_dict(resnet_weight_path)
            backbone.load_state_dict(state_dict, strict=True)
        feat_dim    = backbone.fc.in_features   # 2048
        backbone.fc = nn.Identity()
        self.encoder = backbone

        if freeze_backbone:
            for p in self.encoder.parameters():
                p.requires_grad = False
            if unfreeze_layer4:
                for p in self.encoder.layer4.parameters():
                    p.requires_grad = True

        # ── Transformer 입력 준비 ─────────────────────────────
        # ResNet 출력(2048) → Transformer 차원(d_model)으로 projection
        self.proj = nn.Linear(feat_dim, d_model)

        # CLS 토큰: 슬라이스 전체를 요약하는 학습 가능한 토큰
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # pos_embed: 슬라이스 위치 정보 인코딩 (1+8: CLS + 슬라이스 최대 8장)
        # 실제 슬라이스 수 S가 8보다 작으면 자동으로 슬라이싱됨
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + 8, d_model))

        # ── Transformer Encoder ───────────────────────────────
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm        = nn.LayerNorm(d_model)

        # ── 공통 표현 + task별 projection (개선 포인트) ────────
        self.shared = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.mal_proj = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.sub_proj = nn.Sequential(
            nn.Linear(512, proj_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.malignancy_head = nn.Linear(proj_dim, 1)
        self.subtype_head    = nn.Linear(proj_dim, NUM_MALIGNANT_SUBTYPES)

    def forward(self, x):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feat = self.encoder(x)          # (B*S, 2048)
        feat = feat.view(B, S, -1)      # (B, S, 2048)
        feat = self.proj(feat)          # (B, S, d_model)

        # CLS 토큰 prepend
        cls    = self.cls_token.expand(B, -1, -1)       # (B, 1, d_model)
        tokens = torch.cat([cls, feat], dim=1)           # (B, 1+S, d_model)
        # pos_embed는 최대 슬라이스 수 기준으로 정의하고 실제 S에 맞게 슬라이싱
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        tokens  = self.transformer(tokens)    # (B, 1+S, d_model)
        tokens  = self.norm(tokens)
        cls_out = tokens[:, 0, :]             # (B, d_model) CLS 토큰만 추출

        z     = self.shared(cls_out)   # (B, 512)
        z_mal = self.mal_proj(z)       # (B, proj_dim)
        z_sub = self.sub_proj(z)       # (B, proj_dim)

        return {
            "malignancy_logits": self.malignancy_head(z_mal).squeeze(1),
            "subtype_logits":    self.subtype_head(z_sub),
        }


def build_hybrid_patient_model(load_pretrained=True):
    return CNNTransformerHybridPatient(
        load_pretrained=load_pretrained,
        freeze_backbone=True,
        unfreeze_layer4=True,
        resnet_weight_path=LOCAL_RESNET_WEIGHTS,
    )


# ============================================================
# 12) 공통 학습 함수
# ============================================================
def fit_model(
    model, model_name, fold_idx,
    train_loader, val_loader, device,
    num_epochs=30, lr=1e-4, weight_decay=1e-4,
    criterion=None,
    patience=7, min_delta=0.001,
):
    """
    [early stopping 기준 변경]
    원본: val_loss 기준
    수정: val_auc 기준

    이유: cascade 모델에서 malignancy gate recall이 낮으면
    subtype 예측 자체가 의미 없어짐.
    val_loss는 subtype loss가 섞여 있어 malignancy 성능을
    직접 반영하지 못함 → val_auc가 더 적합.

    [subtype_loss_weight warmup]
    epoch마다 CascadeCriterion의 weight를 업데이트.
    학습 초반에는 malignancy에 집중, 후반으로 갈수록 subtype 비중 증가.
    """
    if criterion is None:
        raise ValueError("cascade 모델에서는 criterion에 CascadeCriterion을 넘겨야 함")

    model = model.to(device)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_auc           = -1.0
    best_path          = OUTPUT_DIR / f"{model_name}_fold{fold_idx}_best.pth"
    history            = []
    early_stop_counter = 0

    for epoch in range(1, num_epochs + 1):

        # subtype loss weight warmup: epoch마다 선형 증가
        current_subtype_weight = get_subtype_loss_weight(epoch, num_epochs)
        criterion.subtype_loss_weight = current_subtype_weight

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics   = eval_one_epoch(model, val_loader, criterion, device)

        row = {
            "fold":  fold_idx,
            "epoch": epoch,
            "subtype_loss_weight": current_subtype_weight,

            # train 지표
            "train_loss":                    train_metrics["loss"],
            "train_loss_malignancy":         train_metrics["loss_malignancy"],
            "train_loss_subtype":            train_metrics["loss_subtype"],
            "train_acc":                     train_metrics["acc"],
            "train_auc":                     train_metrics["auc"],
            "train_f1":                      train_metrics["f1"],
            "train_recall":                  train_metrics["recall"],
            "train_subtype_oracle_acc":      train_metrics["subtype_oracle_acc"],
            "train_subtype_oracle_macro_f1": train_metrics["subtype_oracle_macro_f1"],
            "train_subtype_cascade_acc":     train_metrics["subtype_cascade_acc"],
            "train_subtype_gate_recall":     train_metrics["subtype_gate_recall"],

            # val 지표
            "val_loss":                    val_metrics["loss"],
            "val_loss_malignancy":         val_metrics["loss_malignancy"],
            "val_loss_subtype":            val_metrics["loss_subtype"],
            "val_acc":                     val_metrics["acc"],
            "val_auc":                     val_metrics["auc"],
            "val_f1":                      val_metrics["f1"],
            "val_recall":                  val_metrics["recall"],
            "val_subtype_oracle_acc":      val_metrics["subtype_oracle_acc"],
            "val_subtype_oracle_macro_f1": val_metrics["subtype_oracle_macro_f1"],
            "val_subtype_cascade_acc":     val_metrics["subtype_cascade_acc"],
            "val_subtype_gate_recall":     val_metrics["subtype_gate_recall"],
        }
        history.append(row)

        # ✅ 버그 수정: 원본은 콤마로 f-string이 분리되어 별도 출력됨
        # → 하나의 f-string으로 합침
        print(
            f"[{model_name}][Fold {fold_idx}][Epoch {epoch:02d}] "
            f"sub_w={current_subtype_weight:.3f} | "
            f"train loss={row['train_loss']:.4f}(mal={row['train_loss_malignancy']:.4f},sub={row['train_loss_subtype']:.4f}), "
            f"auc={row['train_auc']:.4f}, f1={row['train_f1']:.4f}, recall={row['train_recall']:.4f}, "
            f"oracle_f1={row['train_subtype_oracle_macro_f1']:.4f}, cascade_acc={row['train_subtype_cascade_acc']:.4f} | "
            f"val loss={row['val_loss']:.4f}(mal={row['val_loss_malignancy']:.4f},sub={row['val_loss_subtype']:.4f}), "
            f"auc={row['val_auc']:.4f}, f1={row['val_f1']:.4f}, recall={row['val_recall']:.4f}, "
            f"oracle_f1={row['val_subtype_oracle_macro_f1']:.4f}, cascade_acc={row['val_subtype_cascade_acc']:.4f}"
        )

        # ✅ early stopping 기준: val_loss → val_auc
        val_auc = row["val_auc"]
        if val_auc > best_auc + min_delta:
            best_auc           = val_auc
            early_stop_counter = 0
            torch.save(model.state_dict(), best_path)
            print(f"  -> best saved (val_auc={best_auc:.4f}): {best_path}")
        else:
            early_stop_counter += 1
            print(f"  -> no improvement ({early_stop_counter}/{patience})")

        if early_stop_counter >= patience:
            print(f"  -> early stopping triggered at epoch {epoch}")
            break

    history_df = pd.DataFrame(history)
    return model, history_df, best_path


# ============================================================
# 13) 모델 로드 함수
# ============================================================
def load_model(model_type, model_path, device):
    if model_type == "resnet":
        model = build_resnet50_patient_model(load_pretrained=False)
    elif model_type == "vit":
        model = build_vit_patient_model(load_pretrained=False)
    elif model_type == "hybrid":
        model = build_hybrid_patient_model(load_pretrained=False)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model


# ============================================================
# 14) CV 실행 함수
# ============================================================
def run_cv_for_model(model_name, model_type, build_fn):
    fold_results   = []
    fold_histories = []

    for fold_idx in FOLDS:
        print("\n" + "=" * 80)
        print(f"Running {model_name} | Fold {fold_idx}")
        print("=" * 80)

        train_loader, val_loader, pos_weight, subtype_class_weights = make_fold_loaders(fold_idx)

        model = build_fn()

        # CascadeCriterion: subtype_loss_weight는 fit_model 내부에서 epoch마다 업데이트
        criterion = CascadeCriterion(
            pos_weight=pos_weight,
            subtype_class_weights=subtype_class_weights,
            subtype_loss_weight=SUBTYPE_LOSS_WEIGHT_START,
        )

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
            criterion=criterion,
            patience=PATIENCE,
            min_delta=MIN_DELTA,
        )

        best_model    = load_model(model_type, best_path, device)
        final_metrics = eval_confusion(best_model, val_loader, criterion, device, threshold=THRESHOLD)

        print(f"\n[{model_name}][Fold {fold_idx}] Final Metrics")
        print(f"Malignancy  AUC={final_metrics['auc']:.4f}, F1={final_metrics['f1']:.4f}, Recall={final_metrics['recall']:.4f}")
        print(f"Subtype Oracle  Acc={final_metrics['subtype_oracle_acc']:.4f}, MacroF1={final_metrics['subtype_oracle_macro_f1']:.4f}")
        print(f"Cascade  Acc={final_metrics['subtype_cascade_acc']:.4f}, GateRecall={final_metrics['subtype_gate_recall']:.4f}")

        fold_results.append({
            "model":                   model_name,
            "fold":                    fold_idx,
            "mal_acc":                 final_metrics["acc"],
            "mal_auc":                 final_metrics["auc"],
            "mal_f1":                  final_metrics["f1"],
            "mal_recall":              final_metrics["recall"],
            "subtype_oracle_acc":      final_metrics["subtype_oracle_acc"],
            "subtype_oracle_macro_f1": final_metrics["subtype_oracle_macro_f1"],
            "subtype_cascade_acc":     final_metrics["subtype_cascade_acc"],
            "subtype_gate_recall":     final_metrics["subtype_gate_recall"],
            "binary_cm":               final_metrics["cm"],
            "subtype_oracle_cm":       final_metrics["subtype_oracle_cm"],
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
# 15) 학습 실행
# ============================================================

# ── ResNet50 ──────────────────────────────────────────────────
print(">>> Training ResNet50...")
resnet_cv_df, resnet_history_df = run_cv_for_model(
    model_name="ResNet50",
    model_type="resnet",
    build_fn=lambda: build_resnet50_patient_model(load_pretrained=True),
)
display(resnet_cv_df)

# ── ViT-B/16 ─────────────────────────────────────────────────
print(">>> Training ViT-B/16...")
vit_cv_df, vit_history_df = run_cv_for_model(
    model_name="ViT-B16",
    model_type="vit",
    build_fn=lambda: build_vit_patient_model(load_pretrained=True),
)
display(vit_cv_df)

# ── Hybrid ────────────────────────────────────────────────────
print(">>> Training Hybrid...")
hybrid_cv_df, hybrid_history_df = run_cv_for_model(
    model_name="Hybrid",
    model_type="hybrid",
    build_fn=lambda: build_hybrid_patient_model(load_pretrained=True),
)
display(hybrid_cv_df)


# ============================================================
# 16) 5-fold 평균 ± 표준편차 요약
# ============================================================
all_cv_df = pd.concat([resnet_cv_df, vit_cv_df, hybrid_cv_df], ignore_index=True)

summary_df = (
    all_cv_df.groupby("model")[[
        "mal_auc", "mal_f1", "mal_recall",
        "subtype_oracle_acc", "subtype_oracle_macro_f1",
        "subtype_cascade_acc", "subtype_gate_recall",
    ]]
    .agg(["mean", "std"])
)
display(summary_df)


# ============================================================
# 17) 학습 곡선 시각화
# ============================================================
def plot_one_model_by_fold(history_df, model_name):
    folds = sorted(history_df["fold"].unique())

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Loss
    ax = axes[0]
    for fold in folds:
        df_f = history_df[history_df["fold"] == fold].sort_values("epoch")
        ax.plot(df_f["epoch"], df_f["train_loss"], linestyle="--", alpha=0.7, label=f"Fold{fold} Train")
        ax.plot(df_f["epoch"], df_f["val_loss"],   alpha=0.7,      label=f"Fold{fold} Val")
    ax.set_title(f"{model_name} - Total Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend(fontsize=7)

    # Malignancy AUC
    ax = axes[1]
    for fold in folds:
        df_f = history_df[history_df["fold"] == fold].sort_values("epoch")
        ax.plot(df_f["epoch"], df_f["val_auc"], alpha=0.8, label=f"Fold{fold}")
    ax.set_title(f"{model_name} - Val Malignancy AUC")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("AUC")
    ax.legend(fontsize=7)

    # Subtype cascade acc
    ax = axes[2]
    for fold in folds:
        df_f = history_df[history_df["fold"] == fold].sort_values("epoch")
        ax.plot(df_f["epoch"], df_f["val_subtype_cascade_acc"], alpha=0.8, label=f"Fold{fold}")
    ax.set_title(f"{model_name} - Val Subtype Cascade Acc")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cascade Acc")
    ax.legend(fontsize=7)

    plt.suptitle(model_name, fontsize=13)
    plt.tight_layout()
    plt.show()


plot_one_model_by_fold(resnet_history_df, "ResNet50")
plot_one_model_by_fold(vit_history_df,    "ViT-B16")
plot_one_model_by_fold(hybrid_history_df, "Hybrid")
