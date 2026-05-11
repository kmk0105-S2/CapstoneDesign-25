# ============================================================
# 셀 1: Import / Seed / Config
# ============================================================
# 난소 종양 CT 2-Task 분류 모델 (이미지 진단 + 시각화 전용)
# - Augmentation 없음 (원본 이미지 학습)
# - Tabular/CSV 없음 (이미지 + radiomics만 사용)
# - Grad-CAM 시각화 포함 (학회 발표용)
# ============================================================

import os
import math
import random
import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models

from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

from PIL import Image, ImageDraw
import matplotlib
matplotlib.use("Agg")           # 서버 환경에서 GUI 없이 저장
import matplotlib.pyplot as plt
import matplotlib.cm as cm

warnings.filterwarnings("ignore")

# ─── 재현성 시드 ───────────────────────────────────────────────
SEED = 42

def set_seed(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

set_seed(SEED)

# ─── Config (수정 포인트는 여기만) ────────────────────────────
CFG = dict(
    # ── 경로 ─────────────────────────────────────────────────
    data_dir          = "./data",          # fold_{n}_{train|val}.npz 위치
    save_dir          = "./checkpoints",   # 모델 저장
    gradcam_dir       = "./gradcam",       # Grad-CAM 이미지 저장

    n_folds           = 5,

    # ── 이미지 ──────────────────────────────────────────────
    n_slices          = 8,
    img_h             = 224,
    img_w             = 224,
    roi_margin        = 20,               # ROI bounding box margin (px)
    use_mask_channel  = False,            # True → backbone 4채널 (옵션)

    # ── NPZ key ──────────────────────────────────────────────
    npz_image_key     = "images",         # (N, 8, 3, H, W) float32
    npz_ptid_key      = "pt_ids",         # (N,) str
    npz_polygon_key   = "polygons",       # (N, 8) object — per-slice polygon
    npz_mal_key       = "malignancy",     # (N,) int  0=benign 1=malignant
    npz_sub_key       = "subtype",        # (N,) int  -1=unknown, 0~K-1=subtype

    # ── 모델 ────────────────────────────────────────────────
    backbone_out      = 2048,             # ResNet50 출력 차원
    attn_hidden       = 512,
    shared_dim        = 512,
    tower_hidden      = 256,
    radio_proj_dim    = 128,
    n_subtypes        = 5,
    dropout           = 0.5,

    # ── Radiomics ───────────────────────────────────────────
    radio_channel         = 0,            # 0=original(기본)
    radio_extra_channels  = False,        # True → CLAHE/Sobel 추가
    radio_select_k        = 40,           # RadiomicsSelector 최종 k

    # ── Loss weights ─────────────────────────────────────────
    lambda_mal        = 1.0,
    lambda_sub        = 2.5,              # subtype 우선

    # ── 학습 ────────────────────────────────────────────────
    stage1_epochs     = 25,
    stage2_epochs     = 15,
    batch_size        = 8,
    lr_stage1         = 1e-4,
    lr_stage2         = 3e-5,
    weight_decay      = 1e-4,

    device            = "cuda" if torch.cuda.is_available() else "cpu",
)

os.makedirs(CFG["save_dir"],    exist_ok=True)
os.makedirs(CFG["gradcam_dir"], exist_ok=True)
print(f"Device : {CFG['device']}")
print("Config loaded ✓")


# ============================================================
# 셀 2: Polygon → Mask / BBox / ROI Crop 유틸
# ============================================================

def polygon_to_mask(polygon, height: int, width: int) -> np.ndarray:
    """
    polygon : list of (x,y) 또는 flat [x0,y0,x1,y1,...]
    → (H, W) uint8 binary mask
    None 또는 빈 polygon → 전체 0 mask 반환
    """
    if polygon is None or len(polygon) == 0:
        return np.zeros((height, width), dtype=np.uint8)

    pts_arr = np.array(polygon).reshape(-1, 2)
    pts     = [(float(p[0]), float(p[1])) for p in pts_arr]
    canvas  = Image.new("L", (width, height), 0)
    ImageDraw.Draw(canvas).polygon(pts, outline=1, fill=1)
    return np.array(canvas, dtype=np.uint8)          # (H, W)


def mask_to_bbox(mask: np.ndarray):
    """
    (H, W) mask → (y1, x1, y2, x2) bounding box
    mask가 비어 있으면 전체 영역 반환
    """
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    if not rows.any():
        return 0, 0, mask.shape[0], mask.shape[1]
    y1 = int(np.where(rows)[0][0]);  y2 = int(np.where(rows)[0][-1])
    x1 = int(np.where(cols)[0][0]);  x2 = int(np.where(cols)[0][-1])
    return y1, x1, y2, x2


def crop_roi(
    image:    np.ndarray,   # (C, H, W) float32
    mask:     np.ndarray,   # (H, W) uint8
    margin:   int = 20,
    target_h: int = 224,
    target_w: int = 224,
):
    """
    mask bounding box + margin 으로 ROI crop 후 target_size 로 resize.
    반환
      roi_img  : (C, target_h, target_w) float32
      roi_mask : (target_h, target_w)    uint8
    """
    H, W = image.shape[1], image.shape[2]
    y1, x1, y2, x2 = mask_to_bbox(mask)

    y1 = max(0, y1 - margin);  x1 = max(0, x1 - margin)
    y2 = min(H, y2 + margin);  x2 = min(W, x2 + margin)

    roi_img  = image[:, y1:y2, x1:x2]
    roi_mask = mask[y1:y2, x1:x2]

    def _resize(arr, h, w, resample):
        return np.array(Image.fromarray(arr).resize((w, h), resample))

    roi_img_r = np.stack([
        _resize(roi_img[c].astype(np.float32), target_h, target_w, Image.BILINEAR)
        for c in range(roi_img.shape[0])
    ], axis=0)                                   # (C, H, W)
    roi_mask_r = _resize(roi_mask, target_h, target_w, Image.NEAREST)

    return roi_img_r.astype(np.float32), roi_mask_r.astype(np.uint8)


def optional_4ch(roi_img: np.ndarray, roi_mask: np.ndarray,
                 use_mask_channel: bool = False) -> np.ndarray:
    """
    False → (3, H, W)  [기본]
    True  → (4, H, W)  mask channel append [옵션]
    """
    if not use_mask_channel:
        return roi_img
    return np.concatenate(
        [roi_img, roi_mask[np.newaxis].astype(np.float32)], axis=0
    )


print("Polygon / ROI utils loaded ✓")


# ============================================================
# 셀 3: SimpleRadiomicsExtractor
#   intensity(10) + texture(5) + shape(8) = 23 features
#   extract() 1회 호출로 전부 반환 (중복 없음)
# ============================================================

try:
    from skimage import measure as _ski_measure
    _SKIMAGE_OK = True
except ImportError:
    _SKIMAGE_OK = False
    print("skimage not found — shape features will be zero-filled")


class SimpleRadiomicsExtractor:
    """
    2D 슬라이스에서 23개 feature 를 한 번에 추출.
    [A] First-order intensity  10개
    [B] GLCM texture            5개
    [C] Shape (skimage)         8개
    """

    FEAT_NAMES = [
        "fo_mean","fo_std","fo_variance","fo_skewness","fo_kurtosis",
        "fo_energy","fo_entropy","fo_p10","fo_p90","fo_iqr",
        "tx_contrast","tx_homogeneity","tx_correlation","tx_asm","tx_dissimilarity",
        "sh_area","sh_perimeter","sh_compactness","sh_eccentricity",
        "sh_major_axis","sh_minor_axis","sh_bbox_aspect","sh_solidity",
    ]
    N_FEATS = 23

    def extract(self, image_2d: np.ndarray, mask_2d: np.ndarray) -> np.ndarray:
        """
        image_2d : (H, W) float32
        mask_2d  : (H, W) uint8
        반환     : (23,)  float32
        """
        roi = image_2d[mask_2d > 0].astype(np.float64)
        if roi.size == 0:
            roi = np.zeros(1, dtype=np.float64)

        f = np.zeros(self.N_FEATS, dtype=np.float32)

        # ── A: First-order ────────────────────────────────────
        f[0]  = float(np.mean(roi))
        f[1]  = float(np.std(roi))
        f[2]  = float(np.var(roi))
        f[3]  = self._skewness(roi)
        f[4]  = self._kurtosis(roi)
        f[5]  = float(np.sum(roi ** 2))
        f[6]  = self._entropy(roi)
        f[7]  = float(np.percentile(roi, 10))
        f[8]  = float(np.percentile(roi, 90))
        f[9]  = float(np.percentile(roi, 75) - np.percentile(roi, 25))

        # ── B: GLCM texture ───────────────────────────────────
        G      = self._glcm(image_2d, mask_2d)
        i_, j_ = np.mgrid[0:G.shape[0], 0:G.shape[1]]
        f[10] = float(np.sum(G * (i_ - j_) ** 2))
        f[11] = float(np.sum(G / (1 + (i_ - j_) ** 2 + 1e-8)))
        f[12] = self._glcm_corr(G, i_, j_)
        f[13] = float(np.sum(G ** 2))
        f[14] = float(np.sum(G * np.abs(i_ - j_)))

        # ── C: Shape ──────────────────────────────────────────
        f[15:] = self._shape(mask_2d)

        return f

    # ── helpers ───────────────────────────────────────────────
    @staticmethod
    def _skewness(x):
        mu, s = np.mean(x), np.std(x)
        return 0.0 if s < 1e-8 else float(np.mean(((x - mu) / s) ** 3))

    @staticmethod
    def _kurtosis(x):
        mu, s = np.mean(x), np.std(x)
        return 0.0 if s < 1e-8 else float(np.mean(((x - mu) / s) ** 4) - 3)

    @staticmethod
    def _entropy(x, bins=64):
        h, _ = np.histogram(x, bins=bins, density=True)
        h    = h[h > 0]
        return float(-np.sum(h * np.log2(h + 1e-10)) / np.log2(bins))

    @staticmethod
    def _quantize(img, levels=8):
        mn, mx = img.min(), img.max()
        if mx - mn < 1e-8:
            return np.zeros_like(img, dtype=np.int32)
        return np.clip(
            ((img - mn) / (mx - mn) * (levels - 1)).astype(np.int32),
            0, levels - 1
        )

    def _glcm(self, img, mask, levels=8):
        q = self._quantize(img, levels)
        G = np.zeros((levels, levels), dtype=np.float64)
        h, w = q.shape
        for r in range(h):
            for c in range(w - 1):
                if mask[r, c] and mask[r, c + 1]:
                    G[q[r, c], q[r, c + 1]] += 1
        s = G.sum()
        return G / s if s > 0 else G

    @staticmethod
    def _glcm_corr(G, i_, j_):
        mu_i = np.sum(i_ * G);  mu_j = np.sum(j_ * G)
        si   = np.sqrt(np.sum((i_ - mu_i) ** 2 * G) + 1e-8)
        sj   = np.sqrt(np.sum((j_ - mu_j) ** 2 * G) + 1e-8)
        return float(np.sum((i_ - mu_i) * (j_ - mu_j) * G) / (si * sj))

    @staticmethod
    def _shape(mask_2d):
        out = np.zeros(8, dtype=np.float32)
        if not _SKIMAGE_OK:
            return out
        lbl   = _ski_measure.label(mask_2d)
        props = _ski_measure.regionprops(lbl)
        if not props:
            return out
        rp    = max(props, key=lambda r: r.area)
        area  = float(rp.area)
        peri  = float(rp.perimeter) if rp.perimeter > 0 else 1.0
        bbox  = rp.bbox
        hb    = bbox[2] - bbox[0];  wb = bbox[3] - bbox[1]
        out[0] = area
        out[1] = peri
        out[2] = (4 * math.pi * area) / (peri ** 2 + 1e-8)
        out[3] = float(rp.eccentricity)
        out[4] = float(rp.major_axis_length)
        out[5] = float(rp.minor_axis_length)
        out[6] = float(hb / (wb + 1e-8))
        out[7] = float(rp.solidity)
        return out


def extract_radiomics_for_patient(
    images_8: np.ndarray,   # (8, 3, H, W)
    masks_8:  np.ndarray,   # (8, H, W)
    channel:  int  = 0,
    extra_channels: bool = False,
) -> np.ndarray:
    """
    8 슬라이스 평균 radiomics → (D_radio,) float32
    intensity + texture + shape 모두 extract() 한 번으로 추출 (중복 없음)
    """
    extractor = SimpleRadiomicsExtractor()
    ch_list   = [channel]
    if extra_channels:
        ch_list += [c for c in [0, 1, 2] if c != channel]

    slices_feats = []
    for s in range(images_8.shape[0]):
        for ch in ch_list:
            slices_feats.append(
                extractor.extract(images_8[s, ch], masks_8[s])
            )

    return np.stack(slices_feats).mean(axis=0).astype(np.float32)


print(f"SimpleRadiomicsExtractor: {SimpleRadiomicsExtractor.N_FEATS} features per channel")
print("Radiomics utils loaded ✓")


# ============================================================
# 셀 4: RadiomicsSelector
#   3단계 필터링 (fold-safe, malignant-only fit)
#   Step1: VarianceThreshold
#   Step2: Correlation Filter (|r| > 0.95)
#   Step3: SelectKBest (ANOVA, subtype target)
# ============================================================

class RadiomicsSelector:
    """
    [규칙] fit() 은 반드시 train fold 의 malignant & valid-subtype 샘플로만.
           val fold 는 transform() 만 호출.
    """

    def __init__(self, k: int = 40,
                 var_thr: float = 1e-4,
                 corr_thr: float = 0.95):
        self.k        = k
        self.var_thr  = var_thr
        self.corr_thr = corr_thr
        self._var     = None
        self._kept    = None
        self._kbest   = None
        self.out_dim_ = None

    def fit(self, X_tr: np.ndarray,
            mal_labels: np.ndarray,
            sub_labels: np.ndarray) -> "RadiomicsSelector":
        """
        X_tr       : (N_tr, D_raw)
        mal_labels : (N_tr,)  0/1
        sub_labels : (N_tr,)  -1=unknown, 0~K-1
        """
        valid = (mal_labels == 1) & (sub_labels >= 0)
        if valid.sum() < 5:
            print("  [RadiomicsSelector] warn: malignant valid < 5, using all")
            valid = np.ones(len(X_tr), dtype=bool)

        Xm = X_tr[valid];  ym = sub_labels[valid]
        print(f"  [RadiomicsSelector] fit on {Xm.shape[0]} mal samples, "
              f"input_dim={Xm.shape[1]}")

        # Step 1: VarianceThreshold
        self._var = VarianceThreshold(threshold=self.var_thr)
        Xv = self._var.fit_transform(Xm)
        print(f"    Step1 VarianceThreshold : {Xv.shape[1]}")

        # Step 2: Correlation filter
        self._kept = self._corr_filter(Xv, self.corr_thr)
        Xc = Xv[:, self._kept]
        print(f"    Step2 Correlation Filter: {Xc.shape[1]}")

        # Step 3: SelectKBest (ANOVA)
        k_act = min(self.k, Xc.shape[1])
        self._kbest = SelectKBest(f_classif, k=k_act)
        self._kbest.fit(Xc, ym)
        self.out_dim_ = k_act
        print(f"    Step3 SelectKBest(k={k_act}) → out_dim={k_act}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._var is not None, "fit() 먼저 호출하세요"
        Xv = self._var.transform(X)
        Xc = Xv[:, self._kept]
        return self._kbest.transform(Xc).astype(np.float32)

    def fit_transform(self, X_tr, mal_labels, sub_labels):
        self.fit(X_tr, mal_labels, sub_labels)
        return self.transform(X_tr)

    @staticmethod
    def _corr_filter(X: np.ndarray, thr: float) -> np.ndarray:
        if X.shape[1] <= 1:
            return np.arange(X.shape[1])
        corr = np.nan_to_num(np.corrcoef(X.T))
        drop = set()
        for i in range(corr.shape[0]):
            if i in drop: continue
            for j in range(i + 1, corr.shape[0]):
                if j in drop: continue
                if abs(corr[i, j]) > thr:
                    drop.add(j)
        return np.array([i for i in range(corr.shape[0]) if i not in drop],
                        dtype=np.int32)


print("RadiomicsSelector loaded ✓")


# ============================================================
# 셀 5: NPZ Fold Loader + Radiomics 준비
# ============================================================

def load_npz_fold(data_dir: str, fold: int, split: str, cfg: dict) -> dict:
    """
    fold_{fold}_{split}.npz 로딩.
    반환 dict:
      images     (N, 8, 3, H, W) float32
      pt_ids     (N,) str
      polygons   (N, 8) object
      mal_labels (N,) int32
      sub_labels (N,) int32  (-1=benign/unknown)
    """
    path = os.path.join(data_dir, f"fold_{fold}_{split}.npz")
    d    = np.load(path, allow_pickle=True)
    return dict(
        images     = d[cfg["npz_image_key"]].astype(np.float32),
        pt_ids     = d[cfg["npz_ptid_key"]],
        polygons   = d[cfg["npz_polygon_key"]],
        mal_labels = d[cfg["npz_mal_key"]].astype(np.int32),
        sub_labels = d[cfg["npz_sub_key"]].astype(np.int32),
    )


def prepare_radiomics(images: np.ndarray,    # (N, 8, 3, H, W)
                      polygons: np.ndarray,  # (N, 8) object
                      cfg: dict) -> np.ndarray:
    """
    전체 샘플 radiomics 추출 (중복 호출 없음).
    반환: (N, D_raw) float32
    """
    N, S, C, H, W = images.shape
    out = []
    for i in range(N):
        masks = np.zeros((S, H, W), dtype=np.uint8)
        for s in range(S):
            poly    = polygons[i][s] if polygons[i] is not None else None
            masks[s] = polygon_to_mask(poly, H, W)
        out.append(extract_radiomics_for_patient(
            images[i], masks,
            channel        = cfg["radio_channel"],
            extra_channels = cfg["radio_extra_channels"],
        ))
    return np.stack(out).astype(np.float32)    # (N, D_raw)


print("NPZ loader / radiomics prep loaded ✓")


# ============================================================
# 셀 6: OvarianDataset
#   입력: roi_imgs, radio_feat, mal_label, sub_label
#   Augmentation 없음 / Tabular 없음
# ============================================================

class OvarianDataset(Dataset):
    """
    환자 단위 Dataset.
    Augmentation 미적용 (원본 이미지 그대로).
    Tabular 없음.

    __getitem__ 반환:
      roi_imgs   (8, C, H, W) float32
      radio_feat (D_radio_sel,) float32
      mal_label  scalar long
      sub_label  scalar long  (-1=benign/unknown)
    """

    def __init__(
        self,
        images:      np.ndarray,   # (N, 8, 3, H, W)
        polygons:    np.ndarray,   # (N, 8) object
        radio_feats: np.ndarray,   # (N, D_radio_sel)
        mal_labels:  np.ndarray,   # (N,)
        sub_labels:  np.ndarray,   # (N,)
        cfg:         dict,
    ):
        self.images      = images
        self.polygons    = polygons
        self.radio_feats = radio_feats
        self.mal_labels  = mal_labels
        self.sub_labels  = sub_labels
        self.cfg         = cfg

        self.n_slices    = cfg["n_slices"]
        self.img_h       = cfg["img_h"]
        self.img_w       = cfg["img_w"]
        self.margin      = cfg["roi_margin"]
        self.use_mask_ch = cfg["use_mask_channel"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx: int) -> dict:
        imgs_8  = self.images[idx]    # (8, 3, H, W)
        polys_8 = self.polygons[idx]  # list/array of 8 polygons
        H, W    = imgs_8.shape[2], imgs_8.shape[3]

        roi_list = []
        for s in range(self.n_slices):
            poly   = polys_8[s] if polys_8 is not None else None
            mask2d = polygon_to_mask(poly, H, W)         # (H, W)
            img_s  = imgs_8[s]                           # (3, H, W)

            roi_img, roi_mask = crop_roi(
                img_s, mask2d,
                margin=self.margin,
                target_h=self.img_h,
                target_w=self.img_w,
            )                                             # (3,H,W), (H,W)

            roi_img = optional_4ch(roi_img, roi_mask, self.use_mask_ch)
            # → (3, H, W) 또는 (4, H, W)

            # Augmentation 없음: 원본 그대로
            roi_list.append(roi_img)

        roi_imgs = np.stack(roi_list, axis=0).astype(np.float32)  # (8, C, H, W)

        return dict(
            roi_imgs   = torch.from_numpy(roi_imgs),
            radio_feat = torch.from_numpy(self.radio_feats[idx]),
            mal_label  = torch.tensor(self.mal_labels[idx], dtype=torch.long),
            sub_label  = torch.tensor(self.sub_labels[idx], dtype=torch.long),
        )


def build_dataloaders(train_ds: OvarianDataset,
                      val_ds:   OvarianDataset,
                      cfg:      dict):
    tr = DataLoader(
        train_ds, batch_size=cfg["batch_size"],
        shuffle=True,  num_workers=0,
        pin_memory=(cfg["device"] == "cuda"), drop_last=True,
    )
    va = DataLoader(
        val_ds,   batch_size=cfg["batch_size"],
        shuffle=False, num_workers=0,
        pin_memory=(cfg["device"] == "cuda"), drop_last=False,
    )
    return tr, va


print("OvarianDataset / DataLoader loaded ✓")


# ============================================================
# 셀 7: SliceEncoder (ResNet50)
# ============================================================

class SliceEncoder(nn.Module):
    """
    단일 슬라이스 인코더.
    입력  : (B*8, C_in, H, W)
    출력  : (B*8, 2048)

    [Grad-CAM 주의]
    features[7] = layer4 가 Grad-CAM hook 대상.
    forward 에서 features 를 Sequential 로 순차 통과 →
    hook 이 features[7] 출력 feature map 을 캡처.
    """

    def __init__(self, in_channels: int = 3, pretrained: bool = True):
        super().__init__()
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        resnet  = models.resnet50(weights=weights)

        if in_channels != 3:
            old = resnet.conv1
            new = nn.Conv2d(in_channels, 64,
                            kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new.weight[:, :3] = old.weight
                if in_channels > 3:
                    new.weight[:, 3:] = old.weight.mean(dim=1, keepdim=True)
            resnet.conv1 = new

        # Sequential 으로 묶어 인덱스 접근 가능하게 유지
        # [0]=conv1 [1]=bn1 [2]=relu [3]=maxpool
        # [4]=layer1 [5]=layer2 [6]=layer3 [7]=layer4
        self.features = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4,
        )
        self.pool    = nn.AdaptiveAvgPool2d(1)
        self.out_dim = 2048

    def forward(self, x):
        # x : (B*8, C, H, W)
        feat = self.features(x)   # (B*8, 2048, h', w')  ← layer4 출력
        feat = self.pool(feat)     # (B*8, 2048, 1, 1)
        return feat.flatten(1)     # (B*8, 2048)


print("SliceEncoder loaded ✓")


# ============================================================
# 셀 8: AttentionPooling
# ============================================================

class AttentionPooling(nn.Module):
    """
    8개 슬라이스 feature 를 attention weighted sum 으로 통합.
    입력  : (B, 8, D)
    출력  : z_pooled (B, D),  attn_w (B, 8)
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        # x       : (B, 8, D)
        scores  = self.attn(x)               # (B, 8, 1)
        weights = F.softmax(scores, dim=1)   # (B, 8, 1)
        out     = (weights * x).sum(dim=1)   # (B, D)
        return out, weights.squeeze(-1)       # (B, D), (B, 8)


print("AttentionPooling loaded ✓")


# ============================================================
# 셀 9: OvarianTwoTaskModel
#   구조: SliceEncoder → AttentionPooling → shared_proj
#            ├── mal_tower → mal_head
#            └── sub_tower + radio_encoder → sub_head
#
#   [Grad-CAM 설계 원칙]
#   - forward 내부에서 slice_enc.features 를 명시적으로 통과시켜
#     hook 이 features[7](layer4) 출력을 캡처할 수 있게 유지.
#   - hook 은 모델 외부(GradCAM 클래스)에서 등록하므로
#     forward 자체는 깔끔하게 유지.
# ============================================================

def _mlp(in_d: int, hid_d: int, out_d: int,
         drop: float = 0.5, n_hid: int = 2) -> nn.Sequential:
    """범용 MLP: in_d → [hid_d × n_hid] → out_d"""
    layers, cur = [], in_d
    for _ in range(n_hid):
        layers += [nn.Linear(cur, hid_d),
                   nn.BatchNorm1d(hid_d),
                   nn.ReLU(inplace=True),
                   nn.Dropout(drop)]
        cur = hid_d
    layers.append(nn.Linear(cur, out_d))
    return nn.Sequential(*layers)


class OvarianTwoTaskModel(nn.Module):
    """
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    입력
      roi_imgs   (B, 8, C, H, W)
      radio_feat (B, D_radio_sel)

    핵심 흐름
      ① SliceEncoder(ResNet50)
            (B,8,C,H,W) → view (B*8,C,H,W)
            → features[0..7] 순차 통과 → (B*8,2048,h,w)
            → AdaptiveAvgPool → flatten → (B*8,2048)
            → view (B, 8, 2048)

      ② AttentionPooling
            (B,8,2048) → z_pooled (B,2048), attn_w (B,8)

      ③ shared_proj
            (B,2048) → z_shared (B,shared_dim)

              ┌────────────────┬─────────────────────────┐
              ▼                ▼
        mal_tower          sub_tower
              │                │
         mal_head          radio_encoder(radio_feat)→z_radio
              │                │
       mal_logit(B,)    sub_head([z_sub ‖ z_radio ‖ p_mal])
                               │
                         sub_logit(B,n_sub)
    ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    """

    def __init__(
        self,
        in_channels:   int,    # 3 or 4
        radio_dim:     int,
        n_subtypes:    int,
        cfg:           dict,
        pretrained_bb: bool = True,
    ):
        super().__init__()
        self.shared_dim   = cfg["shared_dim"]
        self.tower_hidden = cfg["tower_hidden"]
        self.radio_proj_d = cfg["radio_proj_dim"]
        self.drop         = cfg["dropout"]

        # ① Slice Encoder (ResNet50)
        self.slice_enc = SliceEncoder(in_channels=in_channels,
                                       pretrained=pretrained_bb)
        bb_out = self.slice_enc.out_dim  # 2048

        # ② Attention Pooling
        self.attn_pool = AttentionPooling(bb_out, cfg["attn_hidden"])

        # ③ Shared Projection: (B,2048) → (B,shared_dim)
        self.shared_proj = nn.Sequential(
            nn.Linear(bb_out, self.shared_dim),
            nn.BatchNorm1d(self.shared_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop),
        )

        # Malignancy Tower + Head
        self.mal_tower = _mlp(self.shared_dim, self.tower_hidden,
                               self.tower_hidden, self.drop)
        self.mal_head  = nn.Linear(self.tower_hidden, 1)

        # Subtype Tower + Head (Conditional on p_mal)
        self.sub_tower     = _mlp(self.shared_dim, self.tower_hidden,
                                   self.tower_hidden, self.drop)
        self.radio_encoder = _mlp(radio_dim, self.tower_hidden,
                                   self.radio_proj_d, self.drop)
        sub_head_in = self.tower_hidden + self.radio_proj_d + 1
        self.sub_head = nn.Sequential(
            nn.Linear(sub_head_in, self.tower_hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(self.drop),
            nn.Linear(self.tower_hidden, n_subtypes),
        )

    def forward(self, roi_imgs: torch.Tensor,
                radio_feat: torch.Tensor) -> dict:
        """
        roi_imgs   : (B, 8, C, H, W)
        radio_feat : (B, D_radio_sel)

        [Grad-CAM 호환]
        slice_enc.features 를 통과한 feature map 이
        pool 되기 전 (B*8, 2048, h', w') 상태로 존재한다.
        GradCAM 클래스가 features[7] 에 hook 을 달면
        이 map 의 gradient/activation 을 캡처할 수 있다.
        forward 자체는 hook 로직 없이 깔끔하게 유지.
        """
        B, S, C, H, W = roi_imgs.shape

        # ① Slice Encoder
        #   (B,8,C,H,W) → (B*8,C,H,W) [펼치기]
        x_flat    = roi_imgs.view(B * S, C, H, W)
        feat_flat = self.slice_enc(x_flat)          # (B*8, 2048)
        #   (B*8,2048) → (B,8,2048) [되돌리기]
        feat_8    = feat_flat.view(B, S, -1)

        # ② Attention Pooling
        #   (B,8,2048) → z_pooled(B,2048), attn_w(B,8)
        z_pooled, attn_w = self.attn_pool(feat_8)

        # ③ Shared Projection
        #   (B,2048) → z_shared(B,shared_dim)
        z_shared = self.shared_proj(z_pooled)

        # Malignancy Tower
        z_mal     = self.mal_tower(z_shared)             # (B, tower_hidden)
        mal_logit = self.mal_head(z_mal).squeeze(-1)     # (B,)
        p_mal     = torch.sigmoid(mal_logit).unsqueeze(-1)  # (B, 1)

        # Subtype Tower (Conditional)
        z_sub     = self.sub_tower(z_shared)             # (B, tower_hidden)
        z_radio   = self.radio_encoder(radio_feat)       # (B, radio_proj_d)
        sub_in    = torch.cat([z_sub, z_radio, p_mal], dim=-1)
        sub_logit = self.sub_head(sub_in)                # (B, n_subtypes)

        return dict(
            mal_logit = mal_logit,   # (B,)
            sub_logit = sub_logit,   # (B, n_subtypes)
            attn_w    = attn_w,      # (B, 8)
        )


print("OvarianTwoTaskModel loaded ✓")


# ============================================================
# 셀 10: Grad-CAM 구현
#   Target layer : slice_enc.features[7] (ResNet50 layer4)
#   지원 task    : 'mal' (malignancy) | 'sub' (subtype)
# ============================================================

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    사용 흐름:
      gcam = GradCAM(model, target_layer=model.slice_enc.features[7])
      heatmaps = gcam.compute(roi_imgs_1pat, radio_feat_1pat,
                              task='mal', class_idx=None)
      gcam.remove_hooks()

    compute() 반환:
      heatmaps : (8, H, W) float32  0~1 정규화된 슬라이스별 heatmap
    """

    def __init__(self, model: OvarianTwoTaskModel,
                 target_layer: nn.Module):
        self.model        = model
        self.target_layer = target_layer
        self._activations = None   # forward hook 캡처
        self._gradients   = None   # backward hook 캡처
        self._hooks       = []
        self._register_hooks()

    # ── hook 등록 ─────────────────────────────────────────────
    def _register_hooks(self):
        """
        forward hook  : target_layer 출력 feature map 캡처
        backward hook : target_layer 에 흐르는 gradient 캡처
        """
        def _fwd_hook(module, input, output):
            # output : (B*8, C_feat, h, w)
            self._activations = output.detach()

        def _bwd_hook(module, grad_in, grad_out):
            # grad_out[0] : (B*8, C_feat, h, w)
            self._gradients = grad_out[0].detach()

        self._hooks.append(
            self.target_layer.register_forward_hook(_fwd_hook)
        )
        self._hooks.append(
            self.target_layer.register_full_backward_hook(_bwd_hook)
        )

    def remove_hooks(self):
        """hook 제거 — 사용 후 반드시 호출"""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    # ── Grad-CAM 계산 ──────────────────────────────────────────
    def compute(
        self,
        roi_imgs:   torch.Tensor,   # (1, 8, C, H, W)  단일 환자
        radio_feat: torch.Tensor,   # (1, D_radio)
        task:       str = "mal",    # 'mal' or 'sub'
        class_idx:  int = None,     # None → argmax class
    ) -> np.ndarray:
        """
        단일 환자 1명의 8개 슬라이스에 대해 Grad-CAM heatmap 계산.

        task='mal'  → malignancy logit 에 대한 gradient
        task='sub'  → subtype logit[class_idx] 에 대한 gradient

        반환: (8, H_orig, W_orig) float32  [0, 1] 정규화
        """
        assert roi_imgs.shape[0] == 1, "배치 크기 1 (단일 환자)만 지원"
        assert task in ("mal", "sub"), "task 는 'mal' 또는 'sub'"

        self.model.eval()
        roi_imgs   = roi_imgs.to(next(self.model.parameters()).device)
        radio_feat = radio_feat.to(next(self.model.parameters()).device)
        roi_imgs.requires_grad_(False)

        # ── forward ──────────────────────────────────────────
        self.model.zero_grad()
        outputs = self.model(roi_imgs, radio_feat)

        # ── 역전파 대상 스칼라 선택 ──────────────────────────
        if task == "mal":
            # malignancy logit 스칼라
            score = outputs["mal_logit"][0]          # scalar

        else:  # task == 'sub'
            logits = outputs["sub_logit"][0]         # (n_subtypes,)
            if class_idx is None:
                class_idx = int(logits.argmax())
            score = logits[class_idx]

        # ── backward ─────────────────────────────────────────
        self.model.zero_grad()
        score.backward(retain_graph=False)

        # ── activation & gradient: (B*8, C_feat, h, w) ───────
        # B=1 이므로 B*8=8
        act  = self._activations   # (8, C_feat, h, w)
        grad = self._gradients     # (8, C_feat, h, w)

        # ── global average pooling of gradients ───────────────
        # weights: (8, C_feat, 1, 1)
        weights = grad.mean(dim=(2, 3), keepdim=True)

        # ── weighted combination of feature maps ──────────────
        cam = (weights * act).sum(dim=1)          # (8, h, w)
        cam = F.relu(torch.from_numpy(            # ReLU: 양수만
            cam.cpu().numpy()
        )).numpy()                                 # (8, h, w)

        # ── 원본 이미지 사이즈로 upsample + [0,1] 정규화 ──────
        H_orig = roi_imgs.shape[3]
        W_orig = roi_imgs.shape[4]
        heatmaps = []
        for s in range(cam.shape[0]):
            # PIL 로 bilinear upsample
            cam_s   = cam[s]                       # (h, w)
            # 0~255 uint8 로 변환 후 PIL resize
            cam_min, cam_max = cam_s.min(), cam_s.max()
            if cam_max - cam_min > 1e-8:
                cam_norm = (cam_s - cam_min) / (cam_max - cam_min)
            else:
                cam_norm = np.zeros_like(cam_s)
            cam_pil = Image.fromarray(
                (cam_norm * 255).astype(np.uint8)
            )
            cam_up  = np.array(
                cam_pil.resize((W_orig, H_orig), Image.BILINEAR)
            ).astype(np.float32) / 255.0           # (H, W) [0,1]
            heatmaps.append(cam_up)

        return np.stack(heatmaps, axis=0)          # (8, H, W)


# ── Grad-CAM 시각화 저장 유틸 ─────────────────────────────────

def save_gradcam_grid(
    roi_imgs:   np.ndarray,     # (8, C, H, W) float32  원본 ROI 이미지
    heatmaps:   np.ndarray,     # (8, H, W)    float32  [0,1]
    save_path:  str,
    task:       str = "mal",
    pt_id:      str = "",
    alpha:      float = 0.45,   # overlay 투명도
):
    """
    8개 슬라이스에 대해 [원본 | heatmap overlay] 2행 그리드로 저장.

    roi_imgs : channel 0 (original) 을 grayscale 로 시각화.
    overlay  : jet colormap heatmap 을 alpha blending.

    저장 파일: {save_path}
    """
    n_slices = roi_imgs.shape[0]
    fig, axes = plt.subplots(
        2, n_slices,
        figsize=(n_slices * 2.5, 5.5),
        constrained_layout=True,
    )
    fig.suptitle(
        f"Grad-CAM  |  task={task}  |  pt={pt_id}",
        fontsize=11, fontweight="bold"
    )

    cmap_heat = cm.get_cmap("jet")

    for s in range(n_slices):
        # ── 원본 이미지 (channel 0 grayscale) ─────────────────
        img_gray = roi_imgs[s, 0]                   # (H, W) [0,1]
        axes[0, s].imshow(img_gray, cmap="gray",
                          vmin=0, vmax=1, interpolation="bilinear")
        axes[0, s].set_title(f"Slice {s}", fontsize=8)
        axes[0, s].axis("off")

        # ── Heatmap Overlay ─────────────────────────────────
        # grayscale → RGB
        img_rgb  = np.stack([img_gray] * 3, axis=-1)    # (H, W, 3)
        heat_rgb = cmap_heat(heatmaps[s])[:, :, :3]     # (H, W, 3) jet
        overlay  = (1 - alpha) * img_rgb + alpha * heat_rgb
        overlay  = np.clip(overlay, 0, 1)

        axes[1, s].imshow(overlay, interpolation="bilinear")
        axes[1, s].axis("off")

    axes[0, 0].set_ylabel("Original", fontsize=9)
    axes[1, 0].set_ylabel("Grad-CAM", fontsize=9)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Grad-CAM saved → {save_path}")


def run_gradcam_for_patient(
    model:      OvarianTwoTaskModel,
    dataset:    OvarianDataset,
    patient_idx: int,
    task:       str = "mal",
    class_idx:  int = None,
    save_dir:   str = "./gradcam",
    cfg:        dict = None,
) -> np.ndarray:
    """
    특정 환자 인덱스에 대해 Grad-CAM 을 계산하고 이미지로 저장.

    반환: heatmaps (8, H, W)
    """
    device = next(model.parameters()).device
    sample = dataset[patient_idx]

    # (1, 8, C, H, W), (1, D_radio)
    roi_imgs   = sample["roi_imgs"].unsqueeze(0).to(device)
    radio_feat = sample["radio_feat"].unsqueeze(0).to(device)

    # Grad-CAM 대상: slice_enc.features[7] = layer4
    gcam = GradCAM(model, target_layer=model.slice_enc.features[7])

    try:
        heatmaps = gcam.compute(roi_imgs, radio_feat,
                                task=task, class_idx=class_idx)
    finally:
        gcam.remove_hooks()    # 항상 hook 제거

    # 저장 경로
    os.makedirs(save_dir, exist_ok=True)
    mal_lbl = int(sample["mal_label"].item())
    sub_lbl = int(sample["sub_label"].item())
    fname   = (f"idx{patient_idx}_mal{mal_lbl}_sub{sub_lbl}"
               f"_task{task}.png")
    save_path = os.path.join(save_dir, fname)

    # roi_imgs numpy (8, C, H, W)
    roi_np = sample["roi_imgs"].numpy()
    save_gradcam_grid(
        roi_imgs  = roi_np,
        heatmaps  = heatmaps,
        save_path = save_path,
        task      = task,
        pt_id     = f"idx{patient_idx}",
    )
    return heatmaps


print("GradCAM / visualization utils loaded ✓")


# ============================================================
# 셀 11: Loss 함수 (2-Task)
#   L_total = λ_mal * L_mal + λ_sub * L_sub
# ============================================================

class TwoTaskCriterion(nn.Module):
    """
    L_mal : BCEWithLogitsLoss (전체 샘플)
    L_sub : CrossEntropyLoss  (malignant & valid subtype 샘플만)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        self.lam_mal  = cfg["lambda_mal"]
        self.lam_sub  = cfg["lambda_sub"]
        self.mal_loss = nn.BCEWithLogitsLoss()
        self.sub_loss = nn.CrossEntropyLoss()

    def forward(self, outputs: dict, batch: dict) -> dict:
        device = outputs["mal_logit"].device

        # Malignancy
        L_mal = self.mal_loss(
            outputs["mal_logit"],
            batch["mal_label"].float().to(device),
        )

        # Subtype: malignant & valid label 만
        mal_m  = (batch["mal_label"] == 1).to(device)
        sub_lbl = batch["sub_label"].to(device)
        valid   = mal_m & (sub_lbl >= 0)

        if valid.sum() > 0:
            L_sub = self.sub_loss(
                outputs["sub_logit"][valid],
                sub_lbl[valid],
            )
        else:
            L_sub = torch.tensor(0.0, device=device)

        L_total = self.lam_mal * L_mal + self.lam_sub * L_sub
        return dict(total=L_total, mal=L_mal, sub=L_sub)


print("TwoTaskCriterion loaded ✓")


# ============================================================
# 셀 12: Metrics
# ============================================================

def compute_metrics(
    mal_logits: np.ndarray,   # (N,)
    mal_labels: np.ndarray,   # (N,)
    sub_logits: np.ndarray,   # (N, n_sub)
    sub_labels: np.ndarray,   # (N,)
) -> dict:
    """
    반환:
      mal_auc, mal_f1, mal_acc
      subtype_macro_f1
      selection_score = 0.4*mal_auc + 0.6*subtype_macro_f1
    """
    probs    = 1.0 / (1.0 + np.exp(-mal_logits))
    mal_pred = (probs >= 0.5).astype(int)

    try:
        mal_auc = roc_auc_score(mal_labels, probs)
    except Exception:
        mal_auc = float("nan")

    mal_f1  = f1_score(mal_labels, mal_pred, zero_division=0)
    mal_acc = accuracy_score(mal_labels, mal_pred)

    sub_mask = (mal_labels == 1) & (sub_labels >= 0)
    if sub_mask.sum() >= 2:
        sub_pred = np.argmax(sub_logits[sub_mask], axis=1)
        sub_f1   = f1_score(sub_labels[sub_mask], sub_pred,
                            average="macro", zero_division=0)
    else:
        sub_f1 = float("nan")

    a   = mal_auc if not math.isnan(mal_auc) else 0.0
    s   = sub_f1  if not math.isnan(sub_f1)  else 0.0
    sel = 0.4 * a + 0.6 * s

    return dict(
        mal_auc=mal_auc, mal_f1=mal_f1, mal_acc=mal_acc,
        subtype_macro_f1=sub_f1, selection_score=sel,
    )


print("Metrics loaded ✓")


# ============================================================
# 셀 13: FrozenBNGuard + set_stage + build_optimizer
# ============================================================
#
# [문제] PyTorch model.train() 은 모든 하위 BN.training=True 로 전파.
#        freeze 이후 train() 을 부르면 frozen BN 의 eval 상태가 덮어씌워짐.
#        → running_mean/var 가 학습 중 오염.
#
# [해결] FrozenBNGuard: frozen 모듈 목록 보관,
#        train loop 의 매 배치 forward 직전 refreeze() 로 eval 재고정.
# ─────────────────────────────────────────────────────────────

class FrozenBNGuard:
    """
    frozen 모듈의 BN running stat 을 보호.

    - register(module): 보호 대상 등록
    - refreeze()       : 매 배치 forward 전 호출 → eval() 재고정
    - clear()          : stage 전환 시 목록 초기화
    """

    def __init__(self):
        self._modules: list[nn.Module] = []

    def register(self, m: nn.Module) -> "FrozenBNGuard":
        self._modules.append(m)
        return self

    def clear(self):
        self._modules.clear()

    def refreeze(self):
        """
        model.train() 이 덮어쓴 frozen BN 의 training 플래그를
        다시 False(eval) 로 복원.
        joint_finetune 에서는 _modules 가 비어 있으므로 no-op.
        """
        for mod in self._modules:
            for m in mod.modules():
                if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    m.eval()

    def freeze_and_register(self, module: nn.Module) -> "FrozenBNGuard":
        """파라미터 grad 차단 + BN eval + guard 등록을 한 번에."""
        for p in module.parameters():
            p.requires_grad = False
        for m in module.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                m.eval()
        self.register(module)
        return self


def _unfreeze(module: nn.Module):
    """파라미터 grad 재개 + BN train 복구."""
    for p in module.parameters():
        p.requires_grad = True
    for m in module.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.train()


def set_stage(model: OvarianTwoTaskModel,
              stage: str) -> FrozenBNGuard:
    """
    stage 에 따라 freeze/unfreeze 설정.
    반환값 FrozenBNGuard 를 train_one_epoch 에 반드시 전달.

    'cls_pretrain'
      - features[0]=conv1, [1]=bn1, [4]=layer1, [5]=layer2 → freeze
      - features[6]=layer3, [7]=layer4, attn_pool, shared_proj,
        mal/sub 관련 모든 레이어 → 학습

    'joint_finetune'
      - 전체 unfreeze, guard 는 빈 채로 반환
    """
    assert stage in ("cls_pretrain", "joint_finetune")
    guard = FrozenBNGuard()

    if stage == "cls_pretrain":
        # freeze: conv1, bn1, layer1, layer2
        for idx in (0, 1, 4, 5):
            guard.freeze_and_register(model.slice_enc.features[idx])
        # unfreeze: layer3, layer4, pool
        for idx in (6, 7):
            _unfreeze(model.slice_enc.features[idx])
        _unfreeze(model.slice_enc.pool)
        _unfreeze(model.attn_pool)
        _unfreeze(model.shared_proj)
        _unfreeze(model.mal_tower)
        _unfreeze(model.mal_head)
        _unfreeze(model.sub_tower)
        _unfreeze(model.radio_encoder)
        _unfreeze(model.sub_head)

    elif stage == "joint_finetune":
        _unfreeze(model)        # 전체 unfreeze — guard 비어있음

    n_frozen = len(guard._modules)
    print(f"  Stage → '{stage}' | frozen sub-modules: {n_frozen}")
    return guard


def build_optimizer(model: OvarianTwoTaskModel,
                    stage: str, cfg: dict):
    lr = cfg["lr_stage1"] if stage == "cls_pretrain" else cfg["lr_stage2"]
    params = [p for p in model.parameters() if p.requires_grad]
    opt    = torch.optim.AdamW(params, lr=lr,
                                weight_decay=cfg["weight_decay"])
    n_ep   = (cfg["stage1_epochs"] if stage == "cls_pretrain"
              else cfg["stage2_epochs"])
    sch    = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_ep, eta_min=lr * 0.01
    )
    return opt, sch


print("FrozenBNGuard / set_stage / optimizer loaded ✓")


# ============================================================
# 셀 14: train_one_epoch / validate_one_epoch
# ============================================================

def train_one_epoch(
    model:     OvarianTwoTaskModel,
    loader:    DataLoader,
    criterion: TwoTaskCriterion,
    optimizer: torch.optim.Optimizer,
    device:    str,
    bn_guard:  FrozenBNGuard,
) -> dict:
    """
    1 epoch 학습.

    bn_guard.refreeze() 를 매 배치 forward 직전 호출:
      model.train() → 전체 BN.training=True 로 덮임
      → bn_guard.refreeze() → frozen BN 만 eval 재고정
    joint_finetune 시 guard 비어있으므로 refreeze() = no-op.

    반환: {total, mal, sub} 평균 loss
    """
    model.train()

    running = defaultdict(float)
    n_batch = 0

    for batch in loader:
        # ── [핵심] frozen BN eval 재고정 ─────────────────────
        bn_guard.refreeze()

        optimizer.zero_grad()
        roi_imgs   = batch["roi_imgs"].to(device)
        radio_feat = batch["radio_feat"].to(device)

        outputs = model(roi_imgs, radio_feat)
        losses  = criterion(outputs, batch)

        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad],
            max_norm=1.0,
        )
        optimizer.step()

        for k, v in losses.items():
            running[k] += v.item()
        n_batch += 1

    return {k: v / max(n_batch, 1) for k, v in running.items()}


@torch.no_grad()
def validate_one_epoch(
    model:     OvarianTwoTaskModel,
    loader:    DataLoader,
    criterion: TwoTaskCriterion,
    device:    str,
) -> tuple:
    """
    1 epoch validation.
    반환: (avg_loss dict, metrics dict)
    """
    model.eval()
    running = defaultdict(float)
    n_batch = 0

    all_ml, all_lb_m = [], []
    all_sl, all_lb_s = [], []

    for batch in loader:
        roi_imgs   = batch["roi_imgs"].to(device)
        radio_feat = batch["radio_feat"].to(device)

        outputs = model(roi_imgs, radio_feat)
        losses  = criterion(outputs, batch)

        for k, v in losses.items():
            running[k] += v.item()
        n_batch += 1

        all_ml.append(outputs["mal_logit"].cpu().numpy())
        all_lb_m.append(batch["mal_label"].numpy())
        all_sl.append(outputs["sub_logit"].cpu().numpy())
        all_lb_s.append(batch["sub_label"].numpy())

    avg_loss = {k: v / max(n_batch, 1) for k, v in running.items()}
    metrics  = compute_metrics(
        np.concatenate(all_ml),  np.concatenate(all_lb_m),
        np.concatenate(all_sl),  np.concatenate(all_lb_s),
    )
    return avg_loss, metrics


print("Train / Val loop loaded ✓")


# ============================================================
# 셀 15: build_datasets_for_fold
# ============================================================

def build_datasets_for_fold(fold: int, cfg: dict):
    """
    fold 데이터셋 준비 파이프라인.

    LEAKAGE 방지 체크리스트:
      ✓ RadiomicsSelector.fit() → train fold malignant 샘플에서만
      ✓ val fold 는 transform() 만
    """
    print(f"\n  [Fold {fold}] Loading data...")
    tr = load_npz_fold(cfg["data_dir"], fold, "train", cfg)
    va = load_npz_fold(cfg["data_dir"], fold, "val",   cfg)

    # radiomics 추출 (raw)
    print(f"  [Fold {fold}] Extracting radiomics "
          f"(train={len(tr['images'])} / val={len(va['images'])})...")
    tr_radio_raw = prepare_radiomics(tr["images"], tr["polygons"], cfg)
    va_radio_raw = prepare_radiomics(va["images"], va["polygons"], cfg)

    # RadiomicsSelector: fit 은 train malignant only
    sel      = RadiomicsSelector(k=cfg["radio_select_k"])
    tr_radio = sel.fit_transform(tr_radio_raw,
                                  tr["mal_labels"], tr["sub_labels"])
    va_radio = sel.transform(va_radio_raw)
    radio_dim = tr_radio.shape[1]

    tr_ds = OvarianDataset(
        images=tr["images"], polygons=tr["polygons"],
        radio_feats=tr_radio,
        mal_labels=tr["mal_labels"], sub_labels=tr["sub_labels"],
        cfg=cfg,
    )
    va_ds = OvarianDataset(
        images=va["images"], polygons=va["polygons"],
        radio_feats=va_radio,
        mal_labels=va["mal_labels"], sub_labels=va["sub_labels"],
        cfg=cfg,
    )

    print(f"  [Fold {fold}] train={len(tr_ds)}, val={len(va_ds)}, "
          f"radio_dim={radio_dim}")
    return tr_ds, va_ds, radio_dim


print("build_datasets_for_fold loaded ✓")


# ============================================================
# 셀 16: train_single_fold (2-stage)
# ============================================================

def train_single_fold(fold: int, cfg: dict) -> dict:
    """
    2-stage 학습:
      Stage 1: cls_pretrain   (backbone lower frozen + BN guard)
      Stage 2: joint_finetune (전체 low LR)
    """
    device = cfg["device"]
    print(f"\n{'='*60}\n  FOLD {fold}\n{'='*60}")

    tr_ds, va_ds, radio_dim = build_datasets_for_fold(fold, cfg)
    tr_loader, va_loader    = build_dataloaders(tr_ds, va_ds, cfg)

    in_ch = 4 if cfg["use_mask_channel"] else 3
    model = OvarianTwoTaskModel(
        in_channels=in_ch, radio_dim=radio_dim,
        n_subtypes=cfg["n_subtypes"], cfg=cfg,
        pretrained_bb=True,
    ).to(device)

    criterion  = TwoTaskCriterion(cfg).to(device)
    best_score = -1.0
    best_state = None

    # ── Stage 1: cls_pretrain ─────────────────────────────────
    print(f"\n  [Stage 1: cls_pretrain]  epochs={cfg['stage1_epochs']}")
    guard1      = set_stage(model, "cls_pretrain")
    opt1, sch1  = build_optimizer(model, "cls_pretrain", cfg)

    for ep in range(cfg["stage1_epochs"]):
        tr_l = train_one_epoch(model, tr_loader, criterion,
                               opt1, device, bn_guard=guard1)
        va_l, va_m = validate_one_epoch(model, va_loader, criterion, device)
        sch1.step()

        score = va_m["selection_score"]
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Ep{ep+1:3d} | tr={tr_l['total']:.4f}"
                f" va={va_l['total']:.4f}"
                f" | mal_auc={va_m['mal_auc']:.4f}"
                f" sub_f1={va_m['subtype_macro_f1']:.4f}"
                f" sel={score:.4f}"
                + (" ★" if score == best_score else "")
            )

    # ── Stage 2: joint_finetune ───────────────────────────────
    print(f"\n  [Stage 2: joint_finetune]  epochs={cfg['stage2_epochs']}")
    guard2      = set_stage(model, "joint_finetune")
    opt2, sch2  = build_optimizer(model, "joint_finetune", cfg)

    for ep in range(cfg["stage2_epochs"]):
        tr_l = train_one_epoch(model, tr_loader, criterion,
                               opt2, device, bn_guard=guard2)
        va_l, va_m = validate_one_epoch(model, va_loader, criterion, device)
        sch2.step()

        score = va_m["selection_score"]
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())

        if (ep + 1) % 5 == 0 or ep == 0:
            print(
                f"  Ep{ep+1:3d} | tr={tr_l['total']:.4f}"
                f" va={va_l['total']:.4f}"
                f" | mal_auc={va_m['mal_auc']:.4f}"
                f" sub_f1={va_m['subtype_macro_f1']:.4f}"
                f" sel={score:.4f}"
                + (" ★" if score == best_score else "")
            )

    # ── 저장 ──────────────────────────────────────────────────
    spath = os.path.join(cfg["save_dir"], f"fold_{fold}_best.pth")
    torch.save({"state_dict": best_state, "fold": fold,
                "best_score": best_score, "radio_dim": radio_dim}, spath)
    print(f"\n  ✓ Best model saved → {spath}  (sel={best_score:.4f})")

    model.load_state_dict(best_state)
    _, final = validate_one_epoch(model, va_loader, criterion, device)
    final["best_sel_score"] = best_score

    print(f"\n  Fold {fold} Final:")
    for k, v in final.items():
        tag = f"{v:.4f}" if isinstance(v, float) and not math.isnan(v) else "NaN"
        print(f"    {k:25s}: {tag}")

    return final


print("train_single_fold loaded ✓")


# ============================================================
# 셀 17: 5-Fold CV
# ============================================================

def run_5fold_cv(cfg: dict) -> list:
    return [train_single_fold(f, cfg) for f in range(cfg["n_folds"])]


def summarize_cv_results(results: list) -> pd.DataFrame:
    df   = pd.DataFrame(results)
    df.index = [f"fold_{i}" for i in range(len(df))]
    summary  = pd.concat([
        df,
        df.mean(skipna=True).rename("mean").to_frame().T,
        df.std(skipna=True).rename("std").to_frame().T,
    ]).round(4)
    print("\n" + "=" * 65)
    print("  5-Fold CV Summary — Ovarian 2-Task (Image Only)")
    print("=" * 65)
    print(summary.to_string())
    print("=" * 65)
    return summary


print("5-Fold CV / summary loaded ✓")


# ============================================================
# 셀 18: Model Save / Load
# ============================================================

def save_model(model: nn.Module, path: str, meta: dict = None):
    payload = {"state_dict": model.state_dict()}
    if meta:
        payload.update(meta)
    torch.save(payload, path)
    print(f"Saved → {path}")


def load_model(model: nn.Module, path: str, device: str = "cpu"):
    p  = torch.load(path, map_location=device)
    sd = p.get("state_dict", p)
    model.load_state_dict(sd)
    model.to(device)
    print(f"Loaded ← {path}")
    return model


print("Save / Load loaded ✓")


# ============================================================
# 셀 19: FrozenBN Guard 검증
# ============================================================

def verify_frozen_bn_guard(cfg: dict) -> bool:
    """
    FrozenBNGuard 가 frozen BN running_mean 을 실제로 보호하는지 확인.

    검증 절차:
      1) cls_pretrain 세팅 → guard 획득
      2) frozen BN (bn1=features[1]) running_mean 기록
      3) model.train() 호출 → BN.training=True 로 덮임 (문제 재현)
      4) guard.refreeze() → eval() 복원 (해결 적용)
      5) dummy forward (배치 통계 업데이트 시도)
      6) running_mean 불변이면 PASS
    """
    print("\n" + "─" * 55)
    print("  [FrozenBNGuard Verification]")
    print("─" * 55)
    device = cfg["device"]
    C      = 4 if cfg["use_mask_channel"] else 3

    model = OvarianTwoTaskModel(
        in_channels=C, radio_dim=40, n_subtypes=cfg["n_subtypes"],
        cfg=cfg, pretrained_bb=False,
    ).to(device)

    guard     = set_stage(model, "cls_pretrain")
    frozen_bn = model.slice_enc.features[1]         # bn1
    rm_before = frozen_bn.running_mean.clone()

    # model.train() → frozen BN 도 train 으로 덮임 (버그 재현)
    model.train()
    assert frozen_bn.training is True

    # guard.refreeze() → eval 복원 (해결)
    guard.refreeze()
    assert frozen_bn.training is False

    # dummy forward
    B, S, H, W = 2, cfg["n_slices"], cfg["img_h"], cfg["img_w"]
    _ = model(torch.randn(B, S, C, H, W, device=device),
              torch.randn(B, 40, device=device))

    rm_after  = frozen_bn.running_mean.clone()
    protected = torch.allclose(rm_before, rm_after, atol=1e-6)

    if protected:
        print("  PASS : frozen BN running_mean 보호 확인 ✓")
        print("  PASS : model.train() → refreeze() → eval 복원 확인 ✓")
    else:
        print("  FAIL : running_mean 이 변했습니다 ✗")
    print("─" * 55)
    return protected


# ============================================================
# 셀 20: Sanity Check (shape + loss)
# ============================================================

def sanity_check(cfg: dict, radio_dim: int = 40):
    print("\n" + "─" * 55)
    print("  [Sanity Check]")
    print("─" * 55)
    device = cfg["device"]
    B, S   = 2, cfg["n_slices"]
    C      = 4 if cfg["use_mask_channel"] else 3
    H, W   = cfg["img_h"], cfg["img_w"]

    model = OvarianTwoTaskModel(
        in_channels=C, radio_dim=radio_dim,
        n_subtypes=cfg["n_subtypes"], cfg=cfg,
        pretrained_bb=False,
    ).to(device)
    model.eval()
    criterion = TwoTaskCriterion(cfg)

    roi    = torch.randn(B, S, C, H, W, device=device)
    radio  = torch.randn(B, radio_dim, device=device)
    batch  = dict(
        roi_imgs   = roi, radio_feat = radio,
        mal_label  = torch.randint(0, 2, (B,)),
        sub_label  = torch.randint(0, cfg["n_subtypes"], (B,)),
    )

    with torch.no_grad():
        out = model(roi, radio)

    losses = criterion(out, batch)

    print(f"  Input  roi_imgs   : {tuple(roi.shape)}")
    print(f"  Input  radio_feat : {tuple(radio.shape)}")
    print(f"  Output mal_logit  : {tuple(out['mal_logit'].shape)}")
    print(f"  Output sub_logit  : {tuple(out['sub_logit'].shape)}")
    print(f"  Output attn_w     : {tuple(out['attn_w'].shape)}")
    print(f"  L_total={losses['total'].item():.4f}  "
          f"L_mal={losses['mal'].item():.4f}  "
          f"L_sub={losses['sub'].item():.4f}")
    print("  Sanity check PASSED ✓")
    print("─" * 55)


# ── 로드 시 자동 실행 ──────────────────────────────────────────
verify_frozen_bn_guard(CFG)
sanity_check(CFG)


# ============================================================
# 셀 21: 실행 예시
# ============================================================

# ── A: 단일 fold 테스트 ──────────────────────────────────────
# result = train_single_fold(fold=0, cfg=CFG)

# ── B: 전체 5-fold CV ────────────────────────────────────────
# all_results = run_5fold_cv(CFG)
# summary     = summarize_cv_results(all_results)
# summary.to_csv("cv_summary.csv")

# ── C: Grad-CAM 시각화 (학습 완료 후) ────────────────────────
# # 1) 모델 로드
# model = OvarianTwoTaskModel(in_channels=3, radio_dim=40,
#                              n_subtypes=5, cfg=CFG)
# model = load_model(model, "./checkpoints/fold_0_best.pth", CFG["device"])
#
# # 2) 데이터셋 준비 (val set)
# tr_ds, va_ds, radio_dim = build_datasets_for_fold(fold=0, cfg=CFG)
#
# # 3) Malignancy Grad-CAM (환자 idx=0)
# heatmaps = run_gradcam_for_patient(
#     model=model, dataset=va_ds, patient_idx=0,
#     task='mal', save_dir=CFG["gradcam_dir"], cfg=CFG,
# )
#
# # 4) Subtype Grad-CAM (class_idx=None → argmax class)
# heatmaps = run_gradcam_for_patient(
#     model=model, dataset=va_ds, patient_idx=0,
#     task='sub', class_idx=None,
#     save_dir=CFG["gradcam_dir"], cfg=CFG,
# )

# ── D: RadiomicsSelector 단독 테스트 ────────────────────────
# X  = np.random.randn(50, 23).astype(np.float32)
# ml = np.array([1]*30 + [0]*20)
# sl = np.array(list(range(5))*6 + [-1]*20)
# sel = RadiomicsSelector(k=10)
# print("Selected:", sel.fit_transform(X, ml, sl).shape)

print("\nAll cells loaded ✓")
print("Call sanity_check(CFG) or run_5fold_cv(CFG) to proceed.")
