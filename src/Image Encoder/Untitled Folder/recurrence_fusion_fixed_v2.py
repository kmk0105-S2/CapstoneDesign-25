#!/usr/bin/env python
# coding: utf-8
"""
recurrence_fusion_fixed_v2.py

[핵심 수정 사항]
- WeightedContrastiveLoss: true weighted mean + obs_ratio detach
- Phase 2: clinical encoder / projection head LR 분리
- Recurrence loss: malignant 샘플에만 적용
- Phase 3: epoch마다 중복 unfreeze 제거
- best checkpoint: 실제 validation metric 기준으로 저장
- w_rec 실제 반영
- zero-loss fallback: graph-safe zero scalar
- CT branch optional(use_ct)
- config 저장 로직 개선(config.__dict__ 대신 전체 필드 저장)
- dummy fold loader도 use_ct=True 케이스를 지원하도록 개선
- best_epoch 저장 추가

[주의]
- make_fold_loaders()는 여전히 더미 구현체다.
- 실제 실험에서는 반드시 환자 단위 split / fold leakage 점검 필요.
"""

from __future__ import annotations

import gc
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch import Tensor


# =============================================================
# 0. Config
# =============================================================
@dataclass
class Config:
    # modality
    use_ct: bool = False

    # encoder dims
    us_feat_dim: int = 512
    ct_feat_dim: int = 512
    clin_feat_dim: int = 128
    img_fusion_dim: int = 512

    # projection heads
    proj_dim: int = 128
    proj_hidden: int = 256

    # classification heads
    num_subtypes: int = 5
    hidden_cls: int = 256

    # survival head
    survival_hidden: int = 256

    # contrastive
    temperature: float = 0.07
    min_obs_weight: float = 0.2

    # clinical MLP
    clin_input_dim: int = 20
    clin_hidden: int = 64

    # regularization
    dropout: float = 0.1
    weight_decay: float = 1e-4
    use_amp: bool = True

    # learning rates
    lr_phase1: float = 1e-4
    lr_phase2_proj: float = 1e-4
    lr_phase2_clin: float = 5e-5
    lr_phase3_enc: float = 1e-5
    lr_phase3_head: float = 1e-4

    # loss weights
    w_mal: float = 1.0
    w_sub: float = 0.5
    w_rec: float = 1.0

    # epochs
    epochs_phase1: int = 30
    epochs_phase2: int = 10
    epochs_phase3: int = 20

    # cv
    folds: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 5])
    output_dir: str = "./cv_outputs_fixed_v2"
    best_metric: str = "recurrence_auc"  # fallback to val_loss if NaN


cfg = Config()


# =============================================================
# 1. Encoders
# =============================================================
class USEncoder(nn.Module):
    """US encoder. 입력 [B, S, C, H, W] -> 출력 [B, us_feat_dim]"""

    def __init__(self, feat_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, feat_dim),
        )
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, s, c, h, w = x.shape
        feat = self.backbone(x.reshape(b * s, c, h, w)).reshape(b, s, -1)
        weight = torch.softmax(self.attn(feat), dim=1)
        return self.drop((feat * weight).sum(dim=1))


class CTEncoder(nn.Module):
    """CT encoder. 현재는 US encoder와 동일 구조 예시."""

    def __init__(self, feat_dim: int, dropout: float):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(8),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, feat_dim),
        )
        self.attn = nn.Sequential(
            nn.Linear(feat_dim, feat_dim // 2),
            nn.Tanh(),
            nn.Linear(feat_dim // 2, 1),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        b, s, c, h, w = x.shape
        feat = self.backbone(x.reshape(b * s, c, h, w)).reshape(b, s, -1)
        weight = torch.softmax(self.attn(feat), dim=1)
        return self.drop((feat * weight).sum(dim=1))


class ImageFusionModule(nn.Module):
    """US-only 또는 US+CT fusion -> z_img"""

    def __init__(self, us_dim: int, ct_dim: int, fusion_dim: int, dropout: float):
        super().__init__()
        self.us_proj = nn.Sequential(
            nn.Linear(us_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout)
        )
        self.us_ct_proj = nn.Sequential(
            nn.Linear(us_dim + ct_dim, fusion_dim), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward(self, z_us: Tensor, z_ct: Optional[Tensor] = None) -> Tensor:
        if z_ct is None:
            return self.us_proj(z_us)
        return self.us_ct_proj(torch.cat([z_us, z_ct], dim=-1))


class ClinicalEncoder(nn.Module):
    """전처리된 clinical tensor를 small MLP로 인코딩."""

    def __init__(self, input_dim: int, hidden: int, feat_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim),
        )
        self.norm = nn.LayerNorm(feat_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.net(x))


# =============================================================
# 2. Heads
# =============================================================
class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return F.normalize(self.net(x), dim=-1)


class MalignancyHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z).squeeze(-1)


class SubtypeHead(nn.Module):
    def __init__(self, in_dim: int, hidden: int, num_classes: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.net(z)


class SurvivalHead(nn.Module):
    """Binary recurrence head. z_img + z_clin -> recurrence logit"""

    def __init__(self, img_dim: int, clin_dim: int, hidden: int, dropout: float):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(img_dim + clin_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z_img: Tensor, z_clin: Tensor) -> Tensor:
        return self.fusion(torch.cat([z_img, z_clin], dim=-1)).squeeze(-1)


# =============================================================
# 3. Full model
# =============================================================
class MultimodalOvarianModel(nn.Module):
    def __init__(self, config: Config = cfg):
        super().__init__()
        c = config
        self.use_ct_branch = c.use_ct

        self.us_encoder = USEncoder(c.us_feat_dim, c.dropout)
        self.ct_encoder = CTEncoder(c.ct_feat_dim, c.dropout)
        self.image_fusion = ImageFusionModule(c.us_feat_dim, c.ct_feat_dim, c.img_fusion_dim, c.dropout)
        self.clin_encoder = ClinicalEncoder(c.clin_input_dim, c.clin_hidden, c.clin_feat_dim, c.dropout)

        self.img_proj = ProjectionHead(c.img_fusion_dim, c.proj_hidden, c.proj_dim, c.dropout)
        self.clin_proj = ProjectionHead(c.clin_feat_dim, c.proj_hidden, c.proj_dim, c.dropout)

        self.mal_head = MalignancyHead(c.img_fusion_dim, c.hidden_cls, c.dropout)
        self.subtype_head = SubtypeHead(c.img_fusion_dim, c.hidden_cls, c.num_subtypes, c.dropout)
        self.survival_head = SurvivalHead(c.img_fusion_dim, c.clin_feat_dim, c.survival_hidden, c.dropout)

    def forward_features(
        self,
        us: Tensor,
        ct: Optional[Tensor] = None,
        clinical: Optional[Tensor] = None,
    ) -> Dict[str, Optional[Tensor]]:
        z_us = self.us_encoder(us)
        z_ct = self.ct_encoder(ct) if (self.use_ct_branch and ct is not None) else None
        z_img = self.image_fusion(z_us, z_ct)
        z_clin = self.clin_encoder(clinical) if clinical is not None else None
        return {"z_us": z_us, "z_ct": z_ct, "z_img": z_img, "z_clin": z_clin}

    def forward_phase1_classification(self, us: Tensor, ct: Optional[Tensor] = None) -> Dict[str, Tensor]:
        feats = self.forward_features(us, ct, clinical=None)
        z_img = feats["z_img"]
        return {
            "mal_logit": self.mal_head(z_img),
            "subtype_logit": self.subtype_head(z_img),
            "z_img": z_img,
        }

    def forward_phase2_contrastive(self, us: Tensor, ct: Optional[Tensor], clinical: Tensor) -> Dict[str, Tensor]:
        with torch.no_grad():
            feats = self.forward_features(us, ct, clinical=None)
            z_img = feats["z_img"]
        z_clin = self.clin_encoder(clinical)
        return {
            "img_proj_emb": self.img_proj(z_img),
            "clin_proj_emb": self.clin_proj(z_clin),
            "z_clin": z_clin,
        }

    def forward_phase3_recurrence(self, us: Tensor, ct: Optional[Tensor], clinical: Tensor) -> Dict[str, Tensor]:
        feats = self.forward_features(us, ct, clinical)
        z_img, z_clin = feats["z_img"], feats["z_clin"]
        return {
            "recurrence_logit": self.survival_head(z_img, z_clin),
            "z_img": z_img,
            "z_clin": z_clin,
        }


# =============================================================
# 4. Freeze / unfreeze helpers
# =============================================================
def freeze_module(module: nn.Module) -> None:
    module.eval()
    for p in module.parameters():
        p.requires_grad = False


def unfreeze_module(module: nn.Module) -> None:
    module.train()
    for p in module.parameters():
        p.requires_grad = True


def freeze_image_encoders(model: MultimodalOvarianModel) -> None:
    freeze_module(model.us_encoder)
    if model.use_ct_branch:
        freeze_module(model.ct_encoder)
    freeze_module(model.image_fusion)


def unfreeze_all(model: MultimodalOvarianModel) -> None:
    for module in model.children():
        unfreeze_module(module)


# =============================================================
# 5. Optimizers
# =============================================================
def _image_encoder_params(model: MultimodalOvarianModel):
    params = list(model.us_encoder.parameters()) + list(model.image_fusion.parameters())
    if model.use_ct_branch:
        params += list(model.ct_encoder.parameters())
    return params


def build_optimizer_phase1(model: MultimodalOvarianModel, config: Config = cfg) -> torch.optim.Optimizer:
    params = _image_encoder_params(model) + list(model.mal_head.parameters()) + list(model.subtype_head.parameters())
    return torch.optim.Adam(params, lr=config.lr_phase1, weight_decay=config.weight_decay)


def build_optimizer_phase2(model: MultimodalOvarianModel, config: Config = cfg) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        [
            {
                "params": list(model.img_proj.parameters()) + list(model.clin_proj.parameters()),
                "lr": config.lr_phase2_proj,
            },
            {
                "params": list(model.clin_encoder.parameters()),
                "lr": config.lr_phase2_clin,
            },
        ],
        weight_decay=config.weight_decay,
    )


def build_optimizer_phase3(model: MultimodalOvarianModel, config: Config = cfg) -> torch.optim.Optimizer:
    encoder_params = _image_encoder_params(model) + list(model.clin_encoder.parameters())
    return torch.optim.Adam(
        [
            {"params": encoder_params, "lr": config.lr_phase3_enc},
            {"params": list(model.survival_head.parameters()), "lr": config.lr_phase3_head},
        ],
        weight_decay=config.weight_decay,
    )


# =============================================================
# 6. Losses
# =============================================================
def zero_loss_from(logit: Tensor) -> Tensor:
    """Backward-safe zero scalar."""
    return logit.sum() * 0.0


def compute_malignancy_loss(logit: Tensor, target: Tensor) -> Tensor:
    return F.binary_cross_entropy_with_logits(logit, target.float())


def compute_subtype_loss(logit: Tensor, target: Tensor, mal_mask: Tensor) -> Tensor:
    if mal_mask.sum().item() == 0:
        return zero_loss_from(logit)
    return F.cross_entropy(logit[mal_mask], target[mal_mask].long())


def compute_recurrence_loss(logit: Tensor, target: Tensor, mal_mask: Tensor) -> Tensor:
    if mal_mask.sum().item() == 0:
        return zero_loss_from(logit)
    return F.binary_cross_entropy_with_logits(logit[mal_mask], target[mal_mask].float())


class WeightedContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = cfg.temperature, min_obs_weight: float = cfg.min_obs_weight):
        super().__init__()
        self.temperature = temperature
        self.min_obs_weight = min_obs_weight

    def forward(self, img_emb: Tensor, clin_emb: Tensor, obs_ratio: Tensor) -> Tensor:
        batch_size = img_emb.size(0)
        sim = (img_emb @ clin_emb.T) / self.temperature
        labels = torch.arange(batch_size, device=sim.device)

        loss_i2c = F.cross_entropy(sim, labels, reduction="none")
        loss_c2i = F.cross_entropy(sim.T, labels, reduction="none")

        weight = obs_ratio.detach().float().clamp(min=self.min_obs_weight, max=1.0)
        weight_sum = weight.sum().clamp_min(1e-8)

        i2c = (weight * loss_i2c).sum() / weight_sum
        c2i = (weight * loss_c2i).sum() / weight_sum
        return 0.5 * (i2c + c2i)


# =============================================================
# 7. Train / eval loops
# =============================================================
def amp_autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type="cuda", enabled=enabled and device.type == "cuda")


def train_one_epoch_phase1(
    model: MultimodalOvarianModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config = cfg,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    model.us_encoder.train()
    if model.use_ct_branch:
        model.ct_encoder.train()
    model.image_fusion.train()
    model.mal_head.train()
    model.subtype_head.train()

    totals = {"loss": 0.0, "mal": 0.0, "sub": 0.0}
    n = 0

    for batch in loader:
        us = batch["us"].to(device)
        ct = batch["ct"].to(device) if batch.get("ct") is not None else None
        y_mal = batch["y_malignant"].to(device).float()
        y_sub = batch["y_subtype"].to(device).long()
        mal_mask = y_mal == 1

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, config.use_amp):
            out = model.forward_phase1_classification(us, ct)
            loss_mal = compute_malignancy_loss(out["mal_logit"], y_mal)
            loss_sub = compute_subtype_loss(out["subtype_logit"], y_sub, mal_mask)
            loss = config.w_mal * loss_mal + config.w_sub * loss_sub

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        totals["loss"] += float(loss.item())
        totals["mal"] += float(loss_mal.item())
        totals["sub"] += float(loss_sub.item())
        n += 1

    return {k: v / max(n, 1) for k, v in totals.items()}


def train_one_epoch_phase2(
    model: MultimodalOvarianModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config = cfg,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.img_proj.train()
    model.clin_encoder.train()
    model.clin_proj.train()
    criterion = WeightedContrastiveLoss(config.temperature, config.min_obs_weight)

    total_loss = 0.0
    n = 0

    for batch in loader:
        us = batch["us"].to(device)
        ct = batch["ct"].to(device) if batch.get("ct") is not None else None
        clinical = batch["clinical"].to(device)
        obs_ratio = batch["obs_ratio"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, config.use_amp):
            out = model.forward_phase2_contrastive(us, ct, clinical)
            loss = criterion(out["img_proj_emb"], out["clin_proj_emb"], obs_ratio)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n += 1

    return {"loss": total_loss / max(n, 1)}


def train_one_epoch_phase3(
    model: MultimodalOvarianModel,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    config: Config = cfg,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    n = 0

    for batch in loader:
        us = batch["us"].to(device)
        ct = batch["ct"].to(device) if batch.get("ct") is not None else None
        clinical = batch["clinical"].to(device)
        y_recur = batch["y_recurrence"].to(device).float()
        y_mal = batch["y_malignant"].to(device).float()
        mal_mask = y_mal == 1

        optimizer.zero_grad(set_to_none=True)
        with amp_autocast(device, config.use_amp):
            out = model.forward_phase3_recurrence(us, ct, clinical)
            loss_rec = compute_recurrence_loss(out["recurrence_logit"], y_recur, mal_mask)
            loss = config.w_rec * loss_rec

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        n += 1

    return {"loss": total_loss / max(n, 1)}


@torch.no_grad()
def evaluate_phase3(
    model: MultimodalOvarianModel,
    loader,
    device: torch.device,
    config: Config = cfg,
) -> Dict[str, float]:
    model.eval()

    all_logits, all_targets, all_masks = [], [], []
    total_loss = 0.0
    n = 0

    for batch in loader:
        us = batch["us"].to(device)
        ct = batch["ct"].to(device) if batch.get("ct") is not None else None
        clinical = batch["clinical"].to(device)
        y_recur = batch["y_recurrence"].to(device).float()
        y_mal = batch["y_malignant"].to(device).float()
        mal_mask = y_mal == 1

        with amp_autocast(device, config.use_amp):
            out = model.forward_phase3_recurrence(us, ct, clinical)
            loss = config.w_rec * compute_recurrence_loss(out["recurrence_logit"], y_recur, mal_mask)

        total_loss += float(loss.item())
        all_logits.append(out["recurrence_logit"].detach().cpu())
        all_targets.append(y_recur.detach().cpu())
        all_masks.append(mal_mask.detach().cpu())
        n += 1

    logits = torch.cat(all_logits).numpy()
    targets = torch.cat(all_targets).numpy().astype(int)
    masks = torch.cat(all_masks).numpy().astype(bool)

    logits = logits[masks]
    targets = targets[masks]

    metrics = {
        "val_loss": total_loss / max(n, 1),
        "recurrence_auc": float("nan"),
        "recurrence_f1": float("nan"),
        "recurrence_acc": float("nan"),
    }
    if len(targets) == 0 or len(np.unique(targets)) < 2:
        return metrics

    probs = torch.sigmoid(torch.as_tensor(logits)).numpy()
    preds = (probs >= 0.5).astype(int)
    metrics["recurrence_auc"] = float(roc_auc_score(targets, probs))
    metrics["recurrence_f1"] = float(f1_score(targets, preds, average="macro", zero_division=0))
    metrics["recurrence_acc"] = float(accuracy_score(targets, preds))
    return metrics


# =============================================================
# 8. Checkpoint helpers
# =============================================================
def select_score(metrics: Dict[str, float], best_metric: str = cfg.best_metric) -> float:
    value = metrics.get(best_metric, float("nan"))
    if value is None or np.isnan(value):
        return -float(metrics["val_loss"])  # lower val_loss is better
    return float(value)


def config_to_dict(config: Config) -> Dict[str, object]:
    return asdict(config)


def save_checkpoint(
    path: Path,
    model: nn.Module,
    fold_idx: int,
    epoch: int,
    metrics: Dict[str, float],
    config: Config,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {
        "fold_idx": fold_idx,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "metrics": metrics,
        "config": config_to_dict(config),
    }
    torch.save(state, path)


# =============================================================
# 9. One fold
# =============================================================
def run_one_fold(
    fold_idx: int,
    train_loader,
    val_loader,
    device: torch.device,
    config: Config = cfg,
) -> Dict[str, float]:
    model = MultimodalOvarianModel(config).to(device)
    scaler = torch.cuda.amp.GradScaler() if (config.use_amp and torch.cuda.is_available()) else None

    print(f"\n  [Fold {fold_idx}] Phase 1: Image encoder + classification")
    opt1 = build_optimizer_phase1(model, config)
    for ep in range(1, config.epochs_phase1 + 1):
        m = train_one_epoch_phase1(model, train_loader, opt1, device, config, scaler)
        print(f"    [P1 Ep {ep:02d}] loss={m['loss']:.4f}  mal={m['mal']:.4f}  sub={m['sub']:.4f}")

    print(f"\n  [Fold {fold_idx}] Phase 2: Contrastive alignment")
    freeze_image_encoders(model)
    opt2 = build_optimizer_phase2(model, config)
    for ep in range(1, config.epochs_phase2 + 1):
        m = train_one_epoch_phase2(model, train_loader, opt2, device, config, scaler)
        print(f"    [P2 Ep {ep:02d}] contrastive_loss={m['loss']:.4f}")

    print(f"\n  [Fold {fold_idx}] Phase 3: Recurrence fine-tuning")
    unfreeze_all(model)
    opt3 = build_optimizer_phase3(model, config)

    output_dir = Path(config.output_dir)
    best_ckpt_path = output_dir / f"fold{fold_idx}_best.pth"
    last_ckpt_path = output_dir / f"fold{fold_idx}_last.pth"

    best_score = -float("inf")
    best_metrics: Optional[Dict[str, float]] = None
    best_epoch = -1

    for ep in range(1, config.epochs_phase3 + 1):
        train_metrics = train_one_epoch_phase3(model, train_loader, opt3, device, config, scaler)
        val_metrics = evaluate_phase3(model, val_loader, device, config)
        score = select_score(val_metrics, config.best_metric)

        print(
            f"    [P3 Ep {ep:02d}] train_loss={train_metrics['loss']:.4f}  "
            f"val_loss={val_metrics['val_loss']:.4f}  "
            f"AUC={val_metrics['recurrence_auc']:.4f}  "
            f"F1={val_metrics['recurrence_f1']:.4f}  "
            f"Acc={val_metrics['recurrence_acc']:.4f}"
        )

        if score > best_score:
            best_score = score
            best_metrics = dict(val_metrics)
            best_epoch = ep
            save_checkpoint(best_ckpt_path, model, fold_idx, ep, val_metrics, config)

    final_metrics = evaluate_phase3(model, val_loader, device, config)
    save_checkpoint(last_ckpt_path, model, fold_idx, config.epochs_phase3, final_metrics, config)

    if best_metrics is None:
        best_metrics = final_metrics
        best_epoch = config.epochs_phase3

    print(
        f"\n  [Fold {fold_idx}] === Best Validation (epoch {best_epoch}) ===\n"
        f"    val_loss={best_metrics['val_loss']:.4f}  "
        f"AUC={best_metrics['recurrence_auc']:.4f}  "
        f"F1={best_metrics['recurrence_f1']:.4f}  "
        f"Acc={best_metrics['recurrence_acc']:.4f}"
    )
    print(f"  [Fold {fold_idx}] Saved best -> {best_ckpt_path}")
    print(f"  [Fold {fold_idx}] Saved last -> {last_ckpt_path}")

    del model, opt1, opt2, opt3
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return best_metrics


# =============================================================
# 10. CV loop
# =============================================================
def make_fold_loaders(fold_idx: int, config: Config = cfg):
    """
    실제 사용 시 반드시 교체할 것.
    현재는 구조 검증용 더미 DataLoader만 반환.
    """
    from torch.utils.data import DataLoader, TensorDataset

    batch_size, num_slices = 16, 8

    def make_dummy_dataset(n: int):
        us = torch.randn(n, num_slices, 3, 64, 64)
        ct = torch.randn(n, num_slices, 3, 64, 64) if config.use_ct else torch.zeros(n, num_slices, 3, 64, 64)
        clinical = torch.randn(n, config.clin_input_dim)
        obs_ratio = torch.rand(n)
        y_mal = torch.randint(0, 2, (n,)).float()
        y_sub = torch.randint(0, config.num_subtypes, (n,))
        y_rec = torch.randint(0, 2, (n,)).float()
        return TensorDataset(us, ct, clinical, obs_ratio, y_mal, y_sub, y_rec)

    def collate(samples):
        us, ct, clin, obs, y_mal, y_sub, y_rec = zip(*samples)
        return {
            "us": torch.stack(us),
            "ct": torch.stack(ct) if config.use_ct else None,
            "clinical": torch.stack(clin),
            "obs_ratio": torch.stack(obs),
            "y_malignant": torch.stack(y_mal),
            "y_subtype": torch.stack(y_sub),
            "y_recurrence": torch.stack(y_rec),
        }

    train_ds = make_dummy_dataset(64)
    val_ds = make_dummy_dataset(16)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, collate_fn=collate)
    return train_loader, val_loader


def run_cross_validation(config: Config = cfg, device: Optional[torch.device] = None) -> List[Dict[str, float]]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print(f"5-Fold Cross Validation | device: {device} | use_ct={config.use_ct}")
    print("=" * 60)

    all_fold_metrics: List[Dict[str, float]] = []

    for fold_idx in config.folds:
        print("\n" + "=" * 60)
        print(f"FOLD {fold_idx} / {len(config.folds)}")
        print("=" * 60)

        train_loader, val_loader = make_fold_loaders(fold_idx, config)
        fold_metrics = run_one_fold(fold_idx, train_loader, val_loader, device, config)
        all_fold_metrics.append(fold_metrics)
        print(f"\n  [Fold {fold_idx}] Done. Metrics: {fold_metrics}")

    print("\n" + "=" * 60)
    print("5-Fold CV Summary")
    print("=" * 60)

    metric_keys = ["val_loss", "recurrence_auc", "recurrence_f1", "recurrence_acc"]
    for key in metric_keys:
        vals = [m[key] for m in all_fold_metrics if not np.isnan(m[key])]
        if vals:
            mean = float(np.mean(vals))
            std = float(np.std(vals))
            print(f"  {key:<20}: {mean:.4f} ± {std:.4f}")
        else:
            print(f"  {key:<20}: N/A")

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / "cv_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(all_fold_metrics, f, indent=2, ensure_ascii=False)
    print(f"\nCV results saved -> {result_path}")
    return all_fold_metrics


# =============================================================
# 11. Smoke test
# =============================================================
if __name__ == "__main__":
    print("Running smoke test...")
    cfg.use_ct = False
    cfg.epochs_phase1 = 1
    cfg.epochs_phase2 = 1
    cfg.epochs_phase3 = 1
    cfg.folds = [1, 2]
    cfg.use_amp = False

    run_cross_validation(cfg)
    print("\nSmoke test passed.")
