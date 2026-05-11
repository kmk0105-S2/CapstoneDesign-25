import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

# =========================================================
# 1) subtype label 이름
#    네 US subtype 코드의 remap 기준에 맞춰 수정
# =========================================================
SUBTYPE_NAME_MAP = {
    0: "clear_cell",
    1: "endometrioid",
    2: "etc",
    3: "mucinous",
    4: "serous",
}

# =========================================================
# 2) CT binary patient-level probability
#    - ct_model: 네 ct_binary_cv.py 의 load_model()로 로드한 모델
#    - x_ct: shape [S, C, H, W] 또는 [1, S, C, H, W]
# =========================================================
@torch.no_grad()
def predict_ct_malignancy_prob(ct_model, x_ct, device):
    ct_model.eval()

    if isinstance(x_ct, np.ndarray):
        x_ct = torch.from_numpy(x_ct).float()

    if x_ct.dim() == 4:   # [S, C, H, W]
        x_ct = x_ct.unsqueeze(0)
    elif x_ct.dim() != 5:
        raise ValueError(f"x_ct shape must be [S,C,H,W] or [1,S,C,H,W], got {tuple(x_ct.shape)}")

    x_ct = x_ct.to(device, non_blocking=True)
    logit = ct_model(x_ct)                  # [1, 1] 또는 [1]
    prob = torch.sigmoid(logit).view(-1)[0].item()
    pred = int(prob >= 0.5)

    return {
        "mal_logit": float(logit.view(-1)[0].item()),
        "mal_prob": float(prob),
        "mal_pred": pred,   # 0=benign, 1=malignant
    }

# =========================================================
# 3) US subtype prediction
#    - us_model: 네 US subtype-only 코드의 load_model()로 로드한 모델
#    - x_us: shape [S, C, H, W] 또는 [1, S, C, H, W]
# =========================================================
@torch.no_grad()
def predict_us_subtype(us_model, x_us, device, subtype_name_map=SUBTYPE_NAME_MAP):
    us_model.eval()

    if isinstance(x_us, np.ndarray):
        x_us = torch.from_numpy(x_us).float()

    if x_us.dim() == 4:   # [S, C, H, W]
        x_us = x_us.unsqueeze(0)
    elif x_us.dim() != 5:
        raise ValueError(f"x_us shape must be [S,C,H,W] or [1,S,C,H,W], got {tuple(x_us.shape)}")

    x_us = x_us.to(device, non_blocking=True)
    logits = us_model(x_us)                 # [1, K]
    probs = torch.softmax(logits, dim=1)
    pred_idx = int(torch.argmax(probs, dim=1).item())

    return {
        "sub_logits": logits.squeeze(0).detach().cpu().numpy(),
        "sub_probs": probs.squeeze(0).detach().cpu().numpy(),
        "sub_pred_idx": pred_idx,
        "sub_pred_name": subtype_name_map.get(pred_idx, f"class_{pred_idx}")
    }

# =========================================================
# 4) 최종 cascade inference (1명 기준)
#    - CT가 benign이면 끝
#    - CT가 malignant이면 US subtype 결과 사용
# =========================================================
@torch.no_grad()
def cascade_infer_one_patient(
    ct_model,
    us_model,
    x_ct,
    x_us,
    device,
    ct_threshold=0.5,
    subtype_name_map=SUBTYPE_NAME_MAP
):
    ct_model.eval()
    us_model.eval()

    ct_out = predict_ct_malignancy_prob(ct_model, x_ct, device)
    mal_prob = ct_out["mal_prob"]
    mal_pred = int(mal_prob >= ct_threshold)

    result = {
        "mal_logit": ct_out["mal_logit"],
        "mal_prob": mal_prob,
        "mal_pred": mal_pred,
        "final_label": None,
        "final_subtype_idx": None,
        "final_subtype_name": None,
        "sub_probs": None,
    }

    if mal_pred == 0:
        result["final_label"] = "benign"
        return result

    us_out = predict_us_subtype(us_model, x_us, device, subtype_name_map)
    result["final_label"] = "malignant"
    result["final_subtype_idx"] = us_out["sub_pred_idx"]
    result["final_subtype_name"] = us_out["sub_pred_name"]
    result["sub_probs"] = us_out["sub_probs"]

    return result

# =========================================================
# 5) patient_id 기준 매칭용 helper
#    ct_dict, us_dict 예시:
#      ct_dict[patient_id] = x_ct
#      us_dict[patient_id] = x_us
# =========================================================
def build_common_patient_ids(ct_dict, us_dict):
    ct_ids = set(ct_dict.keys())
    us_ids = set(us_dict.keys())
    common_ids = sorted(list(ct_ids & us_ids))
    only_ct = sorted(list(ct_ids - us_ids))
    only_us = sorted(list(us_ids - ct_ids))

    print(f"Common patients: {len(common_ids)}")
    print(f"Only CT: {len(only_ct)}")
    print(f"Only US: {len(only_us)}")

    return common_ids, only_ct, only_us

# =========================================================
# 6) 여러 환자 batch-like inference
#    - ct_inputs_by_pid: {patient_id: x_ct}
#    - us_inputs_by_pid: {patient_id: x_us}
# =========================================================
def run_cascade_for_patients(
    ct_model,
    us_model,
    ct_inputs_by_pid,
    us_inputs_by_pid,
    device,
    ct_threshold=0.5,
    subtype_name_map=SUBTYPE_NAME_MAP
):
    common_ids, only_ct, only_us = build_common_patient_ids(ct_inputs_by_pid, us_inputs_by_pid)

    rows = []
    for pid in common_ids:
        out = cascade_infer_one_patient(
            ct_model=ct_model,
            us_model=us_model,
            x_ct=ct_inputs_by_pid[pid],
            x_us=us_inputs_by_pid[pid],
            device=device,
            ct_threshold=ct_threshold,
            subtype_name_map=subtype_name_map
        )

        row = {
            "patient_id": pid,
            "mal_prob": out["mal_prob"],
            "mal_pred": out["mal_pred"],
            "final_label": out["final_label"],
            "final_subtype_idx": out["final_subtype_idx"],
            "final_subtype_name": out["final_subtype_name"],
        }

        if out["sub_probs"] is not None:
            for k, v in enumerate(out["sub_probs"]):
                row[f"sub_prob_{k}"] = float(v)

        rows.append(row)

    df = pd.DataFrame(rows)
    return df, common_ids, only_ct, only_us

# =========================================================
# 7) 선택: fold별 best model 로드
#    - CT: 네 기존 load_model(model_type, model_path, device) 그대로 사용
#    - US: 아래처럼 us_build_fn 기반 load_model_us 사용
# =========================================================
def load_model_us(us_build_fn, model_path, device):
    model = us_build_fn()
    payload = torch.load(model_path, map_location=device)

    # state_dict 래핑 저장 / 순수 state_dict 둘 다 허용
    state_dict = payload.get("state_dict", payload) if isinstance(payload, dict) else payload
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

# =========================================================
# 8) 예시 사용법
# =========================================================
"""
# --- CT model 로드 (네 기존 코드 그대로) ---
ct_model = load_model(
    model_type="resnet",                     # or "vit", "hybrid"
    model_path="./ct_binary_models/ResNet50_fold1_best.pth",
    device=device
)

# --- US model 로드 ---
# us_build_fn 은 네 US subtype 코드의 build_resnet50_patient_model 같은 함수
us_model = load_model_us(
    us_build_fn=lambda: build_resnet50_patient_model(load_pretrained=True),
    model_path="./us_subtype_models/ResNet50_fold1_best.pth",
    device=device
)

# --- 1명 inference ---
# x_ct: [S, C, H, W]
# x_us: [S, C, H, W]
out = cascade_infer_one_patient(
    ct_model=ct_model,
    us_model=us_model,
    x_ct=x_ct,
    x_us=x_us,
    device=device,
    ct_threshold=0.5
)
print(out)

# --- 여러 환자 inference ---
# ct_inputs_by_pid = {"P001": x_ct_1, "P002": x_ct_2, ...}
# us_inputs_by_pid = {"P001": x_us_1, "P002": x_us_2, ...}

df_result, common_ids, only_ct, only_us = run_cascade_for_patients(
    ct_model=ct_model,
    us_model=us_model,
    ct_inputs_by_pid=ct_inputs_by_pid,
    us_inputs_by_pid=us_inputs_by_pid,
    device=device,
    ct_threshold=0.5
)

print(df_result.head())
df_result.to_csv("cascade_inference_results.csv", index=False)
"""