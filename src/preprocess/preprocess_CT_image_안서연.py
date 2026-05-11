#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import re
import json
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np

from PIL import Image, ImageFilter

from skimage import exposure
from skimage import filters
from skimage.draw import polygon as sk_polygon
from skimage.transform import resize as sk_resize
from skimage.util import img_as_ubyte, img_as_float, random_noise
from scipy.ndimage import gaussian_filter

import matplotlib.pyplot as plt
import matplotlib.patches as patches

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import torchvision.transforms as T
import torchvision.transforms.functional as F

from tqdm import tqdm

warnings.filterwarnings("ignore")


# In[2]:


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# In[3]:


BASE_DIR = Path('/tf/nasw/dataset001/')

TRAIN_SRC = BASE_DIR / 'Training' / '01.원천데이터'
TRAIN_LBL = BASE_DIR / 'Training' / '02.라벨링데이터'
VAL_SRC = BASE_DIR / 'Validation' / '01.원천데이터'
VAL_LBL = BASE_DIR / 'Validation' / '02.라벨링데이터'
EMR_CANCER = BASE_DIR / 'Other' / '메타데이터' / '암' / 'EMR'
EMR_BENIGN = BASE_DIR / 'Other' / '메타데이터' / '양성종양' / 'EMR'

CT_SUBDIR = 'CT' / Path('골반')

CLASSES = {'양성종양': 0, '암': 1}

OUTPUT_DIR = BASE_DIR / 'preprocessed'
SPLIT_META_DIR = OUTPUT_DIR / 'split_metadata'
NPZ_DIR_CT = OUTPUT_DIR / 'npz_ct'
NPZ_DIR_CT_AUGMENTED = OUTPUT_DIR / 'npz_ct_augmented'
NPZ_DIR_CT_AUGMENTED_V2 = OUTPUT_DIR / 'npz_ct_augmented_v2'
NPZ_DIR_CT_NO_3CHANNEL = OUTPUT_DIR / 'npz_ct_no_3channel'

for d in [OUTPUT_DIR, SPLIT_META_DIR, NPZ_DIR_CT, NPZ_DIR_CT_AUGMENTED, NPZ_DIR_CT_AUGMENTED_V2, NPZ_DIR_CT_NO_3CHANNEL]:
    d.mkdir(parents=True, exist_ok=True)


# In[4]:


def explore_patients(src_dir, lbl_dir, class_name):
    """
    한 클래스의 골반 CT 데이터를 탐색하여
    환자별 slice 수 반환

    Returns
    -------
    dict : {patient_id: {'images': [Path, ...], 'label': [Path, ...]}}
    """

    img_dir = src_dir / class_name / CT_SUBDIR
    lbl_dir = lbl_dir / class_name / CT_SUBDIR

    patient_data = defaultdict(lambda: {'images': [], 'labels': []})

    # collect images
    for img_path in sorted(img_dir.glob('*.png')):
        # extract patient_id from filename
        patient_id = img_path.stem.split('_')[0]
        patient_data[patient_id]['images'].append(img_path)

    # collect labels
    for lbl_path in sorted(lbl_dir.glob('*.json')):
        # extract patient_id from filename
        patient_id = lbl_path.stem.split('_')[0]
        patient_data[patient_id]['labels'].append(lbl_path)

    return dict(patient_data)


# In[5]:


# Integrate Training + Validation (for patient-level split)
all_patients = {}

for class_name, label in CLASSES.items():
    train_data = explore_patients(TRAIN_SRC, TRAIN_LBL, class_name)
    val_data = explore_patients(VAL_SRC, VAL_LBL, class_name)

    # merge by patient
    merged = defaultdict(lambda: {'images': [], 'labels': []})
    for pid, data in train_data.items():
        merged[pid]['images'].extend(data['images'])
        merged[pid]['labels'].extend(data['labels'])
    for pid, data in val_data.items():
        merged[pid]['images'].extend(data['images'])
        merged[pid]['labels'].extend(data['labels'])

    for pid, data in merged.items():
        all_patients[pid] = {
            'images': data['images'],
            'labels': data['labels'],
            'class_name': class_name,
            'label': label,
        }

    slice_counts = [len(v['images']) for v in merged.values()]

    print(f"[{class_name} (label={label})]")
    print(f" Total patients : {len(merged)} patients")
    print(f" Total slices : {sum(slice_counts)} slices")
    print(f" Average slice per patient : {np.mean(slice_counts):.1f} slices")
    print(f" Minimum slices : {min(slice_counts)} slices")
    print(f" Maximum slices : {max(slice_counts)} slices")
    print(f" Less than 8 slices : {sum(1 for c in slice_counts if c < 8)} patients")
    print(f" More than 8 slices : {sum(1 for c in slice_counts if c > 8)} patients")
    print(f" Exactly 8 slices : {sum(1 for c in slice_counts if c == 8)} patients")

print(f"=> Entire patients : {len(all_patients)} patients")


# In[6]:


sample_emr_path = list(EMR_CANCER.glob('*.json'))[0]

with open(sample_emr_path, 'r', encoding='utf-8') as f:
    sample_emr = json.load(f)

print(f"{sample_emr_path.name}")
print(f" \n key list and value:")
for key, val in sample_emr.items():
    print(f" - {key}: {val}")


# In[7]:


sample_lbl = all_patients['10008']['labels'][0]

with open(sample_lbl, 'r', encoding='utf-8') as f:
    sample_json = json.load(f)

print(f"{sample_lbl.name}")
print(f" \n key list:")
for key in sample_json.keys():
    print(f" - {key}: {type(sample_json[key]).__name__}")

print(f"Entire structure:")
print(json.dumps(sample_json, indent=2, ensure_ascii=False))


# In[8]:


# Scan all Tumor types (value) 
all_values = set()

for pid, data in tqdm(all_patients.items(), desc='Collecting Tumor Type...'):
    for lbl_path in data['labels']:
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lbl_json = json.load(f)
        for result in lbl_json.get('resultData', []):
            val = result.get('value', None)
            if val:
                all_values.add(val)

print(f" Existing Tumor Type ({len(all_values)} types)")
for i, v in enumerate(sorted(all_values)):
    print(f" {i} : {v}")


# In[9]:


# Tumor types (value) -> integer label mapping dictionary
# Etc -> 암의 Etc/양성종양의 Etc로 세분화
# Total 10 types

TUMOR_TYPE_MAP = {
    'Clear cell carcinoma' : 0,
    'Endometrioid carcinoma' : 1,
    'Endometrioid tumor' : 2, 
    'Etc_cancer' : 3,
    'Mature teratoma' : 4,
    'Mucinous carcinoma' : 5,
    'Mucinous tumor' : 6,
    'Serous carcinoma' : 7,
    'Serous tumor' : 8,
    'Etc_benign' : 9,
}

print("TUMOR_TYPE_MAP:")
for k, v in TUMOR_TYPE_MAP.items():
    print(f" {v} : {k}")


# In[10]:


def shoelace_area(points):
    """
    Calculate Polygon area using Shoelace Formula

    Parameters
    ----------
    points : list of dict [{"x": ..., "y": ...}, ...]

    Returns
    -------
    float : polygon ares (pixel^2)
    """

    n  = len(points)
    if n < 3:
        return 0.0

    xs = np.array([p['x'] for p in points])
    ys = np.array([p['y'] for p in points])

    # shoelace formula 
    area = 0.5 * np.abs(
        np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))
    )
    return float(area)


# In[11]:


# Labeling JSON Parsing - extract tumor area and metadata per patients
def parse_label_json(lbl_path):
    """
    Parameters
    ----------
    lbl_path : Path

    Returns
    -------
    dict : {
        'slice_idx' : int,         # slice number
        'tumor_area' : float,      # tumor area (slice 선택용)
        'tumor_type' : int,        # tumor type integer label
        'points' : list,          # polygon coordinates 
        'all_points' : list,
        'file_name': str,          # ex. "10008_CTA_1"
    """

    with open(lbl_path, 'r', encoding='utf-8') as f:
        lbl_json = json.load(f)

    result_data = lbl_json.get('resultData', [])

    if not result_data:
        return {
            'slice_idx' : lbl_json.get('idx', -1),
            'tumor_area' : 0.0,
            'tumor_type' : -1,
            'points' : [],
            'all_points' : [],
            'file_name': lbl_json.get('fileName', ''),
        }

    # If there are multiple polygons -> select the polygon with the largest agrea
    best = max(
        result_data,
        key=lambda r: shoelace_area(r.get('points', []))
    )

    points = best.get('points', [])
    all_points = [r.get('points', []) for r in result_data
                  if len(r.get('points', [])) >= 3]
    tumor_area = shoelace_area(points)
    tumor_type = TUMOR_TYPE_MAP.get(best.get('value', ''), -1)

    return {
        'slice_idx' : lbl_json.get('idx', -1),
        'tumor_area' : tumor_area,
        'tumor_type' : tumor_type,
        'points' : points,
        'all_points' : all_points,
        'file_name': lbl_json.get('fileName', ''),
    }


# In[12]:


# Builing full patient slice metadata
# add parsing results to all_patients
parse_errors = []

for pid, data in tqdm(all_patients.items(), desc='Parsing JSON'):
    slice_meta = {} # file_stem: parse_label_json result

    for lbl_path in data['labels']:
        try:
            meta = parse_label_json(lbl_path)
            slice_meta[meta['file_name']] = meta
        except Exception as e:
            parse_errors.append((pid, lbl_path.name, str(e)))

    all_patients[pid]['slice_meta'] = slice_meta

print(f"Complete labeling JSON Parsing!")
print(f" Parsing SUCCESS: {len(all_patients)} patients")
print(f" Parsing ERROR: {len(parse_errors)} events")


# In[13]:


pid = '10008'
print(f"{pid} tumor area per slice")
print(f"{'파일명':<20} {'슬라이스idx':>5} {'종양면적':>6} {'종양유형':>6}")
for fname, meta in sorted(all_patients[pid]['slice_meta'].items(),
                          key=lambda x: x[1]['tumor_area'],
                          reverse=True):
    print(f"{fname:<20} {meta['slice_idx']:>10}"
          f" {meta['tumor_area']:>12.1f} {meta['tumor_type']:>8}")


# In[14]:


def parse_emr(emr_dir, label):
    """
    Extract RLPS_YN (재발 여부) and FND (종양 유형) from EMR JSON

    Parameters
    ----------
    label : int  1=암, 0=양성종양


    Returns
    -------
    dict : {patient_id: {'recurrence': int}}
            RLPS_YN Y -> 1 (재발)
                    N -> 0 (없음)
                    U -> -1 (모름)
                    없음 -> -1 (결측)
    """

    emr_data = {}

    for emr_path in sorted(emr_dir.glob('*.json')):
        if '.ipynb_checkpoints' in str(emr_path):
            continue

        with open(emr_path, 'r', encoding='utf-8') as f:
            emr_json = json.load(f)

        pid = str(emr_json.get('PT_ID', emr_path.stem))
        rlps = emr_json.get('RLPS_YN', None)
        fnd = emr_json.get('FND', None) 

        if rlps == 'Y':
            recurrence = 1
        elif rlps == 'N':
            recurrence = 0
        else: # Unknown of Missing
            recurrence = -1

        # tumo_type mapping
        if fnd is None or fnd == 'Etc':
            tumor_type = 3 if label == 1 else 9
        else: 
            tumor_type = TUMOR_TYPE_MAP.get(fnd, 3 if label == 1 else 9)

        emr_data[pid] = {
            'recurrence': recurrence,
            'tumor_type': tumor_type,
        }

    return emr_data


# In[15]:


# EMR parsing
emr_cancer = parse_emr(EMR_CANCER, label=1)
emr_benign = parse_emr(EMR_BENIGN, label=0)
emr_all = {**emr_cancer, **emr_benign}

print("Complete parsing EMR!")
print(f" 암 EMR : {len(emr_cancer)} patients")
print(f" 양성종양 EMR : {len(emr_benign)} patients")
print(f" 전체 EMR : {len(emr_all)} patients")


# In[16]:


# add recurrence and tumor type to all_patients
recu_stats = {1: 0, 0: 0, -1: 0}
type_stats = defaultdict(int)

for pid, data in all_patients.items():
    emr = emr_all[pid]
    all_patients[pid]['recurrence'] = emr['recurrence']
    all_patients[pid]['tumor_type'] = emr['tumor_type']
    recu_stats[emr['recurrence']] += 1
    type_stats[emr['tumor_type']] += 1       

print("Complete adding recurrence and tumor type to all_patients!")
print("[Recurrence]")
print(f" 재발 (Y=1) : {recu_stats[1]} patients")
print(f" 없음 (N=0) : {recu_stats[0]} patients")
print(f" 모름/결측 (-1) : {recu_stats[-1]} patients")

print("[Tumor Type]")
for t, count in sorted(type_stats.items()):
    name = {v: k for k, v in TUMOR_TYPE_MAP.items()}.get(t, '결측')
    print(f" {t:2d} ({name:<25}) : {count} patients")


# In[17]:


pid = '10008'
print(f"{pid}")
print(f" - label : {all_patients[pid]['label']} ({all_patients[pid]['class_name']})")
print(f" - recurrence : {all_patients[pid]['recurrence']}")
print(f" - tumor_type : {all_patients[pid]['tumor_type']}")


# In[18]:


benign_patients = {pid: data for pid, data in all_patients.items()
                   if data['label'] == 0}

recu_benign = {1: 0, 0: 0, -1: 0}
for pid, data in benign_patients.items():
    recu_benign[data['recurrence']] += 1

print(f"양성종양 환자 recurrence 분포:")
print(f" 재발 (Y=1) : {recu_benign[1]} patients")
print(f" 없음 (N=0) : {recu_benign[0]} patients")
print(f" 모름/결측 (-1) : {recu_benign[-1]} patients")

benign_recu0 = [pid for pid, data in benign_patients.items()
                if data['recurrence'] == 0]
print(f"\n양성종양 + 재발 없음 샘플 RLPS_YN:")
for pid in benign_recu0[:5]:
    emr_path = EMR_BENIGN / f'{pid}.json'
    if emr_path.exists():
        with open(emr_path, 'r', encoding='utf-8') as f:
            emr_json = json.load(f)
        print(f" - {pid} RLPS_YN: {emr_json.get('RLPS_YN', None)}")


# In[19]:


pid = '40001'
emr_path = EMR_BENIGN/ f'{pid}.json'

with open(emr_path, 'r', encoding='utf-8') as f:
    emr_json = json.load(f)

print(f" {pid}.json 전체 내용:")
for key, val in emr_json.items():
    print(f" - {key}: {val} (type: {type(val).__name__})")

print(f"\nRLPS_YN 직접 접근: {emr_json.get('RLPS_YN', 'KEY_NOT_FOUND')}")
print(f"RLPAS_YN 타입 : {type(emr_json.get('RLPS_YN', None))}")


# In[20]:


# Select 8 slices
TARGET_SLICES = 8

def select_slices(pid, data, target=TARGET_SLICES):
    """
    Select 8 slices per patient

    - More than 8 : Select the top [target] slices by tumor area
    - 8 or fewer : Sampling with Replacement (Augmentation applied to duplicate slices)

    Returns
    -------
    list of dict : [
        {
            'img_path' : Path,
            'points' : list,
            'all_points' : list,
            'tumor_area' : float
            'tumor_type' : int,
            'augmented': bool,     # Flag - Whether slices added via sampleing with replacement
        }, ...
    ]
    """

    slice_meta = data.get('slice_meta', {})
    img_paths = {p.stem: p for p in data['images']} # {file_stem: Path}

    # use only slices where both images and labels exists
    valid_slices = []
    for stem, meta in slice_meta.items():
        if stem in img_paths:
            valid_slices.append({
                'img_path' : img_paths[stem],
                'points' : meta['points'],
                'all_points' : meta.get('all_points', [meta['points']]
                               if meta['points'] else []), # **추가
                'tumor_area' : meta['tumor_area'],
                'tumor_type' : meta['tumor_type'],
                'augmented': False,
            })

    n = len(valid_slices)

    # 1. if # of slices >= [target], Select the top [target] slices by tumor area
    if n >= target:
        selected = sorted(valid_slices,
                          key=lambda x: x['tumor_area'],
                          reverse=True)[:target]

    # 2. if # of slices < [target], sampling with replacement
    else:
        selected = valid_slices.copy()
        need = target - n
        extra = random.choices(valid_slices, k=need)
        for s in extra:
            selected.append({
                'img_path' : s['img_path'],
                'points' : s['points'],
                'all_points' : s['all_points'],
                'tumor_area' : s['tumor_area'],
                'tumor_type' : s['tumor_type'],
                'augmented': True,
            })

    return selected



# In[21]:


selection_stats = {'over': 0, 'exact': 0, 'under': 0}

for pid, data in tqdm(all_patients.items(), desc='Selecting slices...'):
    n = len(data['images'])
    if n > TARGET_SLICES: selection_stats['over'] += 1
    elif n == TARGET_SLICES: selection_stats['exact'] += 1
    else: selection_stats['under'] += 1

    all_patients[pid]['selected_slices'] = select_slices(pid, data)

print(f"Complete Slice Selection!")
print(f" more than 8 : {selection_stats['over']} patients")
print(f" exact 8 : {selection_stats['exact']} patients")
print(f" less than 8 : {selection_stats['under']} patients")


# In[22]:


pid = '10009'
selected = all_patients[pid]['selected_slices']

print(f"{pid} selected slices ({len(selected)} slices)")
print(f"{'파일명':<20} {'종양면적':>6} {'종양유형':>6} {'augmented':>10}")
for s in selected:
    print(f"{s['img_path'].stem:<20} {s['tumor_area']:>12.1f} "
          f"{s['tumor_type']:>8} {str(s['augmented']):>10}")


# In[23]:


TARGET_SIZE = 224 # ResNet50, ViT input size
PADDING = 0.05 # expand bounding box


# In[24]:


def create_binary_mask(img_shape, all_points, target_size=TARGET_SIZE):
    """ 모든 polygon 좌표로 binary mask 생성 후 resize

    Parameters
    ----------
    img_shape : tuple (H, W) 원본 이미지 크기
    all_points : list of list [[{'x':..., 'y':...}, ...], ...]
                              여러 polygon 좌표 리스트
    target_size : int

    Returns
    -------
    np.ndarray : (target_size, target_size) uint8 0 or 1
    """
    H, W = img_shape
    mask = np.zeros((H, W), dtype=np.uint8)

    for points in all_points:
        if len(points) < 3:
            continue

        xs = np.array([p['x'] for p in points])
        ys = np.array([p['y'] for p in points])

        xs = np.clip(xs, 0, W - 1)
        ys = np.clip(ys, 0, H - 1)

        # polygon으로 mask 채우기
        rr, cc = sk_polygon(ys, xs, shape=(H, W))
        mask[rr, cc] = 1

    mask_pil = Image.fromarray(mask * 255)
    mask_pil = mask_pil.resize((target_size, target_size),
                                Image.NEAREST)
    mask_resized = (np.array(mask_pil) > 127).astype(np.uint8)

    return mask_resized


# In[25]:


# Step 1. ROI Crop
def crop_roi(img_array, points, padding=PADDING):
    """ 
    Calculate bounding box using polygon coourdinates,
    add padding,
    expand into a square, and crop

    Parameters
    ----------
    img_array : np.ndarray (H, W) grayscale
    points : list of dict [{"x": ..., "y": ...}, ...]
    padding : float, bounding box expansion ratio

    Returns
    -------
    np.array : cropped image (square)
    list : polygon coordinates adjusted based on the crop area
    """

    H, W = img_array.shape

    xs = np.array([p['x'] for p in points])
    ys = np.array([p['y'] for p in points])

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    bw = x_max - x_min
    bh = y_max - y_min

    # add padding (includes adjacent tissue)
    x_min = x_min - bw * padding
    x_max = x_max + bw * padding
    y_min = y_min - bh * padding
    y_max = y_max + bh * padding

    # expand to square (prevent ratio distortion)
    bw_pad = x_max - x_min
    bh_pad = y_max - y_min
    side = max(bw_pad, bh_pad)

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2

    x_min = cx - side / 2
    x_max = cx + side / 2
    y_min = cy - side / 2
    y_max = cy + side / 2

    # image boundary clipping
    x_min_c = int(max(0, x_min))
    x_max_c = int(min(W, x_max))
    y_min_c = int(max(0, y_min))
    y_max_c = int(min(H, y_max))

    cropped = img_array[y_min_c:y_max_c, x_min_c: x_max_c]

    adjusted_points = [
        {'x': p['x'] - x_min_c, 'y': p['y'] - y_min_c}
        for p in points
    ]

    return cropped, adjusted_points


# In[26]:


# Step 2. Augmentation (apply only to sampled with replacement slices)
def apply_augmentation(img_array):
    """
    Random choice btw Histogram Equalization or Gaussian Noise or Contrast control(명암 조절)

    Parameters
    ----------
    img_array : np.ndarray (H, W) uint8

    Returns
    -------
    np.array : (H, W) uint8
    """   

    img_float = img_as_float(img_array)
    choice = random.randint(0, 2)

    if choice == 0:
        # Histogram Equalization
        augmented = img_as_ubyte(exposure.equalize_hist(img_float))
    elif choice == 1:
        # Gaussian Noise
        noisy = random_noise(img_float, mode='gaussian', var=0.005)
        noisy = np.clip(noisy, 0, 1)
        augmented = img_as_ubyte(noisy)
    else:
        # 명암 조절 (gamma correction)
        # gamma < 1 : 밝아짐 / gamma > 1 : 어두워짐
        gamma = random.uniform(0.7, 1.5)
        augmented = img_as_ubyte(exposure.adjust_gamma(img_float, gamma=gamma))

    return augmented



# In[27]:


# # Step 3. 3-channel preprocessing
def apply_three_channel(img_array, use_three_channel=False):
    """
    Ch 0: 원본 grayscale
    Ch 1: CLAHE 적용
    Ch 2: CLAHE -> Sobel filter 적용  **수정!

    Parameters
    ----------
    img_array : np.ndarray (H, W) uint8

    Returns
    -------
    np.array : (H, W, 3) uint8
    """
    if not use_three_channel:
        return np.stack([img_array, img_array, img_array], axis=-1)
        
    img_float = img_as_float(img_array)

    # Ch 0: 원본
    ch0 = img_array.copy()

    # Ch 1: CLAHE (clip_limit으로 대비 강화 정도 조절)
    ch1 = img_as_ubyte(
        exposure.equalize_adapthist(img_float, clip_limit=0.03)
    )

    # Ch 2: CLAHE -> Sobel (edge 강조)
    clahe_float = img_as_float(ch1)
    sobel = filters.sobel(clahe_float)
    sobel_norm = (sobel - sobel.min()) / (sobel.max() - sobel.min() + 1e-8)
    ch2 = img_as_ubyte(sobel_norm)

    return np.stack([ch0, ch1, ch2], axis=-1) # (H, W, 3)



# In[28]:


# Step 4. Resize
def resize_image(img_array, target_size=TARGET_SIZE):
    """
    원본 이미지 크기가 (512, 512)이므로 padding이 없어도 비율이 깨지지 않음

    Parameters
    ----------
    img_array : np.ndarray (H, W) uint8
    target_size : int

    Returns
    -------
    np.array : (target_size, target_size, 3) uint8
    """   

    resized = sk_resize(
        img_array,
        (target_size, target_size, 3),
        anti_aliasing=True,
        preserve_range=True,
    ).astype(np.uint8)

    return resized


# In[29]:


# Step 5. Full Pipline
def preprocess_slice(slice_info, use_three_channel=False):
    """
    Apply the entire preprocessing pipeline to a single slice
    1. ROI Crop(생략) -> 2. Augmentation -> 3. 3-channel preprocessing -> 4. Resize

    Parameters
    ----------
    slice_info : dict (select_slices return 값의 원소)

    Returns
    -------
    img_final : np.array : (TARGET_SIZE, TARGET_SIZE, 3) uint8
    mask : np.ndarray( (TARGET_SIZE, TARGET_SIZE) uint8 (0 or 1)
    """   

    # load image (grayscale)
    img = Image.open(slice_info['img_path']).convert('L')
    img_array = np.array(img)

    # points = slice_info['points']

    # 1. ROI Crop -> ** 생략!
    # img_array, _ = crop_roi(img_array, points)

    # 2. Augmentation (only to slices sampled with replacement)
    if slice_info['augmented']:
        img_array = apply_augmentation(img_array)

    # 3. 3-channel preprocessing
    img_3ch = apply_three_channel(img_array, use_three_channel=use_three_channel)

    # 4. Resize
    img_final = resize_image(img_3ch)

    # 5. Binary mask 생성
    all_points = slice_info.get('all_points', [])
    if all_points:
        mask = create_binary_mask(img_array.shape, all_points)
    else:
        mask = np.zero((TARGET_SIZE, TARGET_SIZE), dtype=np.uint8)

    return img_final, mask


# In[30]:


pid = '10008'
selected = all_patients[pid]['selected_slices']

fig, axes = plt.subplots(TARGET_SLICES, 4, figsize=(16, TARGET_SLICES * 4))
ch_names = ['Ch0: Original', 'Ch1: CLAHE', 'Ch2: CLAHE -> Sobel', 'Merged (RGB)']

for i, s in enumerate(selected):
    img_final, _ = preprocess_slice(s, use_three_channel=False)
    aug_str = ' [AUG]' if s['augmented'] else ''

    for j in range(3):
        axes[i, j].imshow(img_final[:, :, j], cmap='gray')
        axes[i, j].set_title(f"{s['img_path'].stem}{aug_str}\n{ch_names[j]}")
        axes[i, j].axis('off')

    # Ch0,1,2를 RGB로 합쳐서 visualization
    axes[i, 3].imshow(img_final)
    axes[i, 3].set_title(f"{s['img_path'].stem}{aug_str}\n{ch_names[3]}")
    axes[i, 3].axis('off')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'preprocess_check_{pid}.png', dpi=300)
plt.show()


# In[31]:


# Patient-level Stratified 5-fold CV
N_FOLDS = 5

def split_patients_kfold(all_patients, n_folds=N_FOLDS):
    """
    Patient-level stratified K-Fold CV

    Returns
    -------
    list of dict : [
        {'train': [pid, ...]. 'val': [pid, ...], 'test': [pid, ...]},
        ...
    ] 길이 = n_folds
    """   

    pids = np.array(list(all_patients.keys()))
    labels = np.array([all_patients[pid]['label'] for pid in pids])

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pids, labels)):
        folds.append({
            'fold' : fold_idx + 1,
            'train' : pids[train_idx].tolist(),
            'val' : pids[val_idx].tolist(),
        })

    return folds

def split_patients_kfold_subtype(all_patients, n_folds=N_FOLDS):
    """
    Patient-level stratified K-Fold CV
    악성 환자는 tumor_type 기준으로 stratify
    양성 환자는 'benign' 단일 클래스로 처리

    Returns
    -------
    list of dict : [
        {'train': [pid, ...]. 'val': [pid, ...], 'test': [pid, ...]},
        ...
    ] 길이 = n_folds
    """   
    pids = np.array(list(all_patients.keys()))

    # combo label 생성
    combo_labels = []
    for pid in pids:
        data = all_patients[pid]
        if data['label'] == 0:
            combo_labels.append('benign')
        else:
            combo_labels.append(f"cancer_{data['tumor_type']}")
    combo_labels = np.array(combo_labels)

    # combo label 분포 확인
    unique, counts = np.unique(combo_labels, return_counts=True)
    print("  Stratify 기준 combo label 분포:")
    for u, c in zip(unique, counts):
        print(f"  {u:<30}: {c}명")

    # StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    folds = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(pids, combo_labels)):
        folds.append({
            'fold' : fold_idx + 1,
            'train' : pids[train_idx].tolist(),
            'val' : pids[val_idx].tolist(),
        })

    return folds


# In[32]:


folds = split_patients_kfold_subtype(all_patients)

print("=" * 55)
for fold in folds:
    fold_idx = fold['fold']
    for split_name in ['train', 'val']:
        pids = fold[split_name]
        labels = [all_patients[pid]['label'] for pid in pids]
        n_cancer = sum(1 for l in labels if l == 1)
        n_benign = sum(1 for l in labels if l == 0)

        print(f"\n{fold_idx} {split_name:5s} ({len(pids)} patients)")
        print(f" 암: {n_cancer:3d} patients ({n_cancer/len(pids)*100:.1f}%)")
        print(f" 양성종양: {n_benign:4d} patients ({n_benign/len(pids)*100:.1f}%)")
    print("-" * 55)
print("=" * 55)


# In[33]:


for fold in folds:
    fold_idx = fold['fold']
    for pid in fold['train']:
        if 'folds' not in all_patients[pid]:
            all_patients[pid]['folds'] = {}
        all_patients[pid]['folds'][fold_idx] = 'train'
    for pid in fold['val']:
        if 'folds' not in all_patients[pid]:
            all_patients[pid]['folds'] = {}
        all_patients[pid]['folds'][fold_idx] = 'val'

print("Complete adding split info to all_patients")
print(f"Ex) - 10008 patient split:")
for fold_idx, split in all_patients['10008']['folds'].items():
    print(f" Fold {fold_idx}: {split}")


# In[34]:


# Save Split Metadata
def save_split_metadata(all_patients, splits, save_dir=SPLIT_META_DIR):
    """
    save split metadata
    - split_metadata.json : 전체 정보
    - split_metadata.csv : 간단한 요약 (EMR 전처리시 사용)

    JSON 구조:
    {
        "fold_1": {
            "train": [
                {
                    "patient_id" : "10008",
                    "fold" : 1,
                    "split" : "train",
                    "label" : 1,
                    "class_name" : "암",
                    "recurrence": data['recurrence'],
                    "n_slices" : 1,
                }, ...
            ],
            "val" : [...],
        },
        "fold_2": {...},
        ...
    }
    """

    # make JSON
    meta_json = {}

    for fold in folds:
        fold_idx = fold['fold']
        fold_key = f'fold_{fold_idx}'
        meta_json[fold_key] = {'train': [], 'val': []}

        for split_name in ['train', 'val']:
            for pid in sorted(fold[split_name]):
                data = all_patients[pid]
                meta_json[fold_key][split_name].append({
                    'patient_id' : pid,
                    'fold' : fold_idx,
                    'split' : split_name,
                    'label' : data['label'],
                    'class_name' : data['class_name'],
                    'tumor_type' : data['tumor_type'],
                    'recurrence' : data['recurrence'],
                    'n_slices' : len(data['images']),
                })

    # save JSON
    json_path = save_dir / 'split_metadata.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(meta_json, f, ensure_ascii=False, indent=2)

    # make CSV
    rows = []
    for fold_key, splits in meta_json.items():
        for split_name, entries in splits.items():
            for entry in entries:
                rows.append(entry)

    df_meta = pd.DataFrame(rows, columns=[
        'patient_id', 'fold', 'split', 'label', 
        'class_name', 'tumor_type', 'recurrence', 'n_slices'
    ])

    # save CSV
    csv_path = save_dir / 'split_metadata.csv'
    df_meta.to_csv(csv_path, index=False, encoding='utf-8-sig')

    return df_meta, json_path, csv_path


# In[35]:


df_meta, json_path, csv_path = save_split_metadata(all_patients, folds)

print("Saved Split Metadata!")
print(f" JSON: {json_path}")
print(f" CSV: {csv_path}")


# In[36]:


print(f"\nFold별 class 분포: ")
print(df_meta.groupby(['fold', 'split', 'class_name'])['patient_id']
      .count().rename('환자 수').to_string())

print(f"\nFold별 recurrence 분포: ")
print(df_meta.groupby(['fold', 'split', 'recurrence'])['patient_id']
      .count().rename('환자 수').to_string())

print(f"\nFold별 tumor_type 분포: ")
inv_map = {v: k for k, v in TUMOR_TYPE_MAP.items()}
df_meta['tumor_type_name'] = df_meta['tumor_type'].map(inv_map)
print(df_meta.groupby(['fold', 'split', 'tumor_type_name'])['patient_id']
      .count().rename('환자 수').to_string())


# In[37]:


# Save to npz file
# 각 fold 마다 별도의 npz 파일 저장

# fold_1_train.npz / fold_1_val.npz
# fold_2_train.npz / fold_2_val.npz
# ...

# 각 npz 파일 내부:
#    images : (N, 8, 224, 224, 3) uint8
#    labels : (N,) int -> 양성/암 (0/1)
#    tumor_types : (N,) int -> 종양 유형 (0~8)
#    recurrence : (N,) int -> 재발 여부(0/1/-1)
#    patient_ids : (N,) str -> 환자 ID


# In[38]:


def preprocess_patient(pid, data, use_three_channel=False):
    """
    Preprocess and return 8 selected slices from one patient

    Returns
    -------
    img_stack : np.ndarray : (8, 224, 224, 3) uint8
    mask_stack : np.ndarray : (8, 224, 224) uint8
    """

    selected = data['selected_slices']
    slices = []
    masks = []

    for s in selected:
        img_final, mask = preprocess_slice(s, use_three_channel=use_three_channel)
        slices.append(img_final)
        masks.append(mask)

    return np.stack(slices, axis=0), np.stack(masks, axis=0) 


# In[39]:


MINORITY_THRESHOLD = 0.8

cancer_patients = {pid: data for pid, data in all_patients.items()
                   if data['label'] == 1}

type_stats = defaultdict(int)
for pid, data in cancer_patients.items():
    type_stats[data['tumor_type']] += 1

mean_count = sum(type_stats.values()) / len(type_stats)
MINORITY_TYPES = {t for t, count in type_stats.items()
                  if count < mean_count * MINORITY_THRESHOLD}

print(f"소수 클래스 기준 : {mean_count:.1f}명 x {MINORITY_THRESHOLD} = {mean_count * MINORITY_THRESHOLD:.1f}명 미만")
print(f"소수 클래스 목록 : {MINORITY_TYPES}")
for t in sorted(MINORITY_TYPES):
    name = {v: k for k, v in TUMOR_TYPE_MAP.items()}.get(t, 'Unknown')
    print(f"  {t:2d} ({name:<25}) -> {type_stats[t]}명")

print(MINORITY_TYPES)


# In[40]:


AUGMENTED_METHODS = ['random_resized_crop', 'elastic_transform', 'gaussian_blur']

def augment_slice_for_npz(slice_info, method):
    """
    소수 클래스 오버샘플링용 augmentation
    공간 변환 없이 픽셀값만 변환 -> mask 변환 불필요

    방법:
        random_resized_crop : 병변 위치 이동
        elastic_transform : 조직 변형 시뮬레이션
        gaussian_blur : 가우시안 블러

    Parameters
    ----------
    slice_info : dict
    method : str

    Returns
    -------
    img_final : np.ndarray (224, 224, 3) uint8
    mask_final : np.ndarray(224, 224) uint 8
    """

    img = Image.open(slice_info['img_path']).convert('L')
    img_array = np.array(img)
    H, W = img_array.shape

    mask = create_binary_mask(slice_info, H, W)

    img_float = img_as_float(img_array)

    if method == 'random_resized_crop':
        crop_transform = T.RandomResizeCrop(
            size = 224,
            scale = (0.8, 1.0),
            ratio = (1.0, 1.0),
        )

        i, j, h, w = crop_transform.get_params(
            Image.fromarray(img_array),
            scale=(0.8, 1.0),
            ratio=(1.0, 1.0)
        )

        img_pil = Image.fromarray(img_array)
        img_cropped = TF.resized_crop(img_pil, i, j, h, w, (224, 224), Image.BILINEAR)
        img_array = np.array(img_cropped)

        mask_pil = Image.fromarray(mask)
        mask_cropped = TF.resized_crop(mask_pil, i, j, h, w, (224, 224), Image.NEAREST)
        mask = np.array(mask_cropped)

        img_3ch = apply_three_channel(img_array)
        img_final = np.array(Image.fromarray(img_3ch.astype(np.uint8))
                             .resize((224, 224))) if img_3ch.shape[:2] != (224, 224) \
                    else img_3ch.astype(np.uint8)
        return img_final, mask.astype(np.uint8)

    elif method == 'gaussian_blur':
        blur = T.GaussianBlur(kernel=3, sigma=(0.1, 2.0))
        img_pil = Image.fromarray(img_array)
        img_array = np.array(blur(img_pil))

        img_3ch = apply_three_channel(img_array)
        img_final = resize_image(img_3ch)
        mask_final = resize_mask(mask, TARGET_SIZE)
        return img_final, mask_final

    elif method == 'elastic_transform':
        elastic = T.ElasticTransform(alpha=50.0, sigma=5.0)

        seed = random.randint(0, 2**32 -1)

        random.seed(seed)
        torch.manual_seed(seed)
        img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float()
        img_elastic = elastic(img_tensor).squeeze().numpy()
        img_array = np.clip(img_elastic, 0, 255).astype(np.uint8)

        random.seed(seed)
        torch.manual_seed(seed)
        mask_tensor = torch.from_numpy(mask_array).unsqueeze(0).unsqueeze(0).float()
        mask_elastic = elastic(mask_tensor).squeeze().numpy()
        mask = np.clip(mask_elastic, 0, 255).astype(np.uint8)

        img_3ch = apply_three_channel(img_array)
        img_final = resize_image(img_3ch)
        mask_final = resize_mask(mask, TARGET_SIZE)
        return img_final, mask_final

    else:
        raise ValueError(f"Unknown augmentation method: {method}")


# In[41]:


def generate_augmented_patient(pid, data, method):
    """
    한 환자의 selected_slices에 augmentation 적용

    Returns
    -------
    img_array : np.ndarray (8, 224, 224, 3) uint8
    mask_array : np.ndarray(8, 224, 224) uint8
    """
    selected = data['selected_slices']
    slices = []
    masks = []

    for s in selected:
        img_final, mask_final = augment_slice_for_npz(s, method)
        slices.append(img_final)
        masks.append(mask_final)

    return np.stack(slices, axis=0), np.stack(masks, axis=0)


# In[42]:


def save_fold_npz(all_patients, folds, npz_dir=NPZ_DIR_CT_NO_3CHANNEL, use_three_channel=False):
    """
    Save train/val per fold

    Saved file:
        npz/fold_1_train.npz
        npz/fold_1_val.npz
        ...
        npz/fold_5_train.npz
        npz/fold_5_val.npz        
    """

    for fold in folds:
        fold_idx = fold['fold']

        for split_name in ['train', 'val']:
            pids = fold[split_name]

            images = []
            masks = []
            labels = []
            tumor_types = []
            recurrences = []
            patient_ids = []
            is_augmented = []

            for pid in tqdm(pids,
                            desc=f'Fold {fold_idx} {split_name}',
                            leave=True):
                data = all_patients[pid]

                # preprocess
                if 'selected_slices' not in data:
                    data['selected_slices'] = select_slices(pid, data)

                try:
                    img_array, mask_array = preprocess_patient(pid, data, use_three_channel=use_three_channel) # (8, 224, 224, 3)
                except Exception as e:
                    print(f" [WARN] Faild to preprocess {pid}: {img}")
                    continue

                images.append(img_array)
                masks.append(mask_array) # **추가
                labels.append(data['label'])
                tumor_types.append(data['tumor_type'])
                recurrences.append(data['recurrence'])
                patient_ids.append(pid)
                is_augmented.append(0)

                # 소수 클래스 augmentation (train split만)
                #if split_name == 'train' and data['label'] == 1 and data['tumor_type'] in MINORITY_TYPES:
                #    for method in AUGMENTED_METHODS:
                #        try:
                #            aug_img, aug_mask = generate_augmented_patient(pid, data, method)
                #        except Exception as e:
                #            print(f" [WARN] {pid} aug({method}) 실패: {e}")
                #            continue

                #        images.append(aug_img)
                #         masks.append(aug_mask) # **추가
                #        labels.append(data['label'])
                #        tumor_types.append(data['tumor_type'])
                #        recurrences.append(data['recurrence'])
                #        patient_ids.append(pid)
                #        is_augmented.append(1)

            # save npz
            npz_path = npz_dir / f'fold_{fold_idx}_{split_name}.npz'
            np.savez_compressed(
                npz_path,
                images = np.array(images, dtype=np.uint8),
                masks = np.array(masks, dtype=np.uint8),
                labels = np.array(labels, dtype=np.int8),
                tumor_types = np.array(tumor_types, dtype=np.int8),
                recurrences = np.array(recurrences, dtype=np.int8),
                patient_ids = np.array(patient_ids, dtype=str),
                is_augmented = np.array(is_augmented, dtype=np.int8),
            )

            n_orig = is_augmented.count(0)
            n_aug = is_augmented.count(1)

            print(f"Fold {fold_idx} {split_name:5s} Saved!"
                  f"-> {npz_path.name} "
                  f"({len(images)} patients, "
                  f"images: {np.array(images).shape}, "
                  f"masks: {np.array(masks).shape}) "
                  f"(원본 {n_orig}명 + augmented {n_aug}명 = 총 {len(images)}명)")



# In[43]:


save_fold_npz(all_patients, folds)
print("\n Saved All .npz files!")


# In[44]:


# check saved npz files
for fold_idx in range(1, N_FOLDS + 1):
    for split_name in ['train', 'val']:
        npz_path = NPZ_DIR_CT_NO_3CHANNEL / f'fold_{fold_idx}_{split_name}.npz'
        data = np.load(npz_path, allow_pickle=True)

        print(f"\n Fold {fold_idx} {split_name:5s} : {npz_path.name}")
        print(f"  images : {data['images'].shape} {data['images'].dtype}")
        print(f"  labels : {data['labels'].shape} "
              f"(암 {(data['labels']==1).sum()}) / "
              f"양성 {(data['labels']==0).sum()})")
        print(f"  tumor_types : {data['tumor_types'].shape}")
        print(f"  recurrences : {data['recurrences'].shape} "
              f"(재발 {(data['recurrences']==1).sum()} / "
              f"없음 {(data['recurrences']==0).sum()} / "
              f"결측 {(data['recurrences']==-1).sum()})")
        print(f"  patient_ids : {data['patient_ids'].shape}")


# In[45]:


# check saved npz files
for fold_idx in range(1, N_FOLDS + 1):
    for split_name in ['train', 'val']:
        npz_path = NPZ_DIR_CT_NO_3CHANNEL / f'fold_{fold_idx}_{split_name}.npz'
        data = np.load(npz_path, allow_pickle=True)

        n_orig = (data['is_augmented'] == 0).sum()
        n_aug = (data['is_augmented'] == 1).sum()

        print(f"\n Fold {fold_idx} {split_name:5s} : {npz_path.name}")
        print(f"  images : {data['images'].shape} {data['images'].dtype}")
        print(f"  원본 : {n_orig}명")
        print(f"  augmented : {n_aug}명")

        if split_name == 'train':
            print(f"    소수 클래스 분포 (원본 + aug):")
            for t in sorted(MINORITY_TYPES):
                name = {v: k for k, v in TUMOR_TYPE_MAP.items()}.get(t, 'Unknown')
                n_ori = ((data['labels'] == 1) &
                         (data['tumor_types'] == t) &
                         (data['is_augmented'] == 0)).sum()
                n_ag = ((data['labels'] == 1) &
                         (data['tumor_types'] == t) &
                         (data['is_augmented'] == 1)).sum()

                print(f"    {t} {name:<25}: "
                      f"원본 {n_ori}명 + aug {n_ag}명 = {n_ori + n_ag}명")

        if split_name == 'val':
            assert n_aug == 0, "[WARN] val에 augmented sample이 존재합니다!"
            print(f"        val augmentation 없음 확인")


# In[46]:


# visualization
npz_path = NPZ_DIR_CT_NO_3CHANNEL / 'fold_1_val.npz'
data = np.load(npz_path, allow_pickle=True)

pid = data['patient_ids'][0]
img_stack = data['images'][0] # (8, 224, 224, 3)
mask_stack = data['masks'][0] # (8, 224, 224)

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for i in range(8):
    axes[i].imshow(img_stack[i])
    #axes[i].imshow(mask_stack[i])
    axes[i].set_title(f'Slice {i+1}')
    axes[i].axis('off')

plt.suptitle(f'Patient {pid} | '
             f'label={data["labels"][0]} | '
             f'tumor_type={data["tumor_types"][0]} | '
             f'recurrence={data["recurrences"][0]}',
             fontsize=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'npz_sample_{pid}.png', dpi=100)
plt.show()
print(f"Saved sample visualization: {OUTPUT_DIR / f'npz_sample_{pid}.png'}")


# In[47]:


# 색깔이 입혀진 건, matplotlib이 (H, W, 3) 배열을 RGB로 해석해서 표기하기 때문
# 실제 의미 있는 색상이 아님. 단순 시각화 -> 즉, 실제 RGB 색상 정보가 담긴 것은 아님


# In[48]:


multi_polygon_cases = []

for pid, data in all_patients.items():
    for lbl_path in data['labels']:
        with open(lbl_path, 'r', encoding='utf-8') as f:
            lbl_json = json.load(f)
        result_data = lbl_json.get('resultData', [])
        if len(result_data) > 1:
            multi_polygon_cases.append({
                'pid': pid,
                'file': lbl_path.name,
                'n_polygon' : len(result_data)
            })

print(f"polygon이 2개 이상인 슬라이스 {len(multi_polygon_cases)} 개")
for case in multi_polygon_cases[:5]:
    print(f" {case['pid']} / {case['file']} -> {case['n_polygon']}개")


# In[49]:


sample_img = Image.open(all_patients['10008']['images'][0]).convert('L')
print(f"원본 이미지 크기: {sample_img.size}")


# In[ ]:





