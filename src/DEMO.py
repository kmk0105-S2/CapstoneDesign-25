#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import random
from pathlib import Path
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import ipywidgets as widgets
from IPython.display import display, clear_output

import torch
import torch.nn as nn
from torchvision import models

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# In[6]:


DATA_ROOT = Path('/tf/nasw/dataset001/preprocessed/npz_ct')
MODEL_DIR = Path('/tf/notebooks/Image Encoder/Untitled Folder/ct_binary_models')
MODEL_NAME = 'Hybrid_R50'

BEST_FOLD = 1
BEST_WEIGHT = MODEL_DIR / f'{MODEL_NAME}_fold{BEST_FOLD}_best.pth'

NORM_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
NORM_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

print(f'Weight exists: {MODEL_DIR.exists()}')
print(f'Weight path: {BEST_WEIGHT}')
print(f'Weight exists: {BEST_WEIGHT.exists()}')


# In[7]:


class CNNTransformerHybridPatient(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2, 
                 dim_feedforward=1024, dropout=0.3,):
        super().__init__()
        backbone = models.resnet50(weights=None)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        
        self.proj = nn.Linear(feat_dim, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+8, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True,
            activation="gelu"
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers= num_layers)
        self.norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 1)
        )

    def forward(self, x, return_attn=False):
        B, S, C, H, W = x.shape
        x = x.view(B * S, C, H, W)

        feat = self.encoder(x)
        feat = feat.view(B, S, -1)
        feat = self.proj(feat)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, feat], dim=1)
        tokens = tokens + self.pos_embed[:, :tokens.size(1), :]

        tokens = self.transformer(tokens)
        tokens = self.norm(tokens)

        cls_out = tokens[:, 0, :]
        logits = self.classifier(cls_out)

        if return_attn:
            cls_vec = tokens[:, 0:1, :]
            slice_vec = tokens[:, 1:, :]
            sim = torch.cosine_similarity(
                cls_vec.expand_as(slice_vec), slice_vec, dim=-1
            )
            attn = torch.softmax(sim, dim=-1)
            return logits, attn

        return logits


model = CNNTransformerHybridPatient()
ckpt = torch.load(BEST_WEIGHT, map_location=device)
model.load_state_dict(ckpt)
model.to(device).eval()
print('Model loadded successfully')


# In[8]:


npz_path = DATA_ROOT / f'fold_{BEST_FOLD}_val.npz'
npz = np.load(npz_path, allow_pickle=False, mmap_mode='r')
images = npz['images']
labels = npz['labels'].astype(int)
N = len(labels)

benign_idx = np.where(labels == 0)[0]
malignant_idx = np.where(labels == 1)[0]

print(f'Val ser -> Benign: {len(benign_idx)}, Malignant: {len(malignant_idx)}, Total: {N}')

    
label_toggle = widgets.ToggleButtons(
    options=[('Benign (양성)', 0), ('Malignant (악성)', 1)],
    description='Class:',
    button_stype='',
    stype={'description_width': 'initial'},
    layout=widgets.Layout(width='480px')
)

sample_slider = widgets.IntSlider(
    value=0, min=0, max=len(benign_idx)-1,
    description='샘플 번호:',
    continuous_update=False,
    stype={'description_width': 'initial'},
    layout=widgets.Layout(width='480px')
)

run_btn = widgets.Button(
    description='>> Start prediction',
    button_stype='primary',
    layout=widgets.Layout(width='180px', height='40px')
)

out = widgets.Output()

def on_toggle_change(change):
    cls = change['new']
    idx_pool = benign_idx if cls == 0 else malignant_idx
    sampler_slider.max = len(idx_pool) - 1
    sampler_slider.value = 0

label_toggle.observe(on_toggle_change, names='value')

def preprocess(img_np):
    x = torch.from_numpy(img_np.copy()).float()
    x = x.permute(0, 3, 1, 2)
    x = x / 255.0
    x = (x - NORM_MEAN) / NORM_STD
    return x.unsqueeze(0).to(device)

def run_prediction(_):
    with out:
        clear_output(wait=True)

        cls = label_toggle.value
        idx_pool = benign_idx if cls == 0 else malignant_idx
        patient_i = idx_pool[sample_slider.value]
        img_np = images[patient_i]
        gt_label = labels[patient_i]
        S = img_np.shape[0]

        x_tensor = preprocess(img_np)
        with torch.no_grad():
            logit, attn= model(x_tensor, return_attn=True)

        prob = torch.sigmoid(logit).item()
        pred_cls = int(prob > 0.5)
        attn_np = attn[0].cpu().numpy()
        correct = (pred_cls == gt_label)

        gt_text = 'Benign' if gt_label == 0 else 'Malignant'
        pred_text = 'Benign' if pred_cls == 0 else 'Malignant'
        gt_color = '#27ae60' if gt_label == 0 else '#e74c3c'
        pred_color = '#27ae60' if pred_cls == 0 else '#e74c3c'
        result_emoji = 'Correct!' if correct else 'Wrong'

        n_cols = min(S, 8)
        fig = plt.figure(figsize=(18, 15), facecolor='white')
        gs_main = gridspec.GridSpec(
            3, 1, figure=fig,
            height_ratios=[0.18, 0.62, 0.20],
            hspace=0.08
        )

        ax_header = fig.add_subplot(gs_main[0])
        ax_header.set_facecolor('white')
        ax_header.axis('off')

        ax_header.text(
            0.5, 0.75,
            'Ovarian Cancer Benign/Malignant Prediction : Hybrid_R50 (ResNet50 + Transformer)',
            ha='center', va='center', fontsize=20, fontweight='bold',
            color='black', transform=ax_header.transAxes
        )

        ax_header.text(
            0.28, 0.22,
            f'Ground True : {gt_text}',
            ha='center', va='center', fontsize=15, 
            color=gt_color, fontweight='bold',
            transform=ax_header.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=gt_color, linewidth=2)
        )

        ax_header.text(
            0.72, 0.22,
            f'Prediction : {pred_text} ({prob*100:.1f}%)   {result_emoji}',
            ha='center', va='center', fontsize=15, 
            color=gt_color, fontweight='bold',
            transform=ax_header.transAxes,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor=gt_color, linewidth=2)
        )

        gs_slices = gridspec.GridSpecFromSubplotSpec(
            1, n_cols, subplot_spec=gs_main[1],
            wspace=0.04
        )

        attn_norm = (attn_np - attn_np.min()) / (attn_np.max() - attn_np.min() + 1e-8)

        for i in range(n_cols):
            ax = fig.add_subplot(gs_slices[i])
            slice_img = img_np[i, :, :, 0]
            ax.imshow(slice_img, cmap='gray', vmin=0, vmax=255)
            ax.axis('off')

            a = attn_norm[i]
            color = plt.cm.YlOrRd(0.3 + 0.7 * a)
            lw = 1.5 + 4.5 * a
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(lw)

            ax.set_title(
                f'S{i+1}\n{attn_np[i]*100:.1f}%',
                color='black', fontsize=8, pad=3
            )

        
        gs_bottm = gridspec.GridSpecFromSubplotSpec(
            1, 2, subplot_spec=gs_main[2],
            width_ratios=[3, 1], wspace=0.08
        )

        ax_bar = fig.add_subplot(gs_bottm[0])
        ax_bar.set_facecolor('white')
        colors_bar = [plt.cm.YlOrRd(0.3 + 0.7 * v) for v in attn_norm]
        bars = ax_bar.bar(
            range(1, n_cols+1), attn_np * 100,
            color=colors_bar, edgecolor='black', linewidth=0.5, width=0.7
        )
        ax_bar.set_xlabel('Slice index', color='white', fontsize=10)
        ax_bar.set_ylabel('Attention (%)', color='white', fontsize=10)
        ax_bar.set_title('Transformer Attention per Slice', color='black', fontsize=11)
        ax_bar.tick_params(colors='black')
        for sp in ax_bar.spines.values():
            sp.set_color('#ccc')
        ax_bar.set_xticks(range(1, n_cols+1))
        ax_bar.yaxis.label.set_color('black')

        ax_gauge = fig.add_subplot(gs_bottm[1])
        ax_gauge.set_facecolor('white')
        ax_gauge.axis('off')

        ax_gauge.barh(0, 1, color='#e0e0e0', height=0.4, left=0)
        bar_color = '#e74c3c' if prob > 0.5 else '#27ae60'
        ax_gauge.barh(0, prob, color=bar_color, height=0.4, left=0)
        ax_gauge.axvline(0.5, color='black', linewidth=1.5, linestyle='--', alpha=0.7)


        ax_gauge.set_xlim(0, 1)
        ax_gauge.set_ylim(-0.6, 0.9)
        ax_gauge.text(0.5, 0.58, 'Malignant Probability',
                      ha='center', color='black', fontsize=10,
                      transform=ax_gauge.transAxes)
        ax_gauge.text(0.5, -0.52, f'{prob*100:.1f}%',
                      ha='center', color=bar_color, fontsize=18, fontweight='bold',
                      transform=ax_gauge.transAxes)
        ax_gauge.text(0.02, -0.1, '0%', color='#555', fontsize=8)
        ax_gauge.text(0.50, -0.1, '50%', color='black', fontsize=8, ha='center')
        ax_gauge.text(0.98, -0.1, '100%', color='#555', fontsize=8, ha='right')

        plt.show()
        print(f'  Patient index: {patient_i}  |  logit: {logit.item():.4f}  |  prob: {prob:.4f}')

run_btn.on_click(run_prediction)

display(
    widgets.VBox([
        widgets.HTML('<h3 style="color:0f1117">Choose Patient!</h3>'),
        label_toggle,
        sample_slider,
        run_btn,
        out
    ])
)


# In[ ]:




