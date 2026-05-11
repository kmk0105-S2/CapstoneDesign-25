#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


def check_path(path):
    if not os.path.exists(path):
        print(f"[WARN] Cannot find path: {path}")
        return False
    return True


# In[3]:


print(os.getcwd())


# In[4]:


BASE_PATH = '/tf/nasw/dataset001/'

emr_paths = {
    "Cancer": os.path.join(BASE_PATH, 'Other/메타데이터/암/EMR'),
    "Benign": os.path.join(BASE_PATH, 'Other/메타데이터/양성종양/EMR')
}

ct_paths = {
    "Training": os.path.join(BASE_PATH, 'Training/01.원천데이터'),
    "Validation": os.path.join(BASE_PATH, 'Validation/01.원천데이터')
}


# In[5]:


if not check_path(BASE_PATH):
    raise SystemExit("Incorrect dataset path.")


# In[6]:


def analyze_emr(path, label="Unknown"):
    files = glob.glob(os.path.join(path, "*.json"))
    patient_ids = [os.path.basename(f).split('.')[0] for f in files]

    features = []
    if files:
        with open(files[0], 'r', encoding='utf-8') as f:
            sample_data = json.load(f)
            features = list(sample_data.keys())

    print(f"[{label}]")
    print(f"- Total patients: {len(patient_ids)}")
    print(f"- feature example: {features[:10]}... (total {len(features)} features)")

    return set(patient_ids), features


# In[7]:


cancer_emr_ids, cancer_features = analyze_emr(emr_paths["Cancer"], "Cancer (Malignant)")
benign_emr_ids, benign_features = analyze_emr(emr_paths["Benign"], "Benign")


# In[8]:


def analyze_ct_images(base_dir, split_name):
    report = {'total_img_count': 0}

    for status in ['암', '양성종양']:
        status_path = os.path.join(base_dir, status, 'CT')

        all_images = glob.glob(os.path.join(status_path, "**/*.png"), recursive=True)
        img_count = len(all_images)
        report['total_img_count'] += img_count

        patient_stats = {} # patient_id: {'CTA' : 0, 'CTC' : 0}}

        for img_path in all_images:
            filename = os.path.basename(img_path)
            # ex. 10008_CTA_1.png -> ID = 10008
            p_id = filename.split('_')[0]
            img_type = 'CTA' if '_CTA_' in filename else 'CTC'

            if p_id not in patient_stats:
                patient_stats[p_id] = {'CTA': 0, 'CTC': 0}
            patient_stats[p_id][img_type] += 1

        num_patients = len(patient_stats)
        cta_counts = [v['CTA'] for v in patient_stats.values() if v['CTA'] >0]
        ctc_counts = [v['CTC'] for v in patient_stats.values() if v['CTC'] >0]

        report[status] = {
            'patient_ids': set(patient_stats.keys()),
            'total_patients': num_patients,
            'total_images': img_count,
            'cta_range': (min(cta_counts), max(cta_counts)) if cta_counts else (0,0),
            'ctc_range': (min(ctc_counts), max(ctc_counts)) if ctc_counts else (0,0)
        }

    return report



# In[9]:


train_ct_stats = analyze_ct_images(ct_paths["Training"], "Training")
valid_ct_stats = analyze_ct_images(ct_paths["Validation"], "Validation")


# In[10]:


print("[CT Image Summary]")
for split_name, stats in [("Training", train_ct_stats), ("Validation", valid_ct_stats)]:
    print(f"\n[{split_name} Split]")
    print(f"Total images : {stats['total_img_count']} images")

    for status in ['암', '양성종양']:
        s = stats[status]
        print(f"- {status}: {s['total_patients']} patients")
        print(f" > {s['total_images']} images")
        print(f" > Range of CTA images : {s['cta_range'][0]} ~ {s['cta_range'][1]}")
        print(f" > Range of CTC images : {s['ctc_range'][0]} ~ {s['ctc_range'][1]}")


# In[11]:


all_emr_ids = cancer_emr_ids.union(benign_emr_ids)
all_ct_ids = train_ct_stats['암']['patient_ids'] | train_ct_stats['양성종양']['patient_ids'] | valid_ct_stats['암']['patient_ids'] | valid_ct_stats['양성종양']['patient_ids']

intersection = all_emr_ids.intersection(all_ct_ids)

print("[Multimodal Summary]")
print(f" 1. Total EMR patients: {len(all_emr_ids)} patients")
print(f" 2. Total CT patients: {len(all_ct_ids)} patients")
print(f" 3. Patients who have both EMR and CT: {len(intersection)} patients")


# In[14]:


# Integrate Training and Validation
all_splits = [ct_paths["Training"], ct_paths["Validation"]]
patient_dict = {}

for split_path in all_splits:
    if not os.path.exists(split_path):
        print(f"[WARN] No such path: {split_path}")
        continue

    for status in ["암", "양성종양"]:
        label = 1 if status == "암" else 0

        # 복부 CT
        ctc_path = os.path.join(split_path, status, "CT", "**/*_CTC_*.png")
        ctc_files = glob.glob(ctc_path, recursive=True)

        # 골반 CT
        cta_path = os.path.join(split_path, status, "CT", "**/*_CTA_*.png")
        cta_files = glob.glob(cta_path, recursive=True)

        # SONO (초음파) 
        sono_path = os.path.join(split_path, status, "초음파", "**/*_SONO_*.png")
        sono_files = glob.glob(sono_path, recursive=True)

        for f in ctc_files:
            p_id = os.path.basename(f).split('_')[0]
            if p_id not in patient_dict:
                patient_dict[p_id] = {'patient_id': p_id, 'label': label, 
                                      'ctc_paths': [], 'cta_paths': [], 'sono_paths': []}
            patient_dict[p_id]['ctc_paths'].append(f)

        for f in cta_files:
            p_id = os.path.basename(f).split('_')[0]
            if p_id not in patient_dict:
                patient_dict[p_id] = {'patient_id': p_id, 'label': label, 
                                      'ctc_paths': [], 'cta_paths': [], 'sono_paths': []}
            patient_dict[p_id]['cta_paths'].append(f)

        for f in sono_files:
            p_id = os.path.basename(f).split('_')[0]
            if p_id not in patient_dict:
                patient_dict[p_id] = {'patient_id': p_id, 'label': label, 
                                      'ctc_paths': [], 'cta_paths': [], 'sono_paths': []}
            patient_dict[p_id]['sono_paths'].append(f)

rows = list(patient_dict.values())
df_integrated = pd.DataFrame(rows)

df_integrated['ctc_count'] = df_integrated['ctc_paths'].apply(len)
df_integrated['cta_count'] = df_integrated['cta_paths'].apply(len)
df_integrated['sono_count'] = df_integrated['sono_paths'].apply(len)

has_cta_and_sono = df_integrated[(df_integrated['cta_count'] > 0) & (df_integrated['sono_count'] > 0)]

output_path = "integrated_metadata.csv"
df_integrated.to_csv(output_path, index=False)

print("Complete data intgration and extract CTC")
print(f"Saved Integrated dataset metadata: {output_path} (total {len(df_integrated)} patients)")
print(f"Both have CTC and CTA: {len(df_integrated[(df_integrated.ctc_count > 0) & (df_integrated.cta_count > 0)])} patients")
print(f"Both have CTA and SONO: {len(has_cta_and_sono)} patients")


# In[ ]:


summary_stats = []

for label, group in df_integrated.groupby('label'):
    status_str = "Cancer" if label == 1 else "Benign"

    num_patients = len(group)
    total_ctc = group['ctc_paths'].apply(len).sum()
    total_cta = group['cta_paths'].apply(len).sum()

    summary_stats.append({
        'Status': status_str,
        'patients number': num_patients,
        'Total CTC images number': total_ctc,
        'Total CTA images number': total_cta,
        'Avg CTC per patient': round(total_ctc / num_patients, 1) if num_patients > 0 else 0,
        'Avg CTA per patient': round(total_cta / num_patients, 1) if num_patients > 0 else 0
    })

df_summary = pd.DataFrame(summary_stats)

print("Integrated Dataset Summary")
display(df_summary)

df_total_dist = df_integrated[['patient_id', 'label', 'ctc_count', 'cta_count']].copy()
df_total_dist['Status'] = df_total_dist['label'].map({1: 'Cancer', 0: 'Benign'})

display(df_total_dist.head())


# In[ ]:


cancer_counts = df_total_dist[df_total_dist['Status'] == 'Cancer']['Count']
benign_counts = df_total_dist[df_total_dist['Status'] == 'Benign']['Count']

plt.figure(figsize=(12, 6))

plt.hist(cancer_counts, bins=30, alpha=0.5, label='Cancer', color='red', edgecolor='black')
plt.hist(benign_counts, bins=30, alpha=0.5, label='Benign', color='blue', edgecolor='black')

plt.title('Distribution of CTC Images per Patient', fontsize=15)
plt.xlabel('Number of Images per Patient', fontsize=12)
plt.ylabel('Number of Patients', fontsize=12)
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

print(df_total_dist.groupby('Status')['Count'].describe())


# In[ ]:


all_ct_images = []
for split_path in all_splits:
    if os.path.exists(split_path):
        all_ct_images.extend(glob.glob(os.path.join(split_path, "**/CT/**/*.png"), recursive=True))

patient_inventory = {}

for img_path in all_ct_images:
    filename = os.path.basename(img_path)
    p_id = filename.split('_')[0]
    img_type = 'CTA' if '_CTA_' in filename else 'CTC'

    status = "Cancer" if "암" in img_path else "Benign"

    if p_id not in patient_inventory:
        patient_inventory[p_id] = {'Status': status, 'CTA':0, 'CTC':0}

    patient_inventory[p_id][img_type] += 1

df_inventory = pd.DataFrame.from_dict(patient_inventory, orient='index').reset_index()
df_inventory.columns = ['PatientID', 'Status', 'CTA_Count', 'CTC_Count']

def check_coverage(row):
    if row['CTA_Count'] > 0 and row['CTC_Count'] > 0:
        return 'Both'
    elif row['CTC_Count'] > 0:
        return 'CTC only'
    elif row['CTA_Count'] > 0:
        return 'CTA only'
    else:
        return 'None'

df_inventory['Coverage'] = df_inventory.apply(check_coverage, axis=1)


# In[ ]:


coverage_summary = df_inventory['Coverage'].value_counts()
print(coverage_summary)


# In[ ]:


stats_list = []

print("Scanning CTC image labeling data...")

for idx, row in tqdm(df_integrated.iterrows(), total=len(df_integrated), desec="Checking Labels"):
    img_path = row['file_path']
    p_id = row['patient_id']

    json_path = img_path.replace("01.원천데이터", "02.라벨링데이터").replace('.png', '.json')

    label_exists = os.path.exists(json_path)
    has_tumor = False
    area = 0

    if label_exists:
        has_tumor

