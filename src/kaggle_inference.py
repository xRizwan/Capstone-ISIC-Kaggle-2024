"""Kaggle notebook script: predicts the hidden ISIC 2024 test set with the final blended model.

Runs on Kaggle without internet. The model files (5 LightGBM + 5 CNN fold models from SageMaker) come from a
private Kaggle dataset. Writes submission.csv with one blended risk score per lesion.
"""
import io
import json
import os

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def find_dir(filename, max_depth=4):
    """Folder under /kaggle/input that contains `filename`. Not a recursive glob, because the competition
    folder has 400k training images and searching all of them takes very long."""
    for root, dirs, files in os.walk('/kaggle/input'):
        if filename in files:
            return root
        depth = root.count(os.sep) - '/kaggle/input'.count(os.sep)
        dirs[:] = [] if depth >= max_depth else [d for d in dirs if d not in ('train-image', 'image')]
    raise FileNotFoundError(filename)


# the two environment variables are only for testing the script outside Kaggle
COMP_DIR = os.environ.get('ISIC_COMP_DIR') or find_dir('test-metadata.csv')
MODEL_DIR = os.environ.get('ISIC_MODEL_DIR') or find_dir('lgbm_meta.json')
print('competition data:', COMP_DIR)
print('model files:', MODEL_DIR)
N_FOLDS = 5
IMG_SIZE = 128

CAT_COLS = ['anatom_site_general', 'sex', 'tbp_tile_type', 'tbp_lv_location']
NORM_SRC_COLS = [
    'tbp_lv_H', 'ABC_score', 'hue_contrast', 'size_and_color_irregularity', 'elongation',
    'tbp_lv_deltaB', 'tbp_lv_deltaA',
    'clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_color_std_mean', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
]


# ---------- metadata features (same as src/features.py) ----------
def add_features(df):
    df = df.copy()
    df['hue_contrast'] = df['tbp_lv_H'] - df['tbp_lv_Hext']
    df['elongation'] = df['tbp_lv_minorAxisMM'] / (df['clin_size_long_diam_mm'] + 1e-6)
    df['chroma_contrast'] = df['tbp_lv_C'] - df['tbp_lv_Cext']
    df['texture_contrast'] = df['tbp_lv_stdL'] / (df['tbp_lv_stdLExt'] + 1e-6)
    df['size_and_color_irregularity'] = df['clin_size_long_diam_mm'] * df['tbp_lv_norm_color']
    df['ABC_score'] = df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'] + (df['tbp_lv_symm_2axis'] * 10)
    return df


def add_patient_features(df, cols):
    df = df.copy()
    for col in cols:
        p_mean = df.groupby('patient_id')[col].transform('mean')
        p_std = df.groupby('patient_id')[col].transform('std')
        df[f'{col}_pnorm'] = (df[col] - p_mean) / (p_std + 1e-6)
    df['patient_lesion_count'] = df.groupby('patient_id')['patient_id'].transform('count')
    return df


def predict_lgbm(test):
    with open(os.path.join(MODEL_DIR, 'lgbm_meta.json')) as f:
        meta = json.load(f)

    df = test.copy()
    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col], categories=meta['categories'][col])
    df = add_features(df)
    df = add_patient_features(df, NORM_SRC_COLS)
    X = df[meta['features']]

    preds = []
    for fold in range(N_FOLDS):
        booster = lgb.Booster(model_file=os.path.join(MODEL_DIR, f'lgbm_fold{fold}.txt'))
        preds.append(booster.predict(X))
    return np.mean(preds, axis=0)


# ---------- CNN ----------
class TestDataset(Dataset):
    def __init__(self, ids, img_path):
        self.ids = ids
        self.img_path = img_path
        self.h5 = None
        self.tfms = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        if self.h5 is None:
            self.h5 = h5py.File(self.img_path, 'r')
        img = Image.open(io.BytesIO(self.h5[self.ids[i]][()])).convert('RGB')
        return self.tfms(img)


@torch.no_grad()
def predict_cnn(test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    models = []
    for fold in range(N_FOLDS):
        model = torchvision.models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        model.load_state_dict(torch.load(os.path.join(MODEL_DIR, f'cnn_fold{fold}.pt'), map_location=device))
        models.append(model.to(device).eval())

    loader = DataLoader(TestDataset(test['isic_id'].to_numpy(), os.path.join(COMP_DIR, 'test-image.hdf5')),
                        batch_size=256, shuffle=False, num_workers=4)
    preds = []
    for x in loader:
        x = x.to(device)
        probs = torch.stack([torch.sigmoid(model(x)[:, 0]) for model in models]).mean(dim=0)
        preds.append(probs.cpu().numpy())
    return np.concatenate(preds)


def main():
    test = pd.read_csv(os.path.join(COMP_DIR, 'test-metadata.csv'), low_memory=False)
    print('test rows:', len(test))

    with open(os.path.join(MODEL_DIR, 'blend.json')) as f:
        w = json.load(f)['w']

    lgbm_score = predict_lgbm(test)
    cnn_score = predict_cnn(test)

    # same blend as in the cross-validation: ranks of both models, 70% LightGBM + 30% CNN
    lgbm_rank = pd.Series(lgbm_score).rank(pct=True)
    cnn_rank = pd.Series(cnn_score).rank(pct=True)
    blend = w * lgbm_rank + (1 - w) * cnn_rank

    submission = pd.DataFrame({'isic_id': test['isic_id'], 'target': blend.to_numpy()})
    submission.to_csv('submission.csv', index=False)
    print(submission.head())


if __name__ == '__main__':
    main()
