"""Kaggle notebook script: predicts the hidden ISIC 2024 test set with the boosting ensemble + CNN.

Same idea as kaggle_inference.py, but the metadata part is the ensemble from 08_boosting_ensemble.ipynb:
LightGBM, XGBoost and CatBoost on the wide feature set, 5 seeds each. The boosting models are small, so they are
trained inside the Kaggle notebook on the full training metadata (that also avoids problems with library versions).
The 5 CNN fold models still come from the private Kaggle dataset.
"""
import io
import os

import h5py
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
import xgboost as xgb
from catboost import CatBoostClassifier
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


# the environment variables are only for testing the script outside Kaggle
COMP_DIR = os.environ.get('ISIC_COMP_DIR') or find_dir('test-metadata.csv')
MODEL_DIR = os.environ.get('ISIC_MODEL_DIR') or find_dir('cnn_fold0.pt')
TRAIN_CSV = os.environ.get('ISIC_TRAIN_CSV') or os.path.join(COMP_DIR, 'train-metadata.csv')
print('competition data:', COMP_DIR)
print('model files:', MODEL_DIR)
N_FOLDS = 5
IMG_SIZE = 128
SEEDS = [42, 43, 44, 45, 46]
NEG_FRAC = 0.01
W = 0.7   # weight of the boosting ensemble in the blend, the CNN gets 1 - W

CAT_COLS = ['anatom_site_general', 'sex', 'tbp_tile_type', 'tbp_lv_location']
ENG_COLS = ['hue_contrast', 'chroma_contrast', 'elongation', 'texture_contrast',
            'size_and_color_irregularity', 'ABC_score']
NO_NORM = ['age_approx', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'size_age', 'size_per_age', 'nevi_conf_per_age']


# ---------- metadata features (same as src/features.py and src/features_wide.py) ----------
def add_features(df):
    df = df.copy()
    df['hue_contrast'] = df['tbp_lv_H'] - df['tbp_lv_Hext']
    df['elongation'] = df['tbp_lv_minorAxisMM'] / (df['clin_size_long_diam_mm'] + 1e-6)
    df['chroma_contrast'] = df['tbp_lv_C'] - df['tbp_lv_Cext']
    df['texture_contrast'] = df['tbp_lv_stdL'] / (df['tbp_lv_stdLExt'] + 1e-6)
    df['size_and_color_irregularity'] = df['clin_size_long_diam_mm'] * df['tbp_lv_norm_color']
    df['ABC_score'] = df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'] + (df['tbp_lv_symm_2axis'] * 10)
    return df


def add_wide_features(df):
    eps = 1e-6
    new = pd.DataFrame(index=df.index)
    new['lesion_shape_index'] = df['tbp_lv_areaMM2'] / (df['tbp_lv_perimeterMM'] ** 2 + eps)
    new['luminance_contrast'] = df['tbp_lv_L'] - df['tbp_lv_Lext']
    new['color_difference'] = np.sqrt(df['tbp_lv_deltaA'] ** 2 + df['tbp_lv_deltaB'] ** 2 + df['tbp_lv_deltaL'] ** 2)
    new['border_complexity'] = df['tbp_lv_norm_border'] + df['tbp_lv_symm_2axis']
    new['color_uniformity'] = df['tbp_lv_color_std_mean'] / (df['tbp_lv_radial_color_std_max'] + eps)
    new['distance_3d'] = np.sqrt(df['tbp_lv_x'] ** 2 + df['tbp_lv_y'] ** 2 + df['tbp_lv_z'] ** 2)
    new['perimeter_to_area'] = df['tbp_lv_perimeterMM'] / (df['tbp_lv_areaMM2'] + eps)
    new['visibility_score'] = df['tbp_lv_deltaLBnorm'] + df['tbp_lv_norm_color']
    new['symmetry_border'] = df['tbp_lv_symm_2axis'] * df['tbp_lv_norm_border']
    new['color_consistency'] = df['tbp_lv_stdL'] / (df['tbp_lv_Lext'] + eps)
    new['size_age'] = df['clin_size_long_diam_mm'] * df['age_approx']
    new['hue_color_std'] = df['tbp_lv_H'] * df['tbp_lv_color_std_mean']
    new['severity_index'] = (df['tbp_lv_norm_border'] + df['tbp_lv_norm_color'] + df['tbp_lv_eccentricity']) / 3
    new['color_contrast_index'] = (df['tbp_lv_deltaA'] + df['tbp_lv_deltaB'] + df['tbp_lv_deltaL']
                                   + df['tbp_lv_deltaLBnorm'])
    new['log_area'] = np.log1p(df['tbp_lv_areaMM2'])
    new['size_per_age'] = df['clin_size_long_diam_mm'] / (df['age_approx'] + eps)
    new['color_variance_ratio'] = df['tbp_lv_color_std_mean'] / (df['tbp_lv_stdLExt'] + eps)
    new['border_color'] = df['tbp_lv_norm_border'] * df['tbp_lv_norm_color']
    new['size_color_contrast'] = df['clin_size_long_diam_mm'] / (df['tbp_lv_deltaLBnorm'] + eps)
    new['nevi_conf_per_age'] = df['tbp_lv_nevi_confidence'] / (df['age_approx'] + eps)
    new['color_asymmetry'] = df['tbp_lv_radial_color_std_max'] * df['tbp_lv_symm_2axis']
    new['color_range'] = ((df['tbp_lv_L'] - df['tbp_lv_Lext']).abs() + (df['tbp_lv_A'] - df['tbp_lv_Aext']).abs()
                          + (df['tbp_lv_B'] - df['tbp_lv_Bext']).abs())
    new['shape_color'] = df['tbp_lv_eccentricity'] * df['tbp_lv_color_std_mean']
    new['border_length_ratio'] = df['tbp_lv_perimeterMM'] / (2 * np.pi * np.sqrt(df['tbp_lv_areaMM2'] / np.pi) + eps)
    return pd.concat([df, new], axis=1), new.columns.tolist()


def add_patient_norms(df, cols):
    # (value - patient mean) / patient std, built in one go so the dataframe doesn't get fragmented
    grouped = df.groupby('patient_id')
    new = {}
    for col in cols:
        new[f'{col}_pnorm'] = (df[col] - grouped[col].transform('mean')) / (grouped[col].transform('std') + 1e-6)
    new['patient_lesion_count'] = grouped['patient_id'].transform('count')
    return pd.concat([df, pd.DataFrame(new, index=df.index)], axis=1), list(new)


def build(df, num_cols, categories):
    df = df.copy()
    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col], categories=categories[col])
    df = add_features(df)
    df, wide_cols = add_wide_features(df)
    norm_src = [col for col in num_cols + ENG_COLS + wide_cols if col not in NO_NORM]
    df, patient_cols = add_patient_norms(df, norm_src)
    return df, num_cols + CAT_COLS + ENG_COLS + wide_cols + patient_cols


def make_model(kind, seed):
    if kind == 'lgbm':
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1,
                                  colsample_bytree=0.8, random_state=seed, verbose=-1)
    if kind == 'xgb':
        return xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, subsample=0.8,
                                 colsample_bytree=0.8, enable_categorical=True, tree_method='hist', random_state=seed)
    return CatBoostClassifier(iterations=400, learning_rate=0.06, depth=6, random_seed=seed, verbose=0,
                              cat_features=CAT_COLS)


def as_strings(X):
    # CatBoost wants categorical columns as strings
    X = X.copy()
    for col in CAT_COLS:
        X[col] = X[col].astype(object).where(X[col].notna(), 'missing').astype(str)
    return X


def predict_boosting(test):
    train = pd.read_csv(TRAIN_CSV, low_memory=False)
    # only the columns that are also in the test set, the others leak the diagnosis
    train = train[[col for col in train.columns if col in test.columns or col == 'target']]
    num_cols = train.select_dtypes('number').drop(columns=['target']).columns.tolist()
    categories = {col: sorted(train[col].dropna().unique().tolist()) for col in CAT_COLS}

    train, features = build(train, num_cols, categories)
    test, _ = build(test, num_cols, categories)
    print('features:', len(features), ' train rows:', len(train))

    pos = train.index[train['target'] == 1]
    neg = train[train['target'] == 0]
    ranks = []
    for kind in ['lgbm', 'xgb', 'cat']:
        X_train = as_strings(train[features]) if kind == 'cat' else train[features]
        X_test = as_strings(test[features]) if kind == 'cat' else test[features]
        rank = np.zeros(len(test))
        for seed in SEEDS:
            idx = np.concatenate([pos, neg.sample(frac=NEG_FRAC, random_state=seed).index])
            model = make_model(kind, seed)
            model.fit(X_train.loc[idx], train.loc[idx, 'target'])
            rank += pd.Series(model.predict_proba(X_test)[:, 1]).rank(pct=True).to_numpy() / len(SEEDS)
        ranks.append(rank)
        print(kind, 'done')
    return np.mean(ranks, axis=0)


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

    boosting_rank = pd.Series(predict_boosting(test)).rank(pct=True)
    cnn_rank = pd.Series(predict_cnn(test)).rank(pct=True)
    blend = W * boosting_rank + (1 - W) * cnn_rank

    submission = pd.DataFrame({'isic_id': test['isic_id'], 'target': blend.to_numpy()})
    submission.to_csv('submission.csv', index=False)
    print(submission.head())


if __name__ == '__main__':
    main()
