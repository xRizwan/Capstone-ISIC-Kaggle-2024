"""SageMaker training script: LightGBM on the metadata (v3 features) with 5-fold patient CV.

Same code as in 03_lightgbm.ipynb, just as a script so SageMaker can run and tune it.
It prints the CV pAUC (the HPO objective) and saves the 5 fold models + out-of-fold predictions.
"""
import argparse
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import CAT_COLS, build_features
from isic_pauc import p_auc_tpr

SEED = 42
N_FOLDS = 5


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--learning-rate', type=float, default=0.05)
    parser.add_argument('--n-estimators', type=int, default=300)
    parser.add_argument('--num-leaves', type=int, default=31)
    parser.add_argument('--min-child-samples', type=int, default=20)
    parser.add_argument('--subsample', type=float, default=0.8)
    parser.add_argument('--colsample-bytree', type=float, default=0.8)
    parser.add_argument('--reg-lambda', type=float, default=0.0)
    parser.add_argument('--neg-frac', type=float, default=0.01)
    # SageMaker folders
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../data'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', 'output'))
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train = pd.read_csv(os.path.join(args.data_dir, 'train-metadata.csv'), low_memory=False)
    test = pd.read_csv(os.path.join(args.data_dir, 'test-metadata.csv'))
    folds = pd.read_csv(os.path.join(args.data_dir, 'folds.csv'))
    train = train.merge(folds[['isic_id', 'fold']], on='isic_id', how='left')
    df, features = build_features(train, test)
    print(f'{len(df)} rows, {len(features)} features')

    params = {
        'objective': 'binary',
        'learning_rate': args.learning_rate,
        'n_estimators': args.n_estimators,
        'num_leaves': args.num_leaves,
        'min_child_samples': args.min_child_samples,
        'subsample': args.subsample,
        'subsample_freq': 1,
        'colsample_bytree': args.colsample_bytree,
        'reg_lambda': args.reg_lambda,
        'random_state': SEED,
        'verbose': -1,
    }
    print('params:', params, 'neg_frac:', args.neg_frac)

    oof = np.zeros(len(df))
    paucs, aucs = [], []
    for fold in range(N_FOLDS):
        trn = df[df['fold'] != fold]
        val = df[df['fold'] == fold]
        # downsample the benign lesions in the training folds only
        trn = pd.concat([trn[trn['target'] == 1],
                         trn[trn['target'] == 0].sample(frac=args.neg_frac, random_state=SEED)])

        model = lgb.LGBMClassifier(**params)
        model.fit(trn[features], trn['target'])

        pred = model.predict_proba(val[features])[:, 1]
        oof[(df['fold'] == fold).to_numpy()] = pred

        pauc = p_auc_tpr(val['target'], pred, 0.80)
        auc = roc_auc_score(val['target'], pred)
        paucs.append(pauc)
        aucs.append(auc)
        print(f'fold {fold}: pAUC={pauc:.4f}  AUC={auc:.4f}')

        model.booster_.save_model(os.path.join(args.model_dir, f'lgbm_fold{fold}.txt'))

    # these two lines are read by SageMaker (metric definitions)
    print(f'cv_pauc: {np.mean(paucs):.4f};')
    print(f'cv_pauc_std: {np.std(paucs):.4f};')
    print(f'cv_auc: {np.mean(aucs):.4f};')

    meta = {
        'features': features,
        'categories': {col: df[col].cat.categories.tolist() for col in CAT_COLS},
        'params': params,
        'neg_frac': args.neg_frac,
        'fold_paucs': paucs,
    }
    with open(os.path.join(args.model_dir, 'lgbm_meta.json'), 'w') as f:
        json.dump(meta, f)

    out = df[['isic_id', 'patient_id', 'target', 'fold']].copy()
    out['oof'] = oof
    out.to_csv(os.path.join(args.output_dir, 'oof_lgbm.csv'), index=False)


if __name__ == '__main__':
    main()
