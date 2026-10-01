"""SageMaker endpoint code for the final model: blend of LightGBM (metadata) and the CNN (images).

Input (JSON): {"lesions": [{<metadata columns>, "image": "<base64 JPEG>"}, ...]}
All lesions of ONE patient should be sent together, because the "ugly duckling" features compare each
lesion to the patient's other lesions.

Output (JSON): one entry per lesion with the scores of both models and the blended risk score (0-1).
"""
import base64
import io
import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from PIL import Image
from torchvision import transforms

from features import CAT_COLS, NORM_SRC_COLS, add_features, add_patient_features

N_FOLDS = 5
IMG_SIZE = 128
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

val_tfms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])


def model_fn(model_dir):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    with open(os.path.join(model_dir, 'lgbm_meta.json')) as f:
        meta = json.load(f)
    with open(os.path.join(model_dir, 'blend.json')) as f:
        blend = json.load(f)

    boosters = [lgb.Booster(model_file=os.path.join(model_dir, f'lgbm_fold{i}.txt')) for i in range(N_FOLDS)]

    cnns = []
    for i in range(N_FOLDS):
        cnn = torchvision.models.resnet18(weights=None)
        cnn.fc = nn.Linear(cnn.fc.in_features, 1)
        cnn.load_state_dict(torch.load(os.path.join(model_dir, f'cnn_fold{i}.pt'), map_location=device))
        cnns.append(cnn.to(device).eval())

    # sorted out-of-fold scores of both models, used to turn a new score into a rank between 0 and 1
    ref = np.load(os.path.join(model_dir, 'reference.npz'))

    return {'meta': meta, 'w': blend['w'], 'boosters': boosters, 'cnns': cnns,
            'ref_lgbm': ref['lgbm'], 'ref_cnn': ref['cnn'], 'device': device}


def input_fn(request_body, content_type='application/json'):
    if content_type != 'application/json':
        raise ValueError(f'Unsupported content type: {content_type}')
    return json.loads(request_body)


def to_rank(scores, reference):
    # share of the reference (out-of-fold) scores that are lower than or equal to this score
    return np.searchsorted(reference, scores, side='right') / len(reference)


def predict_lgbm(lesions, model):
    meta = model['meta']
    df = pd.DataFrame([{k: v for k, v in lesion.items() if k != 'image'} for lesion in lesions])

    for col in CAT_COLS:
        df[col] = pd.Categorical(df[col], categories=meta['categories'][col])
    for col in df.columns:
        if col not in CAT_COLS and col not in ('isic_id', 'patient_id'):
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df = add_features(df)
    df, _ = add_patient_features(df, NORM_SRC_COLS)

    X = df.reindex(columns=meta['features'])
    for col in CAT_COLS:
        X[col] = df[col]
    return np.mean([booster.predict(X) for booster in model['boosters']], axis=0)


@torch.no_grad()
def predict_cnn(lesions, model):
    images = [Image.open(io.BytesIO(base64.b64decode(lesion['image']))).convert('RGB') for lesion in lesions]
    preds = []
    for start in range(0, len(images), 64):
        x = torch.stack([val_tfms(img) for img in images[start:start + 64]]).to(model['device'])
        probs = [torch.sigmoid(cnn(x)[:, 0]) for cnn in model['cnns']]
        preds.append(torch.stack(probs).mean(dim=0).cpu().numpy())
    return np.concatenate(preds)


def predict_fn(data, model):
    lesions = data['lesions']

    lgbm_score = predict_lgbm(lesions, model)
    cnn_score = predict_cnn(lesions, model)

    lgbm_rank = to_rank(lgbm_score, model['ref_lgbm'])
    cnn_rank = to_rank(cnn_score, model['ref_cnn'])
    risk = model['w'] * lgbm_rank + (1 - model['w']) * cnn_rank

    return [{'isic_id': lesion.get('isic_id'),
             'lgbm_score': float(lgbm_score[i]),
             'cnn_score': float(cnn_score[i]),
             'lgbm_rank': float(lgbm_rank[i]),
             'cnn_rank': float(cnn_rank[i]),
             'risk_score': float(risk[i])}
            for i, lesion in enumerate(lesions)]


def output_fn(prediction, accept='application/json'):
    return json.dumps(prediction)
