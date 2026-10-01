"""SageMaker training script: ResNet18 on the lesion images with 5-fold patient CV.

Same code as in 04_cnn.ipynb as a script. For every fold it trains on all malignant lesions + a sample of
the benign ones, predicts the full validation fold, prints the pAUC and saves the fold model.
"""
import argparse
import io
import os
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision
from einops import rearrange
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from isic_pauc import p_auc_tpr

SEED = 42
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def parse_args():
    parser = argparse.ArgumentParser()
    # hyperparameters
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--neg-per-pos', type=int, default=20)
    parser.add_argument('--img-size', type=int, default=128)
    parser.add_argument('--folds', type=str, default='0,1,2,3,4')
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--max-val', type=int, default=0, help='only for quick local tests: limit the validation rows')
    # SageMaker folders
    parser.add_argument('--data-dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', '../data'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model'))
    parser.add_argument('--output-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', 'output'))
    return parser.parse_args()


def get_transforms(img_size):
    train_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tfms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, val_tfms


class LesionDataset(Dataset):
    def __init__(self, df, img_path, tfms):
        self.ids = df['isic_id'].to_numpy()
        self.targets = df['target'].to_numpy().astype('float32')
        self.img_path = img_path
        self.tfms = tfms
        self.h5 = None

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        # opened here and not in __init__, an open h5py file can't be copied to the DataLoader workers
        if self.h5 is None:
            self.h5 = h5py.File(self.img_path, 'r')
        img = Image.open(io.BytesIO(self.h5[self.ids[i]][()])).convert('RGB')
        return self.tfms(img), self.targets[i]


def make_model(device, pretrained=True):
    weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = torchvision.models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model.to(device)


def sample_train(trn, neg_per_pos, fold):
    pos = trn[trn['target'] == 1]
    neg = trn[trn['target'] == 0].sample(n=len(pos) * neg_per_pos, random_state=SEED + fold)
    return pd.concat([pos, neg])


def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            logits = rearrange(model(x), 'b 1 -> b')
            loss = criterion(logits, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total += loss.item() * len(x)
    return total / len(loader.dataset)


@torch.no_grad()
def predict(model, loader, device):
    # float32 (no autocast), in float16 a lot of lesions got exactly the same score
    model.eval()
    preds = []
    for x, _ in loader:
        logits = rearrange(model(x.to(device)), 'b 1 -> b')
        preds.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(preds)


def main():
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device, torch.cuda.get_device_name(0) if device.type == 'cuda' else '')

    img_path = os.path.join(args.data_dir, 'train-image.hdf5')
    folds = pd.read_csv(os.path.join(args.data_dir, 'folds.csv'))
    train_tfms, val_tfms = get_transforms(args.img_size)
    fold_list = [int(f) for f in args.folds.split(',')]

    oof = np.full(len(folds), np.nan)
    paucs, aucs = [], []
    for fold in fold_list:
        trn = folds[folds['fold'] != fold]
        val = folds[folds['fold'] == fold]
        if args.max_val:
            val = pd.concat([val[val['target'] == 1], val[val['target'] == 0].head(args.max_val)])
        trn_s = sample_train(trn, args.neg_per_pos, fold)

        train_loader = DataLoader(LesionDataset(trn_s, img_path, train_tfms), batch_size=args.batch_size,
                                  shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(LesionDataset(val, img_path, val_tfms), batch_size=256,
                                shuffle=False, num_workers=args.num_workers)

        torch.manual_seed(SEED + fold)
        model = make_model(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        scaler = torch.amp.GradScaler(device.type, enabled=device.type == 'cuda')

        for epoch in range(args.epochs):
            t0 = time.time()
            loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, device)
            print(f'fold {fold} epoch {epoch}: train loss {loss:.4f} ({time.time() - t0:.0f}s)')

        t0 = time.time()
        pred = predict(model, val_loader, device)
        oof[val.index.to_numpy()] = pred
        pauc = p_auc_tpr(val['target'], pred, 0.80)
        auc = roc_auc_score(val['target'], pred)
        paucs.append(pauc)
        aucs.append(auc)
        print(f'fold {fold}: pAUC={pauc:.4f}  AUC={auc:.4f}  (predict {time.time() - t0:.0f}s)')

        torch.save(model.state_dict(), os.path.join(args.model_dir, f'cnn_fold{fold}.pt'))

    # these lines are read by SageMaker (metric definitions)
    print(f'cv_pauc: {np.mean(paucs):.4f};')
    print(f'cv_pauc_std: {np.std(paucs):.4f};')
    print(f'cv_auc: {np.mean(aucs):.4f};')

    out = folds[['isic_id', 'patient_id', 'target', 'fold']].copy()
    out['oof_cnn'] = oof
    out.to_csv(os.path.join(args.output_dir, 'oof_cnn.csv'), index=False)


if __name__ == '__main__':
    main()
