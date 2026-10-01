# Capstone Proposal - Skin Cancer Detection from 3D Total Body Photos (ISIC 2024)

## Domain Background

Skin cancer is the most common type of cancer, and melanoma is the deadliest form of it. Melanoma is usually
curable when it is found early: the five-year survival rate for U.S. patients whose melanoma is detected early
is about 99%. Once it spreads, it becomes much harder to treat, and an estimated 8,510 people in the U.S. will
die of melanoma in 2026 [1]. This is why early detection is so important.

Deep learning has already shown good results here. Esteva et al. [2] trained a CNN that classified skin cancer
about as well as dermatologists. The International Skin Imaging Collaboration (ISIC) has run skin cancer
challenges since 2016, but the earlier ones used dermoscopy images, which are close-up photos taken by
dermatologists with a special magnifying device. The ISIC 2024 challenge [4] uses the SLICE-3D dataset [3]
instead, where the lesion crops come from 3D total body photography (3D-TBP), which captures the whole skin
surface of a patient in one session. The crops are lower quality and look more like phone photos, so a model
trained on them could help in places where there is no dermatologist, like primary care or telehealth.

## Problem Statement

A 3D total body photo captures hundreds of lesions per patient (about 385 on average in this dataset), and
there aren't enough dermatologists to review every one of them. The goal of this project is triage: rank the
lesions so that doctors check the most suspicious ones first.

This is a binary classification task. The input for each lesion is its image crop plus its metadata (age, sex,
body site and the scanner measurements), and the output is the probability that the lesion is malignant.
Performance is measured with the pAUC above 80% TPR using patient-based cross-validation. The project can be
repeated because the data is public on Kaggle and the split uses a fixed random seed.

## Datasets and Inputs

The data comes from the Kaggle competition "ISIC 2024 - Skin Cancer Detection with 3D-TBP" [4] and is released
under the CC BY-NC 4.0 license. The lesions come from 3D total body photos collected by several hospitals, and
the dataset is described in the SLICE-3D paper [3].

There are 401,059 rows, and each row is one lesion. 400,666 lesions are benign and only 393 are malignant. The
lesions come from 1,042 patients, and only 259 of them (about 25%) have at least one malignant lesion. Each
lesion has two inputs:

- **Image:** a small JPEG crop of the lesion (around 3 KB). The size isn't fixed (the ones I checked were
  between 109x109 and 147x147 pixels), so the images will be resized before training.
- **Metadata:** age, sex, body site and around 35 numeric measurements from the 3D scanner (`tbp_lv_*`
  columns) that describe the lesion's color, size, shape, border and location.

Some columns are only in the training data. For example, the diagnosis columns (`iddx_*`) and the pathology
results (`mel_*`) are only known after a biopsy, so they would leak the answer. All of these columns will be
dropped except `target`. `age_approx` has 2,798 missing values, which will be filled with the mean for models
that can't handle missing values.

The data will be split by patient. Photos of the same patient share the same skin tone, age and camera
conditions, so with a random split the model could partly learn to recognize the patient instead of the
lesion. Splitting by patient means the validation set only has patients the model has never seen. The real
test set is hidden on Kaggle, so the models will be evaluated with patient-based cross-validation.

## Solution Statement

The solution will combine a model on the metadata with a model on the images:

1. **LightGBM on the metadata**, plus engineered features: combinations of existing columns and features that
   compare each lesion to the same patient's other lesions (the "ugly duckling" idea used by dermatologists).
2. **A CNN on the images** (likely a pretrained ResNet) fine-tuned on SageMaker. It will be trained on all 393
   malignant lesions plus a random sample of the benign ones, to deal with the imbalance and keep training cheap.
3. **Combining both**, first by blending (averaging the ranked predictions). If that helps, I will try stacking,
   where the CNN's prediction is added as an extra LightGBM feature. For stacking, one CNN is trained per fold
   and predicts the fold it didn't see, so the CNN scores don't leak the labels.
4. **Hyperparameter optimization** on SageMaker for the best model.

Every step will be evaluated on the same 5 patient-based folds with the pAUC, so each change can be compared
with the previous step and with the benchmarks.

## Benchmark Model

I set up two benchmarks, evaluated on the same 5 patient-based folds (`StratifiedGroupKFold`) that the final
model will use:

1. **Single feature (`tbp_lv_H`):** the hue of the lesion, where lower values are more red and higher values
   are more brown. It was the strongest single feature in my data exploration, and malignant lesions tend to
   have a lower hue. It needs no training, the lesions are just ranked by the flipped hue value.
2. **Logistic regression** on all 34 numeric metadata features, with mean imputation and standard scaling
   fitted only on the training folds. It shows what a basic ML model can do.

| Benchmark | pAUC (mean ± std) | AUC (mean ± std) |
|---|---|---|
| `tbp_lv_H` (no training) | 0.0809 ± 0.0247 | 0.8035 ± 0.0490 |
| Logistic regression | 0.1116 ± 0.0205 | 0.8684 ± 0.0363 |

The logistic regression beat the single feature on all 5 folds. The final model should beat its pAUC of 0.1116
on most folds, not just on average, because the scores vary a lot between folds (from 0.045 to 0.120 for
`tbp_lv_H`). For context, good solutions in the Kaggle competition reached a pAUC of about 0.15-0.17.

## Evaluation Metrics

With about 1 malignant lesion for every 1,000 benign ones, accuracy can't be used. A model that predicts
"benign" for every lesion would get 99.9% accuracy but would not find a single cancer.

Instead I will use the ROC-AUC. The ROC curve plots the true positive rate (TPR, the share of cancers caught)
against the false positive rate (FPR, the share of benign lesions wrongly flagged) for every threshold. The area
under it is the probability that a random malignant lesion gets a higher score than a random benign one, so it
is not affected by the class imbalance and doesn't depend on a threshold.

The main metric will be the partial AUC (pAUC) above 80% TPR, the official metric of the competition [4], [5].
It only counts the part of the ROC curve where at least 80% of the cancers are caught, which makes sense for
screening, where missing a cancer is much worse than a false alarm. The pAUC ranges from 0 to 0.2. The full
ROC-AUC will be reported as a secondary metric because it's easier to understand.

## Project Design

1. **Data:** Download the metadata and images from Kaggle and upload them to S3 for SageMaker.
2. **Data preparation:** Drop the leaky columns and create the 5 patient-based folds with a fixed seed, saved
   to a file so every model uses the same split. For the CNN, resize and augment the images (flips, rotations)
   and sample the benign lesions.
3. **EDA and benchmarks:** Class balance, lesions per patient, missing values, single-feature AUCs and the two
   benchmarks. This step is already done.
4. **LightGBM:** Encode the categorical columns (sex, body site) and add the engineered features.
5. **CNN:** Fine-tune a pretrained ResNet in a SageMaker training job on a GPU (spot instances if possible, to
   save budget). One CNN per fold, each predicting all the lesions of its validation fold.
6. **Combining:** Blending first, then stacking if blending helps.
7. **HPO:** SageMaker hyperparameter tuning with the pAUC as the objective.
8. **Evaluation:** Compare every step with the benchmarks on the same folds (pAUC and AUC, mean ± std and per
   fold). These results go into the refinement part of the final report.
9. **Deployment:** Deploy the final model to a SageMaker endpoint, test it with a few lesion images, and
   delete it afterwards to save budget.

## References

[1] Skin Cancer Foundation. Skin Cancer Facts & Statistics.
https://www.skincancer.org/skin-cancer-information/skin-cancer-facts/

[2] Esteva, A., Kuprel, B., Novoa, R. A., Ko, J., Swetter, S. M., Blau, H. M., & Thrun, S. (2017).
Dermatologist-level classification of skin cancer with deep neural networks. Nature, 542(7639), 115-118.
https://doi.org/10.1038/nature21056

[3] Kurtansky, N. R., D'Alessandro, B. M., Gillis, M. C., et al. (2024). The SLICE-3D dataset: 400,000 skin
lesion image crops extracted from 3D TBP for skin cancer detection. Scientific Data, 11.
https://doi.org/10.1038/s41597-024-03743-w

[4] ISIC 2024 - Skin Cancer Detection with 3D-TBP. Kaggle.
https://www.kaggle.com/competitions/isic-2024-challenge

[5] ISIC Research. ISIC 2024 Challenge primary metric (pAUC).
https://github.com/ISIC-Research/Challenge-2024-Metrics

