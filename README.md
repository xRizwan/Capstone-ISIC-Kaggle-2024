# Skin Cancer Detection from 3D Total Body Photos (ISIC 2024)

Capstone project for the Udacity AWS Machine Learning Engineer Nanodegree.

The goal is to rank the skin lesions of a patient by how likely they are to be malignant, so that a doctor can
look at the most suspicious ones first. The data is from the Kaggle competition
[ISIC 2024 - Skin Cancer Detection with 3D-TBP](https://www.kaggle.com/competitions/isic-2024-challenge):
401,059 lesion crops from 3D total body photos with metadata, of which only 393 are malignant.

I picked this project because it is a real problem with messy data: a very rare positive class, many lesions per
patient, and both images and tabular data.

## Results

All scores are the partial AUC above 80% TPR (the competition metric, maximum 0.2) from 5-fold cross-validation
with the folds split by patient.

| Model | pAUC (mean ± std) |
|---|---|
| Benchmark: `tbp_lv_H` alone | 0.0809 ± 0.0247 |
| Benchmark: logistic regression | 0.1116 ± 0.0205 |
| LightGBM on the metadata | 0.1644 ± 0.0079 |
| CNN (ResNet18) on the images | 0.1432 ± 0.0134 |
| **Final model: blend of 70% LightGBM + 30% CNN** | **0.1691 ± 0.0067** |

- The final model beats the logistic regression benchmark on all 5 folds.
- The most important feature is the hue of a lesion compared to the same patient's other lesions (the "ugly
  duckling" rule dermatologists use).
- On the hidden test set of the Kaggle competition (late submission) the final model scored 0.1706 on the
  public part and 0.1590 on the private part. The top of the leaderboard is very close together (best public
  score 0.189, silver medal on the private part from about 0.1705), so the model is a good step above the
  benchmarks but far from the best solutions.
- After the review I added XGBoost, CatBoost and more features (`08_boosting_ensemble`). In cross-validation the
  gain is inside the noise (0.1705 vs 0.1691). On Kaggle it scored 0.1745 public and 0.1604 private.
- `09_questions` looks at what the score doesn't show. The main finding: compared only to benign lesions that
  were also biopsied, the pAUC drops from 0.169 to 0.118. The model mostly separates cancers from lesions nobody
  was worried about.
- The models were trained and tuned again on SageMaker and the final model was deployed to an endpoint, tested and
  deleted. The whole AWS part cost less than $1.

The full write-up is in `report/report.pdf` and the proposal in `proposal.pdf`.

## Files

| File | What it does |
|---|---|
| `notebooks/01_eda.ipynb` | Data exploration: class balance, patients, leaky columns, single-feature AUCs |
| `notebooks/02_folds_and_benchmark.ipynb` | 5 patient-based folds and the two benchmarks |
| `notebooks/03_lightgbm.ipynb` | LightGBM on the metadata: downsampling, engineered and patient-normalized features |
| `notebooks/04_cnn.ipynb` | ResNet18 on the images (trained locally on a laptop GPU) |
| `notebooks/05_blend.ipynb` | Blending LightGBM and the CNN on ranks |
| `notebooks/06_stacking.ipynb` | Stacking: the CNN score as a LightGBM feature |
| `notebooks/07_sagemaker.ipynb` | SageMaker: training jobs, hyperparameter tuning, endpoint, test, cleanup, cost |
| `notebooks/08_boosting_ensemble.ipynb` | Added after the review: LightGBM, XGBoost and CatBoost with more features and seeds |
| `notebooks/09_questions.ipynb` | Added after the review: score noise, biopsied lesions, unseen hospitals, lesion count, subgroups, cancer types |
| `src/features.py` | The metadata features (used by the notebooks and the scripts) |
| `src/isic_pauc.py` | The official competition metric, copied from the [ISIC GitHub repository](https://github.com/ISIC-Research/Challenge-2024-Metrics) |
| `src/train_lgbm.py` | SageMaker training script for LightGBM (also used for the tuning) |
| `src/train_cnn.py` | SageMaker training script for the CNN |
| `src/inference.py` | Endpoint code for the blended model |
| `src/kaggle_inference.py` | Script that runs the final model on the hidden Kaggle test set |
| `src/features_wide.py` | The wider feature set for `08_boosting_ensemble` |
| `src/kaggle_inference_boosting.py` | Kaggle script for the boosting ensemble + CNN |
| `src/requirements.txt` | Extra packages for the SageMaker containers |
| `report/report.md`, `report/report.pdf` | The project report |
| `report/figures/`, `report/make_figures.py` | Figures of the report |
| `proposal.md`, `proposal.pdf` | The capstone proposal |

## Data

The data is not in this repository. It is too big and the license (CC BY-NC 4.0) doesn't allow re-uploading it.
After accepting the competition rules on Kaggle it can be downloaded with the Kaggle CLI:

```
kaggle competitions download -c isic-2024-challenge -f train-metadata.csv -p data
kaggle competitions download -c isic-2024-challenge -f test-metadata.csv -p data
kaggle competitions download -c isic-2024-challenge -f train-image.hdf5 -p data
```

The notebooks expect the CSV files in `data/` (unzipped). `notebooks/04_cnn.ipynb` and `07_sagemaker.ipynb` have
the path to `train-image.hdf5` in a variable called `IMG_PATH` at the top. On my laptop it points to another
drive, so it has to be changed.

## Setup

Python 3.11. The project uses [uv](https://docs.astral.sh/uv/):

```
uv sync
uv run jupyter lab
```

`uv sync` installs everything from `pyproject.toml`. PyTorch is installed with CUDA 12.8 for an NVIDIA GPU.

Libraries: pandas, numpy, scikit-learn, scipy, lightgbm, xgboost, catboost, pytorch, torchvision, einops, h5py,
pillow, matplotlib, seaborn, jupyterlab, kaggle.

The notebooks have to be run in order, because the later ones use files saved by the earlier ones (`folds.csv`
and the out-of-fold predictions in `data/`).

`07_sagemaker.ipynb` needs an AWS account with a SageMaker execution role and also the `sagemaker` and `boto3`
packages. I installed those in a separate environment, because the SageMaker SDK needs older versions of some
packages. It uses the PyTorch 2.5.1 containers, `ml.m5.xlarge` and `ml.g4dn.xlarge` for training and `ml.m5.large`
for the endpoint.

## Acknowledgements

- The dataset: Kurtansky et al., "The SLICE-3D dataset: 400,000 skin lesion image crops extracted from 3D TBP for
  skin cancer detection", Scientific Data, 2024. https://doi.org/10.1038/s41597-024-03743-w
- The competition and the metric: International Skin Imaging Collaboration (ISIC) and Kaggle.
- The idea of comparing a lesion to the patient's other lesions and of combining an image model with a gradient
  boosting model on the metadata comes from the discussions and public notebooks of the competition.
- Udacity for the project template and the AWS credits.
