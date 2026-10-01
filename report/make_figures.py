"""Makes the data exploration figures for the report (run from the project folder)."""
import matplotlib.pyplot as plt
import pandas as pd

train = pd.read_csv('data/train-metadata.csv', low_memory=False)

# class balance + lesions per patient
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
counts = train['target'].value_counts().sort_index()
axes[0].bar(['benign', 'malignant'], counts.values, color=['tab:blue', 'tab:red'])
axes[0].set_yscale('log')
axes[0].set_ylabel('number of lesions (log scale)')
axes[0].set_title('Class balance')
for i, v in enumerate(counts.values):
    axes[0].text(i, v, f'{v:,}', ha='center', va='bottom')

per_patient = train.groupby('patient_id').size()
axes[1].hist(per_patient, bins=40, color='tab:blue')
axes[1].axvline(per_patient.median(), color='black', linestyle='--', label=f'median = {per_patient.median():.0f}')
axes[1].set_xlabel('lesions per patient')
axes[1].set_ylabel('number of patients')
axes[1].set_title('Lesions per patient')
axes[1].legend()
plt.tight_layout()
plt.savefig('report/figures/class_balance_and_patients.png', dpi=120)

# two strong features by class
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for ax, col, name in [(axes[0], 'tbp_lv_H', 'hue inside the lesion (tbp_lv_H)'),
                      (axes[1], 'clin_size_long_diam_mm', 'longest diameter in mm')]:
    lo, hi = train[col].quantile([0.001, 0.999])
    for target, label, color in [(0, 'benign', 'tab:blue'), (1, 'malignant', 'tab:red')]:
        ax.hist(train.loc[train['target'] == target, col], bins=40, range=(lo, hi), density=True, alpha=0.5,
                label=label, color=color)
    ax.set_xlabel(name)
    ax.set_ylabel('density')
    ax.legend()
axes[0].set_title('Malignant lesions have a lower hue (more red)')
axes[1].set_title('Malignant lesions are bigger')
plt.tight_layout()
plt.savefig('report/figures/features_by_class.png', dpi=120)
print('saved')
