"""Metadata features for LightGBM (same as in 03_lightgbm.ipynb)."""
import pandas as pd

CAT_COLS = ['anatom_site_general', 'sex', 'tbp_tile_type', 'tbp_lv_location']

ENG_COLS = ['hue_contrast', 'chroma_contrast', 'elongation', 'texture_contrast',
            'size_and_color_irregularity', 'ABC_score']

NORM_SRC_COLS = [
    'tbp_lv_H', 'ABC_score', 'hue_contrast', 'size_and_color_irregularity', 'elongation',
    'tbp_lv_deltaB', 'tbp_lv_deltaA',
    'clin_size_long_diam_mm', 'tbp_lv_areaMM2', 'tbp_lv_color_std_mean', 'tbp_lv_norm_border', 'tbp_lv_norm_color',
]


def drop_leaks(train, test):
    # columns that are only in the training data (diagnosis, pathology, ...) except the target and the fold
    leak_cols = [col for col in train.columns if col not in test.columns and col not in ('target', 'fold')]
    return train.drop(columns=leak_cols)


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
    # how unusual a lesion is compared to the same patient's other lesions ("ugly duckling")
    df = df.copy()
    new_cols = []
    for col in cols:
        p_mean = df.groupby('patient_id')[col].transform('mean')
        p_std = df.groupby('patient_id')[col].transform('std')
        # patients with 1 lesion get NaN (no std), LightGBM handles that
        df[f'{col}_pnorm'] = (df[col] - p_mean) / (p_std + 1e-6)
        new_cols.append(f'{col}_pnorm')

    df['patient_lesion_count'] = df.groupby('patient_id')['patient_id'].transform('count')
    new_cols.append('patient_lesion_count')
    return df, new_cols


def build_features(train, test):
    """Everything from 03_lightgbm v3. Returns the dataframe and the list of feature columns."""
    df = drop_leaks(train, test)

    num_cols = df.select_dtypes('number').drop(columns=['target', 'fold'], errors='ignore').columns.tolist()
    for col in CAT_COLS:
        df[col] = df[col].astype('category')

    df = add_features(df)
    df, patient_cols = add_patient_features(df, NORM_SRC_COLS)

    features = num_cols + CAT_COLS + ENG_COLS + patient_cols
    return df, features
