"""Wider feature set for 08_boosting_ensemble.ipynb: the v3 features, 24 more ratios and patient-normalized
versions of almost every numeric column."""
import numpy as np
import pandas as pd

from features import CAT_COLS, ENG_COLS, add_features, drop_leaks

# no patient-normalized version of these: age is the same for all lesions of a patient and x/y/z is only a position
NO_NORM = ['age_approx', 'tbp_lv_x', 'tbp_lv_y', 'tbp_lv_z', 'size_age', 'size_per_age', 'nevi_conf_per_age']


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


def build_wide_features(train, test):
    """Returns the dataframe and the list of feature columns (126 features)."""
    df = drop_leaks(train, test)

    num_cols = df.select_dtypes('number').drop(columns=['target', 'fold'], errors='ignore').columns.tolist()
    for col in CAT_COLS:
        df[col] = df[col].astype('category')

    df = add_features(df)
    df, wide_cols = add_wide_features(df)
    norm_src = [col for col in num_cols + ENG_COLS + wide_cols if col not in NO_NORM]
    df, patient_cols = add_patient_norms(df, norm_src)

    features = num_cols + CAT_COLS + ENG_COLS + wide_cols + patient_cols
    return df, features
