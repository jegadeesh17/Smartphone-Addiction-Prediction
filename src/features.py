"""
Advanced Feature Engineering Module for Smartphone Addiction Prediction (Kaggle S6E8).
Includes robust missingness preservation, filled arithmetic representations,
domain time-budget dynamics, fragmentation metrics, interaction cross-products,
and multi-cohort GroupBy aggregations.
"""

import numpy as np
import pandas as pd


def create_features(df_train: pd.DataFrame, df_test: pd.DataFrame = None) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Extracts high-signal domain features and cohort statistics across combined train+test splits.
    
    Returns:
        tr_fe: Transformed training set
        te_fe: Transformed test set (or None)
        feat_cols: List of final feature column names
    """
    if df_test is not None:
        df_all = pd.concat([df_train.assign(is_train=1), df_test.assign(is_train=0, addicted_label=np.nan)], ignore_index=True)
    else:
        df_all = df_train.assign(is_train=1).copy()

    raw_num = [
        'age', 'daily_screen_time_hours', 'social_media_hours', 'gaming_hours',
        'work_study_hours', 'sleep_hours', 'notifications_per_day', 'app_opens_per_day', 'weekend_screen_time'
    ]
    raw_cat = ['gender', 'stress_level', 'academic_work_impact']

    # 1. Explicit Missingness Flags & Count
    for c in raw_num + raw_cat:
        df_all[f'{c}_isna'] = df_all[c].isna().astype(np.int8)
    df_all['num_missing'] = df_all[[f'{c}_isna' for c in raw_num + raw_cat]].sum(axis=1)

    # 2. Categorical Encodings (Standard & Composite)
    df_all['gender_code'] = df_all['gender'].astype('category').cat.codes
    df_all['stress_code'] = df_all['stress_level'].map({'Low': 0, 'Medium': 1, 'High': 2}).fillna(-1).astype(int)
    df_all['impact_code'] = df_all['academic_work_impact'].map({'No': 0, 'Yes': 1}).fillna(-1).astype(int)

    df_all['gender_stress'] = df_all['gender'].astype(str) + '_' + df_all['stress_level'].astype(str)
    df_all['gender_stress_code'] = df_all['gender_stress'].astype('category').cat.codes

    df_all['stress_impact'] = df_all['stress_level'].astype(str) + '_' + df_all['academic_work_impact'].astype(str)
    df_all['stress_impact_code'] = df_all['stress_impact'].astype('category').cat.codes

    # 3. Frequency Encodings
    for c in ['age', 'gender', 'stress_level', 'academic_work_impact', 'sleep_hours', 'daily_screen_time_hours', 'gender_stress', 'stress_impact']:
        df_all[f'{c}_freq'] = df_all[c].map(df_all[c].value_counts(normalize=True))

    # 4. Safe filled variables for domain arithmetic (avoids cascading NaNs)
    soc_f = df_all['social_media_hours'].fillna(0.0)
    game_f = df_all['gaming_hours'].fillna(0.0)
    screen_f = df_all['daily_screen_time_hours'].fillna(df_all['daily_screen_time_hours'].median())
    sleep_f = df_all['sleep_hours'].fillna(df_all['sleep_hours'].median())
    work_f = df_all['work_study_hours'].fillna(df_all['work_study_hours'].median())
    weekend_f = df_all['weekend_screen_time'].fillna(df_all['weekend_screen_time'].median())
    opens_f = df_all['app_opens_per_day'].fillna(df_all['app_opens_per_day'].median())
    notifs_f = df_all['notifications_per_day'].fillna(df_all['notifications_per_day'].median())

    # 5. Core Domain Time-Budget Features
    df_all['recreational_hours'] = soc_f + game_f
    df_all['recreational_to_screen'] = df_all['recreational_hours'] / (screen_f + 1e-5)
    df_all['non_recreational_screen'] = np.maximum(0.0, screen_f - df_all['recreational_hours'])
    df_all['total_accounted_hours'] = screen_f + work_f + sleep_f
    df_all['unaccounted_hours'] = 24.0 - df_all['total_accounted_hours']
    df_all['waking_hours'] = 24.0 - sleep_f
    df_all['screen_fraction_of_waking'] = screen_f / df_all['waking_hours'].clip(lower=1.0)
    df_all['screen_fraction_of_day'] = screen_f / 24.0

    # Weekend Surge Dynamics
    df_all['weekend_vs_weekday_diff'] = weekend_f - screen_f
    df_all['weekend_to_weekday_ratio'] = weekend_f / (screen_f + 1e-5)
    df_all['weighted_weekly_screen'] = (5.0 * screen_f + 2.0 * weekend_f) / 7.0

    # Fragmentation, Interruption & Habit Intensity
    df_all['app_opens_per_screen_hour'] = opens_f / (screen_f + 1e-5)
    df_all['avg_unlock_minutes'] = (screen_f * 60.0) / (opens_f + 1e-5)
    df_all['notifications_per_app_open'] = notifs_f / (opens_f + 1e-5)
    df_all['notifications_per_screen_hour'] = notifs_f / (screen_f + 1e-5)
    df_all['interaction_density'] = (opens_f * notifs_f) / 1000.0

    # Cross Interactions & Proportions
    df_all['screen_to_sleep_ratio'] = screen_f / (sleep_f + 1e-5)
    df_all['work_to_sleep_ratio'] = work_f / (sleep_f + 1e-5)
    df_all['work_to_screen_ratio'] = work_f / (screen_f + 1e-5)
    df_all['social_to_gaming_ratio'] = (soc_f + 1e-5) / (game_f + 1e-5)
    df_all['social_share_of_screen'] = soc_f / (screen_f + 1e-5)
    df_all['gaming_share_of_screen'] = game_f / (screen_f + 1e-5)

    # Cross-Product Interactions
    df_all['screen_x_social'] = screen_f * soc_f
    df_all['screen_x_weekend'] = screen_f * weekend_f
    df_all['screen_x_opens'] = screen_f * (opens_f / 100.0)

    # 6. Multi-Cohort GroupBy Aggregations
    # Cohort 1: (Age, Gender) Screen Time Distribution
    grp_ag = df_all.groupby(['age', 'gender'])['daily_screen_time_hours'].agg(['mean', 'std']).reset_index()
    grp_ag.columns = ['age', 'gender', 'screen_by_ag_mean', 'screen_by_ag_std']
    df_all = df_all.merge(grp_ag, on=['age', 'gender'], how='left')
    df_all['screen_diff_ag_mean'] = screen_f - df_all['screen_by_ag_mean']
    df_all['screen_zscore_ag'] = df_all['screen_diff_ag_mean'] / (df_all['screen_by_ag_std'].fillna(1.0) + 1e-5)

    # Cohort 2: (Stress Level, Academic Impact) Sleep Distribution
    grp_stress = df_all.groupby(['stress_level', 'academic_work_impact'])['sleep_hours'].agg(['mean', 'std']).reset_index()
    grp_stress.columns = ['stress_level', 'academic_work_impact', 'sleep_by_stress_mean', 'sleep_by_stress_std']
    df_all = df_all.merge(grp_stress, on=['stress_level', 'academic_work_impact'], how='left')
    df_all['sleep_diff_stress_mean'] = sleep_f - df_all['sleep_by_stress_mean']
    df_all['sleep_zscore_stress'] = df_all['sleep_diff_stress_mean'] / (df_all['sleep_by_stress_std'].fillna(1.0) + 1e-5)

    # Cohort 3: (Gender, Stress Level) Behavioral Means
    grp_gs = df_all.groupby(['gender', 'stress_level'])[['app_opens_per_day', 'notifications_per_day', 'daily_screen_time_hours']].agg('mean').reset_index()
    grp_gs.columns = ['gender', 'stress_level', 'opens_by_gs_mean', 'notifs_by_gs_mean', 'screen_by_gs_mean']
    df_all = df_all.merge(grp_gs, on=['gender', 'stress_level'], how='left')
    df_all['opens_diff_gs_mean'] = opens_f - df_all['opens_by_gs_mean']
    df_all['notifs_diff_gs_mean'] = notifs_f - df_all['notifs_by_gs_mean']
    df_all['screen_diff_gs_mean'] = screen_f - df_all['screen_by_gs_mean']

    # 7. Split Train / Test and Final Feature List
    drop_cols = ['id', 'is_train', 'addicted_label', 'gender', 'stress_level', 'academic_work_impact', 'gender_stress', 'stress_impact']
    feat_cols = [c for c in df_all.columns if c not in drop_cols]

    tr_fe = df_all[df_all['is_train'] == 1].copy()
    te_fe = df_all[df_all['is_train'] == 0].copy() if df_test is not None else None

    return tr_fe, te_fe, feat_cols
