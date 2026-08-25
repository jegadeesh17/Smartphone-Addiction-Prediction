"""
Advanced Multi-Model Training Pipeline for Smartphone Addiction Prediction (Kaggle S6E8).
Executes 5-Fold Stratified Cross-Validation across 4 model families:
1. Deep LightGBM (num_leaves=191, max_depth=10)
2. Regularized Hist XGBoost (max_depth=8)
3. Symmetric CatBoost (depth=7, l2_leaf_reg=4.0)
4. PyTorch Tabular ResNet (Entity Embeddings + Residual Blocks)

Saves Out-of-Fold (OOF) predictions, test predictions, and model artifacts for ensembling.
"""

import os
import sys
import gc
import joblib
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score
import torch
import warnings
warnings.filterwarnings('ignore')

# Set standard output encoding for cross-platform compatibility
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from features import create_features
from nn_model import train_tabular_nn


def train_pipeline(n_splits: int = 5, random_state: int = 42, include_nn: bool = True):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)

    print("=" * 65)
    print("Starting 5-Fold Stratified CV Training Pipeline")
    print("=" * 65)

    print("\n[Step 1/5] Loading raw datasets...")
    train = pd.read_csv(os.path.join(data_dir, 'train.csv'))
    test = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f"Train samples: {len(train):,} | Test samples: {len(test):,}")

    print("\n[Step 2/5] Computing advanced domain & cohort features...")
    tr_fe, te_fe, feat_cols = create_features(train, test)
    print(f"Engineered {len(feat_cols)} total features for modeling.")

    # Identify categorical vs continuous columns for neural network
    cat_cols = ['gender_code', 'stress_code', 'impact_code', 'gender_stress_code', 'stress_impact_code']
    cat_cols = [c for c in cat_cols if c in feat_cols]
    num_cols = [c for c in feat_cols if c not in cat_cols]
    cat_cardinalities = [int(tr_fe[c].max() + 2) for c in cat_cols]

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    # Initialize OOF prediction arrays
    lgb_oof = np.zeros(len(train))
    xgb_oof = np.zeros(len(train))
    cat_oof = np.zeros(len(train))
    nn_oof = np.zeros(len(train))

    # Initialize Test prediction arrays
    lgb_test = np.zeros(len(test))
    xgb_test = np.zeros(len(test))
    cat_test = np.zeros(len(test))
    nn_test = np.zeros(len(test))

    print(f"\n[Step 3/5] Executing {n_splits}-Fold Stratified Cross-Validation...\n")

    for fold, (tr_idx, va_idx) in enumerate(skf.split(tr_fe, tr_fe['addicted_label'])):
        print(f"{'='*25} FOLD {fold + 1} / {n_splits} {'='*25}")
        X_tr, y_tr = tr_fe.iloc[tr_idx][feat_cols], tr_fe.iloc[tr_idx]['addicted_label'].values
        X_va, y_va = tr_fe.iloc[va_idx][feat_cols], tr_fe.iloc[va_idx]['addicted_label'].values
        X_te = te_fe[feat_cols]

        # ----------------------------------------------------
        # 1. Deep LightGBM
        # ----------------------------------------------------
        print("  [1/4] Training Deep LightGBM...", end=" ", flush=True)
        m_lgb = lgb.LGBMClassifier(
            n_estimators=2500,
            learning_rate=0.02,
            num_leaves=191,
            max_depth=10,
            min_child_samples=40,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=2.0,
            random_state=random_state + fold,
            n_jobs=-1,
            verbose=-1
        )
        m_lgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], callbacks=[lgb.early_stopping(50, verbose=False)])
        p_lgb_va = m_lgb.predict_proba(X_va)[:, 1]
        lgb_oof[va_idx] = p_lgb_va
        lgb_test += m_lgb.predict_proba(X_te)[:, 1] / n_splits
        joblib.dump(m_lgb, os.path.join(models_dir, f'lgb_fold_{fold+1}.joblib'))
        print(f"Done (AUC: {roc_auc_score(y_va, p_lgb_va):.5f})")

        # ----------------------------------------------------
        # 2. Regularized Hist XGBoost
        # ----------------------------------------------------
        print("  [2/4] Training Regularized Hist XGBoost...", end=" ", flush=True)
        m_xgb = xgb.XGBClassifier(
            n_estimators=1500,
            learning_rate=0.03,
            max_depth=8,
            min_child_weight=30,
            subsample=0.8,
            colsample_bytree=0.75,
            reg_alpha=0.2,
            reg_lambda=2.0,
            tree_method='hist',
            random_state=random_state + fold,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='auc'
        )
        m_xgb.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        p_xgb_va = m_xgb.predict_proba(X_va)[:, 1]
        xgb_oof[va_idx] = p_xgb_va
        xgb_test += m_xgb.predict_proba(X_te)[:, 1] / n_splits
        joblib.dump(m_xgb, os.path.join(models_dir, f'xgb_fold_{fold+1}.joblib'))
        print(f"Done (AUC: {roc_auc_score(y_va, p_xgb_va):.5f})")

        # ----------------------------------------------------
        # 3. Symmetric CatBoost
        # ----------------------------------------------------
        print("  [3/4] Training Symmetric CatBoost...", end=" ", flush=True)
        m_cat = CatBoostClassifier(
            iterations=1200,
            learning_rate=0.04,
            depth=7,
            l2_leaf_reg=4.0,
            random_seed=random_state + fold,
            thread_count=-1,
            verbose=0
        )
        m_cat.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50)
        p_cat_va = m_cat.predict_proba(X_va)[:, 1]
        cat_oof[va_idx] = p_cat_va
        cat_test += m_cat.predict_proba(X_te)[:, 1] / n_splits
        joblib.dump(m_cat, os.path.join(models_dir, f'cat_fold_{fold+1}.joblib'))
        print(f"Done (AUC: {roc_auc_score(y_va, p_cat_va):.5f})")

        # ----------------------------------------------------
        # 4. PyTorch Tabular ResNet
        # ----------------------------------------------------
        if include_nn:
            print("  [4/4] Training PyTorch Tabular ResNet...", end=" ", flush=True)
            p_nn_va, p_nn_te, m_nn = train_tabular_nn(
                X_tr_num=X_tr[num_cols].values,
                X_tr_cat=X_tr[cat_cols].values,
                y_tr=y_tr,
                X_va_num=X_va[num_cols].values,
                X_va_cat=X_va[cat_cols].values,
                y_va=y_va,
                X_te_num=X_te[num_cols].values,
                X_te_cat=X_te[cat_cols].values,
                cat_cardinalities=cat_cardinalities,
                epochs=8,
                batch_size=2048,
                lr=1e-3,
                device='cpu'
            )
            nn_oof[va_idx] = p_nn_va
            nn_test += p_nn_te / n_splits
            torch.save(m_nn.state_dict(), os.path.join(models_dir, f'nn_fold_{fold+1}.pt'))
            print(f"Done (AUC: {roc_auc_score(y_va, p_nn_va):.5f})")

        # Fold Blend Summary
        fold_blend = 0.5 * p_lgb_va + 0.3 * p_xgb_va + 0.2 * p_cat_va
        print(f"  --> Fold {fold+1} Blend ROC AUC: {roc_auc_score(y_va, fold_blend):.6f} | Acc: {accuracy_score(y_va, (fold_blend >= 0.5).astype(int)):.4f}\n")
        gc.collect()

    # ----------------------------------------------------
    # Overall OOF Performance Report
    # ----------------------------------------------------
    y_true = train['addicted_label'].values
    auc_lgb = roc_auc_score(y_true, lgb_oof)
    auc_xgb = roc_auc_score(y_true, xgb_oof)
    auc_cat = roc_auc_score(y_true, cat_oof)
    auc_nn = roc_auc_score(y_true, nn_oof) if include_nn else 0.0

    print("=" * 65)
    print("5-Fold Stratified Out-of-Fold (OOF) Benchmark Summary")
    print("=" * 65)
    print(f"  * LightGBM OOF ROC AUC:       {auc_lgb:.6f} | Acc: {accuracy_score(y_true, (lgb_oof >= 0.5).astype(int)):.4f}")
    print(f"  * XGBoost  OOF ROC AUC:       {auc_xgb:.6f} | Acc: {accuracy_score(y_true, (xgb_oof >= 0.5).astype(int)):.4f}")
    print(f"  * CatBoost OOF ROC AUC:       {auc_cat:.6f} | Acc: {accuracy_score(y_true, (cat_oof >= 0.5).astype(int)):.4f}")
    if include_nn:
        print(f"  * Tabular ResNet OOF ROC AUC: {auc_nn:.6f} | Acc: {accuracy_score(y_true, (nn_oof >= 0.5).astype(int)):.4f}")
    print("=" * 65)

    # Save OOF and Test predictions
    oof_save_path = os.path.join(models_dir, 'oof_predictions.npz')
    np.savez_compressed(
        oof_save_path,
        lgb_oof=lgb_oof,
        xgb_oof=xgb_oof,
        cat_oof=cat_oof,
        nn_oof=nn_oof,
        lgb_test=lgb_test,
        xgb_test=xgb_test,
        cat_test=cat_test,
        nn_test=nn_test,
        y_true=y_true,
        has_nn=include_nn
    )
    print(f"\n[Step 4/5] Checkpoint saved successfully to: {oof_save_path}")


if __name__ == '__main__':
    train_pipeline()
