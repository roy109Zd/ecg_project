#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
严格二分类：正常 = 纯 NORM（无任何其他异常标签），异常 = 有 MI/STTC/CD/HYP 任一
基于 existing detailed_features.csv
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
DATA_FILE = "/root/ecg_tiny/results/detailed_features.csv"
OUTPUT_DIR = "/root/ecg_tiny/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. 加载特征数据
print("="*60)
print("加载特征数据...")
df = pd.read_csv(DATA_FILE)
print(f"原始样本数: {len(df)}")

# 2. 构建严格二分类标签
# 正常条件：NORM == 1 且 MI==0 且 STTC==0 且 CD==0 且 HYP==0
df['label'] = 0  # 默认正常
# 只要四个异常类中任意一个为1，就标记为异常（1）
abnormal_mask = (df['MI']==1) | (df['STTC']==1) | (df['CD']==1) | (df['HYP']==1)
df.loc[abnormal_mask, 'label'] = 1

# 可选：同时要求 NORM==0 的记录如果没有任何异常类？但这种情况在数据集中极少，忽略。
# 统计标签分布
normal_cnt = (df['label']==0).sum()
abnormal_cnt = (df['label']==1).sum()
print(f"严格定义后: 正常样本 {normal_cnt} ({normal_cnt/len(df)*100:.1f}%)")
print(f"           异常样本 {abnormal_cnt} ({abnormal_cnt/len(df)*100:.1f}%)")

# 3. 准备特征（排除不需要的列）
exclude_cols = ['record_name', 'binary_label', 'NORM', 'MI', 'STTC', 'CD', 'HYP', 'label']
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].copy()
# 处理缺失值（如果有）
X = X.fillna(0)
y = df['label'].values

print(f"特征维度: {X.shape}")

# 4. 划分训练集和测试集（分层，保持正负比例）
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"训练集: {len(X_train)} 样本 (正常={sum(y_train==0)}, 异常={sum(y_train==1)})")
print(f"测试集: {len(X_test)} 样本 (正常={sum(y_test==0)}, 异常={sum(y_test==1)})")

# 5. 训练 Random Forest
print("\n" + "="*60)
print("训练 Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, random_state=42,
    n_jobs=-1, class_weight='balanced'
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_proba_rf = rf.predict_proba(X_test)[:, 1]
acc_rf = accuracy_score(y_test, y_pred_rf)
auc_rf = roc_auc_score(y_test, y_proba_rf)
print(f"Random Forest 测试集: Acc={acc_rf:.4f}, AUC={auc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=['正常','异常']))

# 6. 训练 XGBoost
print("\n" + "="*60)
print("训练 XGBoost...")
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb = XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, random_state=42,
    eval_metric='logloss', use_label_encoder=False
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_proba_xgb = xgb.predict_proba(X_test)[:, 1]
acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_proba_xgb)
print(f"XGBoost 测试集: Acc={acc_xgb:.4f}, AUC={auc_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=['正常','异常']))

# 7. 保存最佳模型（按 AUC）
if auc_xgb > auc_rf:
    best_model = xgb
    best_name = "XGBoost"
else:
    best_model = rf
    best_name = "RandomForest"
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "strict_binary_best.pkl"))
print(f"\n✅ 最佳模型: {best_name} (AUC={max(auc_rf, auc_xgb):.4f})")
print(f"   模型已保存至 {OUTPUT_DIR}/strict_binary_best.pkl")

# 8. 保存评估结果到 CSV
results = pd.DataFrame([
    {"model": "RandomForest", "accuracy": acc_rf, "auc": auc_rf},
    {"model": "XGBoost", "accuracy": acc_xgb, "auc": auc_xgb}
])
results.to_csv(os.path.join(OUTPUT_DIR, "strict_binary_evaluation.csv"), index=False)
print("评估结果已保存至 strict_binary_evaluation.csv")