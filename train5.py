#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
五分类（单标签）训练：NORM, MI, STTC, CD, HYP
基于 existing detailed_features.csv
冲突处理：按优先级 MI > STTC > CD > HYP > NORM
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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

# 2. 构造单标签（五分类）
# 定义优先级顺序（从高到低）
priority_order = ['MI', 'STTC', 'CD', 'HYP', 'NORM']

def assign_single_label(row):
    for cls in priority_order:
        if row[cls] == 1:
            return cls
    # 理论上不应该到这里，因为所有样本至少有一个标签？但以防万一，返回 NORM
    return 'NORM'

df['single_label'] = df.apply(assign_single_label, axis=1)

# 统计各类别数量
label_counts = df['single_label'].value_counts()
print("单标签类别分布:")
for lbl, cnt in label_counts.items():
    print(f"  {lbl}: {cnt} ({cnt/len(df)*100:.1f}%)")

# 3. 准备特征
exclude_cols = ['record_name', 'binary_label', 'NORM', 'MI', 'STTC', 'CD', 'HYP', 'single_label']
feature_cols = [c for c in df.columns if c not in exclude_cols]
X = df[feature_cols].copy().fillna(0)
y = df['single_label'].values

# 将类别名称编码为数字 (0-4)
class_names = priority_order   # ['MI','STTC','CD','HYP','NORM']
class_to_id = {cls: i for i, cls in enumerate(class_names)}
y_id = np.array([class_to_id[cls] for cls in y])

print(f"特征维度: {X.shape}")
print(f"类别映射: {class_to_id}")

# 4. 划分训练集和测试集（分层）
X_train, X_test, y_train, y_test = train_test_split(
    X, y_id, test_size=0.2, random_state=42, stratify=y_id
)
print(f"训练集: {len(X_train)} 样本")
print(f"测试集: {len(X_test)} 样本")

# 5. 训练 Random Forest 多分类
print("\n" + "="*60)
print("训练 Random Forest (五分类)...")
rf = RandomForestClassifier(
    n_estimators=200, max_depth=15, random_state=42,
    n_jobs=-1, class_weight='balanced'
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest 测试集准确率: {acc_rf:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_rf, target_names=class_names))

# 6. 训练 XGBoost 多分类
print("\n" + "="*60)
print("训练 XGBoost (五分类)...")
xgb = XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.05,
    objective='multi:softmax', num_class=5, random_state=42,
    eval_metric='mlogloss', use_label_encoder=False
)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"XGBoost 测试集准确率: {acc_xgb:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred_xgb, target_names=class_names))

# 7. 保存最佳模型（按准确率）
if acc_xgb > acc_rf:
    best_model = xgb
    best_name = "XGBoost"
else:
    best_model = rf
    best_name = "RandomForest"
joblib.dump(best_model, os.path.join(OUTPUT_DIR, "5class_best_model.pkl"))
print(f"\n✅ 最佳模型: {best_name} (准确率={max(acc_rf, acc_xgb):.4f})")
print(f"   模型已保存至 {OUTPUT_DIR}/5class_best_model.pkl")

# 8. 保存评估结果到 CSV
results = pd.DataFrame([
    {"model": "RandomForest", "accuracy": acc_rf},
    {"model": "XGBoost", "accuracy": acc_xgb}
])
results.to_csv(os.path.join(OUTPUT_DIR, "5class_evaluation.csv"), index=False)
print("评估结果已保存至 5class_evaluation.csv")