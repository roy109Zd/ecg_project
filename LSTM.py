#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LSTM 二分类：正常 = 记录包含 'NORM' 代码（不管是否有其他异常）
异常 = 记录不含 'NORM' 代码
"""

import os
import numpy as np
import pandas as pd
import wfdb
import ast
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
DATA_ROOT = "/root/ecg_tiny/"
FS = 500
LEAD = 1                # 使用 Lead II
SEQ_LEN = 5000          # 10秒 * 500Hz
TASK = 'binary'         # 固定为二分类
USE_GPU = True
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-3
HIDDEN_SIZE = 128
NUM_LAYERS = 2
OUTPUT_DIR = "/root/ecg_tiny/models"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载元数据
df_meta = pd.read_csv(os.path.join(DATA_ROOT, "ptbxl_database.csv"))

# 定义标签函数：只要 scp_codes 中包含 'NORM' 就算正常 (0)，否则异常 (1)
def get_binary_label_norm(scp_str):
    try:
        codes = ast.literal_eval(scp_str)
        if 'NORM' in codes:
            return 0   # 正常
        else:
            return 1   # 异常
    except:
        return 1   # 解析失败视为异常

# 仅保留实际存在的文件
df_meta['file_path'] = df_meta['filename_hr'].apply(lambda x: os.path.join(DATA_ROOT, x))
df_meta['exists'] = df_meta['file_path'].apply(lambda p: os.path.exists(p + '.dat'))
df_valid = df_meta[df_meta['exists']].copy()
print(f"可用记录数: {len(df_valid)}")

# 生成标签
df_valid['label'] = df_valid['scp_codes'].apply(get_binary_label_norm)
num_classes = 2
class_names = ['Normal (has NORM)', 'Abnormal (no NORM)']

print("标签分布:\n", df_valid['label'].value_counts())

# ========== 数据集类 ==========
class ECG_Dataset(Dataset):
    def __init__(self, df, normalize=True):
        self.df = df.reset_index(drop=True)
        self.normalize = normalize
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        try:
            record = wfdb.rdrecord(row['file_path'])
            sig = record.p_signal[:, LEAD]
            if len(sig) < SEQ_LEN:
                sig = np.pad(sig, (0, max(0, SEQ_LEN - len(sig))), mode='constant')[:SEQ_LEN]
            else:
                sig = sig[:SEQ_LEN]
            if self.normalize:
                sig = (sig - np.mean(sig)) / (np.std(sig) + 1e-8)
            label = row['label']
            return torch.tensor(sig, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
        except Exception as e:
            print(f"Error loading {row['filename_hr']}: {e}")
            return torch.zeros(SEQ_LEN, dtype=torch.float32), torch.tensor(0, dtype=torch.long)

# ========== 划分数据集 ==========
# 优先使用 strat_fold 划分，若不存在则随机分层
if 'strat_fold' in df_valid.columns and df_valid['strat_fold'].isin([9,10]).any():
    train_idx = df_valid['strat_fold'].isin(range(1,9))
    val_idx   = df_valid['strat_fold'] == 9
    test_idx  = df_valid['strat_fold'] == 10
else:
    # 随机分层划分 70% 训练，15% 验证，15% 测试
    train_idx, temp_idx = train_test_split(np.arange(len(df_valid)), test_size=0.3, random_state=42, stratify=df_valid['label'])
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=df_valid.iloc[temp_idx]['label'])
    # 转为布尔索引
    train_mask = np.zeros(len(df_valid), dtype=bool)
    val_mask = np.zeros(len(df_valid), dtype=bool)
    test_mask = np.zeros(len(df_valid), dtype=bool)
    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True
    train_idx, val_idx, test_idx = train_mask, val_mask, test_mask

train_df = df_valid[train_idx]
val_df   = df_valid[val_idx]
test_df  = df_valid[test_idx]
print(f"训练集: {len(train_df)} | 验证集: {len(val_df)} | 测试集: {len(test_df)}")

train_dataset = ECG_Dataset(train_df, normalize=True)
val_dataset   = ECG_Dataset(val_df, normalize=True)
test_dataset  = ECG_Dataset(test_df, normalize=True)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
test_loader  = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

# ========== LSTM 模型 ==========
class ECG_LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, num_classes=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=False)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes)
        )
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        last_out = out[:, -1, :]
        logits = self.classifier(last_out)
        return logits

device = torch.device("cuda" if USE_GPU and torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

model = ECG_LSTM(input_size=1, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS, num_classes=num_classes, dropout=0.5)
model.to(device)

# 类别权重（处理不平衡）
train_labels = train_df['label'].values
class_counts = np.bincount(train_labels)
class_weights = 1.0 / class_counts
class_weights = class_weights / class_weights.sum() * num_classes
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

# ========== 训练循环 ==========
best_val_loss = float('inf')
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for signals, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        signals = signals.unsqueeze(-1).to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(signals)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    
    # 验证
    model.eval()
    val_loss = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for signals, labels in val_loader:
            signals = signals.unsqueeze(-1).to(device)
            labels = labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    val_loss /= len(val_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
    scheduler.step(val_loss)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "lstm_binary_norm_as_normal_best.pth"))
        print(f"  保存最佳模型 (val_loss={val_loss:.4f})")

# ========== 测试集评估 ==========
model.load_state_dict(torch.load(os.path.join(OUTPUT_DIR, "lstm_binary_norm_as_normal_best.pth")))
model.eval()
y_true, y_pred, y_proba = [], [], []
with torch.no_grad():
    for signals, labels in test_loader:
        signals = signals.unsqueeze(-1).to(device)
        labels = labels.cpu().numpy()
        outputs = model(signals)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        y_true.extend(labels)
        y_pred.extend(preds)
        y_proba.extend(probs)

acc = accuracy_score(y_true, y_pred)
auc = roc_auc_score(y_true, [p[1] for p in y_proba])
print("\n===== 测试集结果 =====")
print(f"准确率: {acc:.4f}")
print(f"AUC: {auc:.4f}")
print("分类报告:")
print(classification_report(y_true, y_pred, target_names=class_names))

# 保存评估结果
results = pd.DataFrame([{
    'accuracy': acc,
    'auc': auc,
    'task': 'binary_norm_as_normal'
}])
results.to_csv(os.path.join(OUTPUT_DIR, "lstm_binary_norm_as_normal_evaluation.csv"), index=False)
print(f"模型和评估结果已保存至 {OUTPUT_DIR}")