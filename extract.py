#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PTB-XL 详细临床特征提取 (500Hz, Lead II)
基于 neurokit2 实际返回结构 (tuple)
"""

import os
import numpy as np
import pandas as pd
import wfdb
import neurokit2 as nk
from scipy import signal, stats, integrate
from tqdm import tqdm
import ast
import warnings
warnings.filterwarnings('ignore')

# ========== 配置 ==========
DATA_ROOT = "/root/ecg_tiny/"
FS = 500
LEAD = 1          # 第二导联 (Lead II)
OUTPUT_CSV = "detailed_features.csv"

# 加载元数据
df_meta = pd.read_csv(os.path.join(DATA_ROOT, "ptbxl_database.csv"))
scp_df = pd.read_csv(os.path.join(DATA_ROOT, "scp_statements.csv"), index_col=0)

# 诊断代码集合
diag_codes = set(scp_df[scp_df['diagnostic'] == 1].index)

# 诊断超类映射
code_to_class = {}
for code, row in scp_df.iterrows():
    if pd.notna(row.get('diagnostic_class')):
        code_to_class[code] = row['diagnostic_class']

def get_binary_label(scp_str):
    try:
        codes = ast.literal_eval(scp_str)
        has_diag = any(c in diag_codes for c in codes.keys())
        if not has_diag:
            return 0
        if len(codes) == 1 and 'NORM' in codes:
            return 0
        return 1
    except:
        return 0

def get_multilabel(scp_str):
    try:
        codes = ast.literal_eval(scp_str)
        classes = set()
        for c in codes:
            if c in code_to_class:
                classes.add(code_to_class[c])
        if len(classes) == 0:
            classes.add('NORM')
        return [1 if cls in classes else 0 for cls in ['NORM','MI','STTC','CD','HYP']]
    except:
        return [1,0,0,0,0]

# ========== 特征提取函数（适配正确的delineate结构） ==========
def extract_ecg_features(signal_raw, fs):
    features = {}
    # 带通滤波 0.5-40 Hz
    nyq = 0.5 * fs
    low = 0.5 / nyq
    high = 40 / nyq
    b, a = signal.butter(4, [low, high], btype='band')
    ecg_filtered = signal.filtfilt(b, a, signal_raw)

    try:
        # R峰检测
        _, rpeaks_dict = nk.ecg_peaks(ecg_filtered, sampling_rate=fs, method='neurokit')
        r_peaks = rpeaks_dict['ECG_R_Peaks']
        r_peaks = r_peaks[r_peaks > 0]
        if len(r_peaks) < 2:
            raise ValueError("R峰不足2个")

        # 波形分割（返回 (df, dict)）
        delineate_tuple = nk.ecg_delineate(ecg_filtered, r_peaks, sampling_rate=fs, method='dwt')
        # 关键点字典在元组的第二个元素
        delineate_dict = delineate_tuple[1]

        # 从字典中提取各个关键点列表（每个列表长度等于心拍数）
        # 注意：这些列表可能含有 None 值，需要转换为整数或0
        def get_points(key):
            pts = delineate_dict.get(key, [])
            if not pts:
                return np.zeros(len(r_peaks), dtype=int)
            # 确保列表长度匹配心拍数，不足补0
            if len(pts) < len(r_peaks):
                pts = list(pts) + [0] * (len(r_peaks) - len(pts))
            # 将 None 或 NaN 转换为 0，并转换为 int
            clean = []
            for p in pts:
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    clean.append(0)
                else:
                    try:
                        clean.append(int(p))
                    except (ValueError, TypeError):
                        clean.append(0)
            return np.array(clean, dtype=int)       
        q_onsets = get_points('ECG_Q_Peaks')
        s_offsets = get_points('ECG_S_Peaks')
        p_peaks = get_points('ECG_P_Peaks')
        t_peaks = get_points('ECG_T_Peaks')

        # 心率统计
        rr_intervals = np.diff(r_peaks) / fs
        heart_rates = 60 / rr_intervals
        features['heart_rate_mean'] = np.mean(heart_rates)
        features['heart_rate_std'] = np.std(heart_rates)
        features['heart_rate_min'] = np.min(heart_rates)
        features['heart_rate_max'] = np.max(heart_rates)

        # HRV时域
        features['HRV_SDNN_ms'] = np.std(rr_intervals) * 1000
        if len(rr_intervals) > 1:
            features['HRV_RMSSD_ms'] = np.sqrt(np.mean(np.diff(rr_intervals)**2)) * 1000
            features['HRV_pNN50_percent'] = (np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / (len(rr_intervals)-1)) * 100
        else:
            features['HRV_RMSSD_ms'] = 0
            features['HRV_pNN50_percent'] = 0

        # 频域HRV（简化，跳过）
        features['HRV_VLF_power'] = 0
        features['HRV_LF_power'] = 0
        features['HRV_HF_power'] = 0
        features['HRV_LF_HF_ratio'] = 0

        # 形态学特征（取前10个有效心拍）
        valid_mask = (q_onsets > 0) & (s_offsets > 0) & (p_peaks > 0) & (t_peaks > 0) & (r_peaks > 0)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            raise ValueError("无有效心拍")
        n_beats = min(10, len(valid_idx))
        idx_use = valid_idx[:n_beats]

        pr_intervals, qrs_durations, qt_intervals, st_shifts = [], [], [], []
        p_amp, t_amp, r_amp, s_depth = [], [], [], []

        for beat in idx_use:
            # PR间期（使用P峰到QRS起点，近似）
            if q_onsets[beat] > 0 and p_peaks[beat] > 0:
                pr = (q_onsets[beat] - p_peaks[beat]) / fs * 1000
                pr_intervals.append(pr)
            # QRS宽度
            if s_offsets[beat] > 0 and q_onsets[beat] > 0:
                qrs = (s_offsets[beat] - q_onsets[beat]) / fs * 1000
                qrs_durations.append(qrs)
            # QT间期（QRS起点到T波峰值，近似）
            if t_peaks[beat] > 0 and q_onsets[beat] > 0:
                qt = (t_peaks[beat] - q_onsets[beat]) / fs * 1000
                qt_intervals.append(qt)
            # ST段偏移（J点 - PR段基线）
            if s_offsets[beat] > 0 and q_onsets[beat] > 0:
                j_point = min(s_offsets[beat] + int(0.06*fs), len(ecg_filtered)-1)
                pr_start = max(q_onsets[beat] - int(0.04*fs), 0)
                baseline = np.mean(ecg_filtered[pr_start:q_onsets[beat]])
                st_shift = (ecg_filtered[j_point] - baseline) * 1000   # μV
                st_shifts.append(st_shift)
            # 幅度特征
            if r_peaks[beat] < len(ecg_filtered):
                r_amp.append(ecg_filtered[r_peaks[beat]])
            if p_peaks[beat] > 0:
                p_amp.append(ecg_filtered[p_peaks[beat]])
            if t_peaks[beat] > 0:
                t_amp.append(ecg_filtered[t_peaks[beat]])
            # S波深度（S峰附近最小值）
            if s_offsets[beat] > 0:
                search_start = max(0, s_offsets[beat] - int(0.02*fs))
                search_end = min(len(ecg_filtered), s_offsets[beat] + int(0.02*fs))
                s_val = np.min(ecg_filtered[search_start:search_end])
                s_depth.append(s_val)

        features['PR_interval_ms'] = np.mean(pr_intervals) if pr_intervals else 0
        features['QRS_duration_ms'] = np.mean(qrs_durations) if qrs_durations else 0
        features['QT_interval_ms'] = np.mean(qt_intervals) if qt_intervals else 0
        avg_rr = np.mean(rr_intervals) * 1000
        if features['QT_interval_ms'] > 0 and avg_rr > 0:
            features['QTc_Bazett_ms'] = features['QT_interval_ms'] / np.sqrt(avg_rr / 1000)
        else:
            features['QTc_Bazett_ms'] = 0
        features['ST_segment_deviation_uV'] = np.mean(st_shifts) if st_shifts else 0
        features['P_amplitude_mV'] = np.mean(p_amp) if p_amp else 0
        features['T_amplitude_mV'] = np.mean(t_amp) if t_amp else 0
        features['R_amplitude_mV'] = np.mean(r_amp) if r_amp else 0
        features['S_depth_mV'] = np.mean(s_depth) if s_depth else 0
        features['R_S_ratio'] = features['R_amplitude_mV'] / (abs(features['S_depth_mV']) + 1e-8)

        # 病理性Q波
        q_wave = False
        for beat in idx_use:
            if q_onsets[beat] > 0 and r_peaks[beat] > 0:
                q_region = ecg_filtered[q_onsets[beat]:r_peaks[beat]]
                if len(q_region) > 0:
                    q_val = np.min(q_region)
                    q_width = (np.argmin(q_region) / fs) * 1000
                    if q_val < -0.1 and q_width > 30:
                        q_wave = True
                        break
        features['pathological_Q_wave'] = 1 if q_wave else 0

        # T波对称性
        t_sym = []
        for beat in idx_use:
            if t_peaks[beat] > 0:
                start = max(0, t_peaks[beat] - int(0.04*fs))
                end = min(len(ecg_filtered), t_peaks[beat] + int(0.04*fs))
                t_seg = ecg_filtered[start:end]
                if len(t_seg) > 10:
                    mid = len(t_seg)//2
                    up = t_seg[mid] - t_seg[0] if mid>0 else 0
                    down = t_seg[-1] - t_seg[mid] if len(t_seg)-mid>0 else 0
                    sym = up / (down + 1e-8)
                    t_sym.append(sym)
        features['T_wave_symmetry'] = np.mean(t_sym) if t_sym else 0

        # QRS面积
        qrs_areas = []
        for beat in idx_use:
            if q_onsets[beat] > 0 and s_offsets[beat] > 0:
                seg = ecg_filtered[q_onsets[beat]:s_offsets[beat]]
                area = integrate.simpson(np.abs(seg), dx=1/fs)
                qrs_areas.append(area)
        features['QRS_area_mVs'] = np.mean(qrs_areas) if qrs_areas else 0

        # 信号质量
        noise_band = signal.butter(4, [30,45], btype='band', fs=fs, output='sos')
        noise = signal.sosfilt(noise_band, ecg_filtered)
        signal_power = np.var(ecg_filtered)
        noise_power = np.var(noise)
        features['SNR_dB'] = 10 * np.log10(signal_power/(noise_power+1e-8))
        low_sos = signal.butter(2, 0.5, btype='low', fs=fs, output='sos')
        baseline = signal.sosfilt(low_sos, ecg_filtered)
        features['baseline_drift_ratio'] = np.var(baseline) / (signal_power+1e-8)

    except Exception as e:
        print(f"特征提取错误: {e}")
        default_keys = ['heart_rate_mean','heart_rate_std','heart_rate_min','heart_rate_max',
                        'HRV_SDNN_ms','HRV_RMSSD_ms','HRV_pNN50_percent','HRV_VLF_power',
                        'HRV_LF_power','HRV_HF_power','HRV_LF_HF_ratio','PR_interval_ms',
                        'QRS_duration_ms','QT_interval_ms','QTc_Bazett_ms','ST_segment_deviation_uV',
                        'P_amplitude_mV','T_amplitude_mV','R_amplitude_mV','S_depth_mV',
                        'R_S_ratio','pathological_Q_wave','T_wave_symmetry','QRS_area_mVs',
                        'SNR_dB','baseline_drift_ratio']
        features = {k: 0 for k in default_keys}
    return features

# ========== 主程序：动态处理所有存在的文件 ==========
print("正在读取 PTB-XL 元数据...")
df_meta['file_path'] = df_meta['filename_hr'].apply(lambda x: os.path.join(DATA_ROOT, x))
df_meta['file_exists'] = df_meta['file_path'].apply(lambda p: os.path.exists(p + '.dat'))
df_valid = df_meta[df_meta['file_exists']].copy()
print(f"总记录数: {len(df_meta)}，实际存在的记录数: {len(df_valid)}")

if len(df_valid) == 0:
    raise SystemExit("错误：没有找到任何 .dat 文件，请检查数据路径和文件命名。")

# 预计算标签
df_valid['binary_label'] = df_valid['scp_codes'].apply(get_binary_label)
multilabels = df_valid['scp_codes'].apply(get_multilabel)
df_valid[['NORM','MI','STTC','CD','HYP']] = pd.DataFrame(multilabels.tolist(), index=df_valid.index)

# 批量提取特征
all_features = []
for idx, row in tqdm(df_valid.iterrows(), total=len(df_valid), desc="提取特征"):
    try:
        base_path = row['file_path']   # 已包含 _hr 后缀，不含扩展名
        record = wfdb.rdrecord(base_path)
        signal_lead = record.p_signal[:, LEAD]
        features = extract_ecg_features(signal_lead, FS)
        features['record_name'] = row['filename_hr']
        features['binary_label'] = row['binary_label']
        for sc in ['NORM','MI','STTC','CD','HYP']:
            features[sc] = row[sc]
        all_features.append(features)
    except Exception as e:
        print(f"\n处理 {row['filename_hr']} 时出错: {e}")
        continue

if len(all_features) == 0:
    raise SystemExit("没有成功提取任何特征，请检查信号读取部分。")

df_features = pd.DataFrame(all_features)
first_cols = ['record_name','binary_label','NORM','MI','STTC','CD','HYP']
other_cols = [c for c in df_features.columns if c not in first_cols]
df_features = df_features[first_cols + other_cols]
df_features.to_csv(OUTPUT_CSV, index=False)
print(f"\n特征提取完成，共 {len(df_features)} 条记录，保存至 {OUTPUT_CSV}")