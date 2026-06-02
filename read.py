#!/usr/bin/env python3
import wfdb
import neurokit2 as nk
import numpy as np

dat_path = "/root/ecg_tiny/records500/00000/00002_hr"

# 1. 读取头文件信息
record = wfdb.rdrecord(dat_path)
print("=== 头文件信息 ===")
print(f"采样率: {record.fs} Hz")
print(f"信号长度: {record.sig_len} 个采样点")
print(f"时长: {record.sig_len / record.fs:.2f} 秒")
print(f"导联数: {record.n_sig}")
print(f"导联名称: {record.sig_name}")
print(f"增益（每单位幅度对应的微伏）: {record.adc_gain} (μV/bit)")
print(f"基线: {record.baseline}")
print(f"初始值: {record.init_value}")

# 2. 读取信号（使用第一导联进行R波检测）
signal = record.p_signal[:, 1]  # 第二导联 (Lead II), 索引为1
print(f"\n信号统计: min={np.min(signal):.3f}, max={np.max(signal):.3f}, mean={np.mean(signal):.3f}")

# 3. R波检测
try:
    _, rpeaks_dict = nk.ecg_peaks(signal, sampling_rate=record.fs, method='neurokit')
    r_peaks = rpeaks_dict['ECG_R_Peaks']
    r_peaks = r_peaks[r_peaks > 0]
    num_r_peaks = len(r_peaks)
    print(f"R波数量: {num_r_peaks}")
    if num_r_peaks > 1:
        rr_intervals = np.diff(r_peaks) / record.fs
        avg_hr = 60 / np.mean(rr_intervals)
        print(f"平均心率: {avg_hr:.1f} bpm")
except Exception as e:
    print(f"R波检测失败: {e}")

print("\n=== .hea 文件原始内容（前几行）===")
with open(dat_path + ".hea", "r") as f:
    for i, line in enumerate(f):
        print(line.rstrip())
        if i >= 10:  # 只显示前10行
            print("...")
            break