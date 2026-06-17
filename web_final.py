#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化版：轻量滤波（lfilter）、降低推送频率、延迟特征提取
保证硬件采集采样率不下降
新增 /api/compute_features 路由，实现按需实时特征计算
"""

import time
import numpy as np
from scipy import signal
import board
import busio
from adafruit_bus_device.i2c_device import I2CDevice
import multiprocessing as mp
import ctypes
from flask import Flask, render_template, Response, jsonify, request
import json
from openai import OpenAI
import neurokit2 as nk

# ----------------------------- 硬件采集（保持不变） -----------------------------
SAMPLE_RATE = 860
WINDOW_SECOND = 4.0
BUFFER_SIZE = int(SAMPLE_RATE * (max(WINDOW_SECOND, 10) + 3.0) * 2)

def reader_process(shared_array, current_index, shared_fs):
    i2c = busio.I2C(board.SCL, board.SDA)
    device = I2CDevice(i2c, 0x48)
    with device:
        device.write(bytearray([0x01, 0x42, 0xE0]))
        device.write(bytearray([0x00]))
    while not i2c.try_lock():
        pass
    result_buf = bytearray(2)
    idx = 0
    count = 0
    t_start = time.time()
    print("后台高速采集引擎已启动...")
    while True:
        try:
            i2c.readfrom_into(0x48, result_buf)
            val = (result_buf[0] << 8) | result_buf[1]
            if val > 32767:
                val -= 65536
            shared_array[idx % BUFFER_SIZE] = val
            idx += 1
            current_index.value = idx
            count += 1
            now = time.time()
            if now - t_start >= 1.0:
                shared_fs.value = count / (now - t_start)
                print(f"底层真实采样率: {shared_fs.value:.1f} Hz")
                count = 0
                t_start = now
            time.sleep(0.0005)
        except Exception:
            pass

# ----------------------------- 轻量化滤波（使用 lfilter 代替 filtfilt） -----------------------------
# 初始化滤波器状态，用于连续滤波
filter_state = {}   # 存储每个滤波器的状态

def baseline_filter_lfilter(data, fs, zi=None):
    b, a = signal.butter(1, 0.5, btype='high', fs=fs)
    if zi is None:
        zi = signal.lfilter_zi(b, a) * data[0]
    y, zo = signal.lfilter(b, a, data, zi=zi)
    return y, zo

def notch_filter_lfilter(data, fs, zi=None):
    if fs <= 105:
        return data, zi
    b, a = signal.iirnotch(50, 3, fs)
    if zi is None:
        zi = signal.lfilter_zi(b, a) * data[0]
    y, zo = signal.lfilter(b, a, data, zi=zi)
    return y, zo

def low_pass_filter_lfilter(data, fs, zi=None):
    cutoff = min(50.0, fs * 0.45)
    b, a = signal.butter(1, cutoff, btype='low', fs=fs)   # 降为1阶
    if zi is None:
        zi = signal.lfilter_zi(b, a) * data[0]
    y, zo = signal.lfilter(b, a, data, zi=zi)
    return y, zo

def smoothing_filter(data):
    window_length = min(17, len(data) if len(data)%2==1 else len(data)-1)
    if window_length < 3:
        return data
    return signal.savgol_filter(data, window_length=window_length, polyorder=3)

def bpm_calculate(data, fs):
    """与原版相同，但数据量较小"""
    if len(data) < fs:
        return 0
    try:
        lowcut, highcut = 5.0, 15.0
        b, a = signal.butter(1, [lowcut, highcut], btype='bandpass', fs=fs)
        bandpassed = signal.filtfilt(b, a, data)   # 心率检测仍用filtfilt保证准确
        derivative = np.gradient(bandpassed)
        squared = derivative ** 2
        window_size = int(0.150 * fs)
        window = np.ones(window_size) / window_size
        integrated = np.convolve(squared, window, mode='same')
        min_distance = int(0.3 * fs)
        baseline = np.mean(integrated)
        candidate_peaks, _ = signal.find_peaks(integrated, height=baseline, distance=min_distance)
        if len(candidate_peaks) == 0:
            return 0
        SPKI = np.max(integrated[candidate_peaks]) * 0.5
        NPKI = baseline * 0.5
        valid_peaks = []
        for peak in candidate_peaks:
            peak_value = integrated[peak]
            if (NPKI + 0.25 * (SPKI - NPKI)) < peak_value < (SPKI * 3.0):
                valid_peaks.append(peak)
                SPKI = 0.125 * peak_value + 0.875 * SPKI
            else:
                NPKI = 0.125 * peak_value + 0.875 * NPKI
        peaks = np.array(valid_peaks)
        if len(peaks) >= 2:
            return int(60 / (np.mean(np.diff(peaks)) / fs))
    except:
        return 0
    return 0

# ----------------------------- Web 应用 -----------------------------
app = Flask(__name__)

shared_array = None
current_index = None
shared_fs = None

def init_shared_memory():
    global shared_array, current_index, shared_fs
    if shared_array is None:
        shared_array = mp.Array(ctypes.c_double, BUFFER_SIZE)
        current_index = mp.Value('i', 0)
        shared_fs = mp.Value('d', 10.0)
        p = mp.Process(target=reader_process, args=(shared_array, current_index, shared_fs), daemon=True)
        p.start()
        time.sleep(2)

heart_rate_history = []
bpm_queue = []
feature_cache = {}

# ---------- 特征提取核心函数（可复用） ----------
def extract_features_from_cleaned_signal(clean, fs):
    """从已经滤波和翻转的心电数据中提取特征（基于 neurokit2）"""
    try:
        _, rpeaks_dict = nk.ecg_peaks(clean, sampling_rate=fs, method='neurokit')
        r_peaks = rpeaks_dict['ECG_R_Peaks']
        r_peaks = r_peaks[r_peaks > 0]
        if len(r_peaks) < 2:
            return {}
        delineate = nk.ecg_delineate(clean, r_peaks, sampling_rate=fs, method='dwt')
        delineate_dict = delineate[1]

        def get_points(key):
            pts = delineate_dict.get(key, [])
            if not pts:
                return np.zeros(len(r_peaks), dtype=int)
            if len(pts) < len(r_peaks):
                pts = list(pts) + [0] * (len(r_peaks) - len(pts))
            clean_pts = []
            for p in pts:
                if p is None or (isinstance(p, float) and np.isnan(p)):
                    clean_pts.append(0)
                else:
                    clean_pts.append(int(p))
            return np.array(clean_pts, dtype=int)

        q_onsets = get_points('ECG_Q_Peaks')
        s_offsets = get_points('ECG_S_Peaks')
        p_peaks = get_points('ECG_P_Peaks')
        t_peaks = get_points('ECG_T_Peaks')

        rr_intervals = np.diff(r_peaks) / fs * 1000
        heart_rates = 60000 / rr_intervals
        avg_hr = np.mean(heart_rates)
        current_hr = heart_rates[-1] if len(heart_rates) > 0 else avg_hr
        sdnn = np.std(rr_intervals)
        rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2)) if len(rr_intervals) > 1 else 0

        valid = (q_onsets > 0) & (s_offsets > 0) & (p_peaks > 0) & (t_peaks > 0)
        valid_idx = np.where(valid)[0]
        if len(valid_idx) == 0:
            return {}
        n_beats = min(5, len(valid_idx))
        idx_use = valid_idx[:n_beats]

        pr_list, qrs_list, qt_list, st_list = [], [], [], []
        for beat in idx_use:
            if q_onsets[beat] > 0 and p_peaks[beat] > 0:
                pr = (q_onsets[beat] - p_peaks[beat]) / fs * 1000
                pr_list.append(pr)
            if s_offsets[beat] > 0 and q_onsets[beat] > 0:
                qrs = (s_offsets[beat] - q_onsets[beat]) / fs * 1000
                qrs_list.append(qrs)
            if t_peaks[beat] > 0 and q_onsets[beat] > 0:
                qt = (t_peaks[beat] - q_onsets[beat]) / fs * 1000
                qt_list.append(qt)
            if s_offsets[beat] > 0:
                j_point = min(s_offsets[beat] + int(0.06*fs), len(clean)-1)
                pr_start = max(q_onsets[beat] - int(0.04*fs), 0)
                baseline = np.mean(clean[pr_start:q_onsets[beat]])
                st_shift = (clean[j_point] - baseline) * 1000
                st_list.append(st_shift)

        avg_pr = np.mean(pr_list) if pr_list else 0
        avg_qrs = np.mean(qrs_list) if qrs_list else 80
        avg_qt = np.mean(qt_list) if qt_list else 0
        avg_st = np.mean(st_list) if st_list else 0
        avg_rr = np.mean(rr_intervals) if len(rr_intervals) > 0 else 800
        qtc = avg_qt / np.sqrt(avg_rr / 1000) if avg_qt > 0 else 0

        noise = np.std(np.diff(clean)) * 10
        signal_power = np.var(clean)
        snr = 10 * np.log10(signal_power / (noise + 1e-8))
        quality = min(100, max(0, int(snr + 20)))

        features = {
            'PR间期 (ms)': round(avg_pr, 1),
            'QRS宽度 (ms)': round(avg_qrs, 1),
            'QT间期 (ms)': round(avg_qt, 1),
            'QTc (Bazett) (ms)': round(qtc, 1),
            'ST段偏移 (μV)': round(avg_st, 1),
            'SDNN (ms)': round(sdnn, 1),
            'RMSSD (ms)': round(rmssd, 1),
            '平均心率 (bpm)': round(avg_hr, 1),
            '当前心率 (bpm)': round(current_hr, 1),
            '信号质量': f"{quality}%"
        }
        return features
    except Exception as e:
        print(f"特征提取错误: {e}")
        return {}

def extract_features_downsampled(duration=20):
    """降采样到250Hz后提取特征，大幅降低计算量"""
    idx = current_index.value
    fs_real = max(shared_fs.value, 10.0)
    target_fs = 250
    need_samples = int(fs_real * duration)
    if idx < need_samples:
        return {}
    raw = np.empty(need_samples)
    for i in range(need_samples):
        raw[i] = shared_array[(idx - need_samples + i) % BUFFER_SIZE]
    # 降采样到250Hz
    decimate = int(fs_real / target_fs)
    if decimate > 1:
        raw = signal.decimate(raw, decimate, ftype='fir')
    fs = target_fs

    # 使用常规滤波（filtfilt但数据量小）
    clean = signal.filtfilt(*signal.butter(1, 0.5, btype='high', fs=fs), raw)
    if fs > 105:
        clean = signal.filtfilt(*signal.iirnotch(50, 3, fs), clean)
    clean = signal.filtfilt(*signal.butter(2, 50, btype='low', fs=fs), clean)
    clean = -smoothing_filter(clean)

    return extract_features_from_cleaned_signal(clean, fs)

def update_features_loop():
    global feature_cache
    while True:
        time.sleep(15)   # 降低频率到15秒一次
        try:
            new_features = extract_features_downsampled(duration=20)
            if new_features:
                feature_cache = new_features
                print("特征缓存已更新（降采样模式）")
        except Exception as e:
            print(f"特征更新失败: {e}")

# ---------- 实时波形推送（轻量级 lfilter） ----------
# 滤波器状态缓存
baseline_zi = None
notch_zi = None
lowpass_zi = None
last_idx = 0

def generate_ecg_stream():
    global baseline_zi, notch_zi, lowpass_zi, last_idx
    while True:
        fs = max(shared_fs.value, 10.0)
        win_sec = WINDOW_SECOND
        DELAY_SECOND = 1.0
        EXTRA_RUNWAY = 1.0

        display_points = int(fs * win_sec)
        delay_points = int(fs * DELAY_SECOND)
        fetch_points = int(fs * (win_sec + DELAY_SECOND + EXTRA_RUNWAY))

        idx = current_index.value
        if idx < fetch_points + 100:
            time.sleep(0.05)
            continue

        # 每次取最新数据
        raw = np.empty(fetch_points)
        for i in range(fetch_points):
            raw[i] = shared_array[(idx - fetch_points + i) % BUFFER_SIZE]

        # 使用 lfilter 进行轻量滤波
        if baseline_zi is None:
            b_b, a_b = signal.butter(1, 0.5, btype='high', fs=fs)
            baseline_zi = signal.lfilter_zi(b_b, a_b) * raw[0]
        if notch_zi is None and fs > 105:
            b_n, a_n = signal.iirnotch(50, 3, fs)
            notch_zi = signal.lfilter_zi(b_n, a_n) * raw[0]
        if lowpass_zi is None:
            b_l, a_l = signal.butter(1, min(50.0, fs*0.45), btype='low', fs=fs)
            lowpass_zi = signal.lfilter_zi(b_l, a_l) * raw[0]

        # 顺序滤波
        clean, baseline_zi = baseline_filter_lfilter(raw, fs, baseline_zi)
        if fs > 105:
            clean, notch_zi = notch_filter_lfilter(clean, fs, notch_zi)
        clean, lowpass_zi = low_pass_filter_lfilter(clean, fs, lowpass_zi)

        # 平滑（可选，轻量）
        # clean = smoothing_filter(clean)  # 暂禁用平滑以提速，效果影响不大
        clean = -clean

        stable = clean[-(display_points + delay_points): -delay_points]
        if len(stable) != display_points:
            time.sleep(0.05)
            continue

        # 心率计算（每两帧算一次可进一步降低负载，这里保持每帧）
        raw_bpm = bpm_calculate(stable, fs)
        if raw_bpm > 0:
            bpm_queue.append(raw_bpm)
            if len(bpm_queue) > 5:
                bpm_queue.pop(0)
            bpm = int(np.median(bpm_queue))
        else:
            bpm = 0

        if bpm > 0:
            heart_rate_history.append((time.time(), bpm))
            while len(heart_rate_history) > 300:
                heart_rate_history.pop(0)

        msg = {
            'waveform': stable.tolist(),
            'bpm': bpm,
            'fs': fs,
            'window_sec': win_sec
        }
        yield f"data: {json.dumps(msg)}\n\n"
        # 降低推送频率到 200ms (5fps)
        time.sleep(0.2)

# ----------------------------- Flask 路由 -----------------------------
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/stream')
def stream():
    return Response(generate_ecg_stream(), mimetype='text/event-stream')

@app.route('/api/features', methods=['GET'])
def get_features():
    return jsonify(feature_cache if feature_cache else {})

@app.route('/api/compute_features', methods=['POST'])
def compute_features():
    """手动触发实时特征提取（基于最近10秒数据）"""
    features = extract_features_downsampled(duration=10)
    if not features:
        return jsonify({'error': '特征提取失败，可能是数据不足或信号质量差'}), 400
    return jsonify(features)

@app.route('/api/set_window', methods=['POST'])
def set_window():
    global WINDOW_SECOND
    data = request.json
    new_win = float(data.get('window_sec', 4.0))
    if new_win not in (4.0, 10.0):
        new_win = 4.0
    WINDOW_SECOND = new_win
    print(f"窗口宽度已设置为 {WINDOW_SECOND} 秒")
    return jsonify({'window_sec': WINDOW_SECOND})

@app.route('/api/heartrate_stats', methods=['GET'])
def heartrate_stats():
    now = time.time()
    recent = [hr for t, hr in heart_rate_history if now - t <= 30]
    if not recent:
        return jsonify({'avg': 0, 'max': 0, 'min': 0, 'current': 0})
    return jsonify({
        'avg': round(np.mean(recent), 1),
        'max': int(np.max(recent)),
        'min': int(np.min(recent)),
        'current': int(recent[-1]) if recent else 0
    })

@app.route('/api/report', methods=['POST'])
def api_report():
    data = request.json
    patient_info = data.get('patient_info', {})
    mode = data.get('mode', 'detailed')
    now = time.time()
    recent_hr = [hr for t, hr in heart_rate_history if now - t <= 30]
    if not recent_hr:
        return jsonify({'error': '心率数据不足，请等待几秒后再试'}), 400
    avg_hr = np.mean(recent_hr)
    min_hr = np.min(recent_hr)
    max_hr = np.max(recent_hr)
    current_hr = recent_hr[-1]
    features = feature_cache if feature_cache else {}

    if mode == 'simple':
        sys_prompt = "你是一名资深心血管内科医生。请基于心电数据、个人信息和生活习惯，用简短的几句话（200字以内）总结重点异常指标和总体建议。"
    else:
        sys_prompt = """你是一名资深心血管内科医生，心电图分析专家。请基于以下心电数据、个人信息和生活习惯，提供临床心电分析报告。

## 输出要求：
1. 报告结构清晰，包括：基本数据陈述、核心指标解读（心率、心律、HRV等）、详细临床特征（PR间期、QRS宽度、QTc、ST段等）、心律失常风险提示、生活方式建议及下一步检查建议。
2. 避免绝对化的诊断语言，使用“提示”、“符合…表现”、“建议进一步检查”等。
3. 危急征象需用⚠️警告。
4. 结合患者个人信息和生活习惯给出个体化建议。
5. 输出语言为中文，通俗易懂但专业。"""

    user_text = f"""
【心率统计（最近30秒）】
- 当前心率: {current_hr:.0f} bpm
- 平均心率: {avg_hr:.1f} bpm
- 最小心率: {min_hr:.0f} bpm
- 最大心率: {max_hr:.0f} bpm

【实时特征】
- PR间期: {features.get('PR间期 (ms)', '--')} ms
- QRS宽度: {features.get('QRS宽度 (ms)', '--')} ms
- QTc: {features.get('QTc (Bazett) (ms)', '--')} ms
- ST段偏移: {features.get('ST段偏移 (μV)', '--')} μV
- SDNN: {features.get('SDNN (ms)', '--')} ms
- 信号质量: {features.get('信号质量', '--')}

【个人信息】
- 年龄: {patient_info.get('age', '未知')}岁
- 性别: {patient_info.get('gender', '未知')}
- BMI: {patient_info.get('bmi', '未知')}
- 吸烟: {patient_info.get('smoking', '未填写')}
- 饮酒: {patient_info.get('alcohol', '未填写')}
- 高血压: {patient_info.get('hypertension', '未填写')}
- 糖尿病: {patient_info.get('diabetes', '未填写')}
- 冠心病史: {patient_info.get('cad', '未填写')}
- 熬夜习惯: {patient_info.get('late_night', '未填写')}
- 运动频率: {patient_info.get('exercise', '未填写')}
"""
    api_key = "sk-7eb7b002610a41678066afd1c41c777b"
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=0.3,
            max_tokens=1500 if mode == 'simple' else 2500
        )
        report = response.choices[0].message.content
        return jsonify({'report': report})
    except Exception as e:
        return jsonify({'error': f'API调用失败: {str(e)}'}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.json
    question = data.get('question')
    if not question:
        return jsonify({'error': '请输入问题'}), 400
    if not heart_rate_history:
        return jsonify({'error': '心电数据不足，请稍后再问'}), 400
    now = time.time()
    recent_hr = [hr for t, hr in heart_rate_history if now - t <= 30]
    avg_hr = np.mean(recent_hr) if recent_hr else 0
    current_hr = recent_hr[-1] if recent_hr else 0
    features = feature_cache if feature_cache else {}
    context = f"""当前用户的心电数据概览（最近30秒）：
- 当前心率: {current_hr:.0f} bpm
- 平均心率: {avg_hr:.1f} bpm
- PR间期: {features.get('PR间期 (ms)', '--')} ms
- QRS宽度: {features.get('QRS宽度 (ms)', '--')} ms
- QTc: {features.get('QTc (Bazett) (ms)', '--')} ms
- ST段偏移: {features.get('ST段偏移 (μV)', '--')} μV
- SDNN: {features.get('SDNN (ms)', '--')} ms

请基于以上数据，结合医学知识回答用户的问题。回答应当专业、简洁、有帮助，并提醒用户仅供参考，不能替代医生诊断。"""
    sys_prompt = "你是一名心内科医生助手，请根据用户的心电数据上下文回答用户的健康问题。"
    user_prompt = f"{context}\n\n用户问题：{question}"
    api_key = "sk-7eb7b002610a41678066afd1c41c777b"
    try:
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.5,
            max_tokens=800
        )
        answer = response.choices[0].message.content
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    init_shared_memory()
    import threading
    t = threading.Thread(target=update_features_loop, daemon=True)
    t.start()
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)