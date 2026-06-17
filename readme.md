# PTB-XL 心电信号特征提取与智能分类
<!--
上面这一行是 一级标题（#）
语法：# 标题文字（#和文字之间必须有空格）
-->

基于 PTB-XL 公开心电数据库，通过 NeuroKit2 提取临床特征，构建随机森林（Random Forest）与 XGBoost 模型，实现心电信号的二分类（正常/异常）与五分类（MI/STTC/CD/HYP/NORM）任务。

## 项目结构
<!--
上面这一行是 二级标题（##）
语法：## 标题文字
-->

```
<!--
上面这里是 代码块（```），用于展示文件目录结构
语法：``` 开头，``` 结尾，中间放纯文本
不加语言名称表示纯文本，GitHub 不会做语法高亮
-->

.
├── extract.py                  # 特征提取脚本（500Hz, Lead II）
├── train.py                    # 二分类训练脚本（正常 vs 异常）
├── train5.py                   # 五分类训练脚本（MI/STTC/CD/HYP/NORM）
├── detailed_features.csv       # 提取后的特征文件（运行 extract.py 生成）
├── models/                     # 训练好的模型
│   ├── strict_binary_best.pkl
│   └── 5class_best_model.pkl
├── results/                    # 评估结果
│   ├── strict_binary_evaluation.csv
│   └── 5class_evaluation.csv
└── README.md
```

## 数据集

- **来源**：PhysioNet PTB-XL，大规模公开12导联心电图数据集
- **完整样本量**：21,837 条临床记录（本项目使用约 400 条子集）
- **采样率**：500 Hz（使用 `records500/`）
- **导联**：Lead II（第二导联）
- **标签体系**：SCP-ECG 标准，包含 71 种诊断语句
<!--
上面这几行是 无序列表（-）
语法：- 列表项（横杠和文字之间必须有空格）
`records500/` 是 行内代码，语法：用反引号 ` 包裹文字
-->

### 五大诊断超类
<!--
上面这一行是 三级标题（###）
-->

| 缩写 | 英文全称 | 中文全称 |
| :--- | :--- | :--- |
| NORM | Normal | 正常心电图 |
| MI | Myocardial Infarction | 心肌梗塞 |
| STTC | ST/T Change | ST/T段改变 |
| CD | Conduction Disturbance | 传导障碍 |
| HYP | Hypertrophy | 心肌肥厚 |
<!--
上面这个是一个 表格
语法：
1. 第一行是表头：| 列1 | 列2 | 列3 |
2. 第二行是分隔线：| :--- | :---: | ---: |
   - :--- 表示左对齐
   - :---: 表示居中对齐
   - ---: 表示右对齐
3. 后面的行是表格内容：| 内容A | 内容B | 内容C |
注意：每行的 | 数量必须一致，空格不影响渲染
-->

> **注意**：PTB-XL 为多标签数据集，一条记录可同时包含多个诊断。五分类任务中采用优先级策略处理标签冲突：**MI > STTC > CD > HYP > NORM**。
<!--
上面这一段是 引用块（>）
语法：> 引用内容
**注意** 是 粗体，语法：**文字** 或 __文字__
-->

## 特征提取

### 预处理
- 带通滤波：0.5–40 Hz
- 工具：`wfdb` 读取原始信号，`neurokit2` 进行波形分割（DWT方法）

### 提取特征（共 26 维）

| 类别 | 特征 |
| :--- | :--- |
| 心率变异性 | 心率均值/标准差/最大/最小值、SDNN、RMSSD、pNN50 |
| 间期与时限 | PR间期、QRS宽度、QT间期、QTc（Bazett校正）|
| 幅度与偏移 | P/T/R/S波幅度、ST段偏移、R/S比值、QRS面积 |
| 病理标志 | 病理性Q波、T波对称性 |
| 信号质量 | 信噪比（SNR）、基线漂移比 |

## 实验设计

### 任务一：二分类（疾病筛查）

| 类别 | 定义 |
| :--- | :--- |
| 正常（0） | NORM = 1，且 MI=STTC=CD=HYP=0 |
| 异常（1） | 包含 MI / STTC / CD / HYP 中任意一项 |

### 任务二：五分类（细粒度诊断）

| 类别 | 优先级 |
| :--- | :---: |
| MI（心肌梗塞） | 1（最高） |
| STTC（ST/T段改变） | 2 |
| CD（传导障碍） | 3 |
| HYP（心肌肥厚） | 4 |
| NORM（正常） | 5 |
<!--
上面第二个表格的第二列用了 :---: 表示居中对齐，数字看起来更整齐
-->

## 实验结果

### 二分类任务（正常 vs 异常）

| 模型 | 评估维度 | 准确率 | AUC | 精确率 | 召回率 | F1分数 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 随机森林 | 整体（加权平均） | 80.00% | 0.8560 | 80% | 80% | 80% |
| 随机森林 | 正常（NORM） | - | - | 81% | 88% | 85% |
| 随机森林 | 异常（患病） | - | - | 77% | 67% | 71% |
| XGBoost | 整体（加权平均） | 80.00% | 0.8513 | 80% | 80% | 80% |
| XGBoost | 正常（NORM） | - | - | 83% | 86% | 84% |
| XGBoost | 异常（患病） | - | - | 75% | 70% | 72% |

### 五分类任务（MI/STTC/CD/HYP/NORM）

| 模型 | 评估维度 | 准确率 | 精确率 | 召回率 | F1分数 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| 随机森林 | 整体（加权平均） | 67.50% | 59% | 68% | 61% |
| 随机森林 | 心肌梗死（MI） | - | 25% | 9% | 13% |
| 随机森林 | ST段异常（STTC） | - | 43% | 27% | 33% |
| 随机森林 | 传导障碍（CD） | - | 50% | 14% | 22% |
| 随机森林 | 心肌肥厚（HYP） | - | 0% | 0% | 0% |
| 随机森林 | 正常心电（NORM） | - | 73% | 98% | 84% |
| XGBoost | 整体（加权平均） | 66.25% | 60% | 66% | 62% |
| XGBoost | 心肌梗死（MI） | - | 17% | 9% | 12% |
| XGBoost | ST段异常（STTC） | - | 55% | 55% | 55% |
| XGBoost | 传导障碍（CD） | - | 25% | 14% | 18% |
| XGBoost | 心肌肥厚（HYP） | - | 0% | 0% | 0% |
| XGBoost | 正常心电（NORM） | - | 76% | 90% | 83% |

## 分析与讨论

- **二分类性能良好**：随机森林与 XGBoost 在二分类任务中表现相当，AUC 均达到 0.85 以上，验证了特征工程路线的有效性，具备初步疾病筛查潜力。
- **五分类面临挑战**：整体准确率约 67%，模型对 NORM 识别效果最佳（F1约84%），但对 MI、CD、HYP 等少数类识别能力有限。
- **根本瓶颈**：少数类样本严重不足（如 HYP 在测试集中仅 1 例），导致模型难以有效学习此类特征。后续可通过数据增强（SMOTE等）、重采样或引入完整数据集进一步优化。
- **模型对比**：随机森林与 XGBoost 在各任务中表现接近，未出现明显优劣势分化。

## 快速开始

### 1. 安装依赖

```bash
pip install numpy pandas scipy scikit-learn xgboost wfdb neurokit2 tqdm joblib
```
<!--
上面这个是 代码块指定了语言 bash
语法：```bash 开头，``` 结尾，中间放 bash 命令
GitHub 会自动高亮显示为 shell 语法
-->

### 2. 下载数据

```bash
wget -r -N -c -np https://physionet.org/files/ptb-xl/1.0.1/
```

### 3. 特征提取

```bash
python extract.py
```

### 4. 训练与评估

```bash
# 二分类
python train.py

# 五分类
python train5.py
```
<!--
上面 `# 二分类` 是注释，不影响命令执行，只是给人看的
-->

## 依赖环境

- Python >= 3.8
- numpy
- pandas
- scipy
- scikit-learn
- xgboost
- wfdb
- neurokit2
- tqdm
- joblib

## 引用

如使用本工作，请引用原始数据集：

```bibtex
@article{wagner2020ptb,
  title={PTB-XL, a large publicly available electrocardiography dataset},
  author={Wagner, Patrick and Strodthoff, Nils and Bousseljot, Ralf-Dieter and Samek, Wojciech and Schaeffter, Tobias},
  journal={Scientific Data},
  volume={7},
  number={1},
  pages={1--15},
  year={2020},
  publisher={Nature Publishing Group}
}
```
<!--
上面指定了 bibtex 语言，GitHub 会按 BibTeX 格式高亮
-->

## 许可证

- 项目代码：MIT License
- 数据集：CC BY 4.0 International Public License

---
<!--
上面这一行是 分割线（--- 或 ***）
用来分隔文档的最后一个章节，表示文档结束
-->