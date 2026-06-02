import pandas as pd
import ast
from collections import Counter

DATA_PATH = "/root/ecg_tiny/"

# 1. 加载数据
df = pd.read_csv(DATA_PATH + "ptbxl_database.csv")
# 重要：scp_statements.csv 第一列是代码（无列名），设置为索引
scp_df = pd.read_csv(DATA_PATH + "scp_statements.csv", index_col=0)

# 2. 构建诊断性代码集合 (diagnostic == 1) ，代码来自索引
diag_codes = set(scp_df[scp_df['diagnostic'] == 1].index)

# 3. 定义函数：将scp_codes字符串转换为诊断超类列表
def get_superclasses(scp_str):
    try:
        codes = ast.literal_eval(scp_str)
        superclasses = set()
        for code in codes.keys():
            if code in diag_codes:
                # 从scp_df中查找该代码的 diagnostic_class
                row = scp_df.loc[code]   # 通过索引取行
                diag_class = row.get('diagnostic_class')
                if pd.notna(diag_class):
                    superclasses.add(diag_class)
        return list(superclasses)
    except:
        return []

df['superclasses'] = df['scp_codes'].apply(get_superclasses)

# 4. 定义二分类：正常当且仅当 superclasses == ['NORM'] 或 superclasses == [] (无诊断代码)
df['is_normal'] = df['superclasses'].apply(lambda x: x == ['NORM'] or len(x) == 0)
df['is_abnormal'] = ~df['is_normal']

# 5. 统计
total = len(df)
normal_cnt = df['is_normal'].sum()
abnormal_cnt = df['is_abnormal'].sum()

print("="*50)
print("PTB-XL 数据集标签统计")
print("="*50)
print(f"总记录数: {total}")
print(f"正常记录: {normal_cnt} ({normal_cnt/total*100:.1f}%)")
print(f"异常记录: {abnormal_cnt} ({abnormal_cnt/total*100:.1f}%)")
print()

# 6. 多标签统计（五个超类）
target_classes = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
class_counter = Counter()
for classes in df['superclasses']:
    for c in classes:
        if c in target_classes:
            class_counter[c] += 1

print("多标签分布（每个超类出现的次数，一条记录可能贡献多个超类）:")
for c in target_classes:
    cnt = class_counter.get(c, 0)
    print(f"  {c}: {cnt} 次 ({cnt/total*100:.1f}%)")
print("="*50)