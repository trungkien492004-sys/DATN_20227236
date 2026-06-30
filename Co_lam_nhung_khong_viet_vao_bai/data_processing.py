import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os

os.makedirs('outputs', exist_ok=True)

def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def plot_label_dist(df, label_cols, title, filename):
    n = len(label_cols)
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for i, col in enumerate(label_cols):
        df[col].value_counts().plot(kind='bar', ax=axes[i], 
                                     color=['steelblue','coral'])
        axes[i].set_title(col)
        axes[i].tick_params(rotation=0)
    plt.suptitle(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'outputs/{filename}', dpi=150)
    plt.show()
    print(f"Saved: outputs/{filename}")

print("Setup done")
df_essays = pd.read_csv('essays.csv', encoding='cp1252')

# Xóa cột rác
df_essays = df_essays.drop(columns=['Unnamed: 7'], errors='ignore')

# Đổi tên cho nhất quán
df_essays = df_essays.rename(columns={'text': 'TEXT'})

print("=== ESSAYS DATASET ===")
print(f"Shape: {df_essays.shape}")

# Độ dài
df_essays['text_length'] = df_essays['TEXT'].apply(
    lambda x: len(str(x).split()))
print(df_essays['text_length'].describe())

# Làm sạch
df_essays['TEXT_CLEAN'] = df_essays['TEXT'].apply(clean_text)

# Convert nhãn y/n → 1/0
for col in ['cEXT','cNEU','cAGR','cCON','cOPN']:
    df_essays[col] = df_essays[col].map({'y':1, 'n':0})

# Plot
plot_label_dist(df_essays,
                ['cEXT','cNEU','cAGR','cCON','cOPN'],
                'Essays Dataset — Label Distribution',
                'essays_labels.png')

df_essays.to_csv('outputs/essays_clean.csv', index=False)
print(f"Essays sau làm sạch: {len(df_essays)} mẫu")
# Tải từ Kaggle: mbti_1.csv

df_mbti = pd.read_csv('mbti_1.csv')
print("\n=== MBTI DATASET ===")
print(f"Shape: {df_mbti.shape}")
print(f"Columns: {df_mbti.columns.tolist()}")
print(df_mbti['type'].value_counts())

# Ánh xạ MBTI → Big Five (4 chiều)
def mbti_to_bigfive(mbti):
    return {
        'E': 1 if mbti[0] == 'E' else 0,  # Extraversion
        'N': 1 if mbti[1] == 'N' else 0,  # Openness
        'F': 1 if mbti[2] == 'F' else 0,  # Agreeableness
        'P': 1 if mbti[3] == 'P' else 0,  # low Conscientiousness
    }

mbti_mapped = df_mbti['type'].apply(mbti_to_bigfive)
df_mbti[['mEXT','mOPN','mAGR','mCON']] = pd.DataFrame(
    mbti_mapped.tolist(), index=df_mbti.index)

# Độ dài
df_mbti['text_length'] = df_mbti['posts'].apply(
    lambda x: len(str(x).split()))
print(df_mbti['text_length'].describe())

# Làm sạch
df_mbti['TEXT_CLEAN'] = df_mbti['posts'].apply(clean_text)
df_mbti = df_mbti[df_mbti['text_length'] >= 50].reset_index(drop=True)

# Plot phân phối 16 type
plt.figure(figsize=(14, 4))
df_mbti['type'].value_counts().plot(kind='bar', color='steelblue')
plt.title('MBTI Dataset — Type Distribution')
plt.tight_layout()
plt.savefig('outputs/mbti_types.png', dpi=150)
plt.show()

df_mbti.to_csv('outputs/mbti_clean.csv', index=False)
print(f"MBTI sau làm sạch: {len(df_mbti)} mẫu")

# Đã có sẵn mypersonality_clean.csv từ hôm trước
def clean_text(text):
    text = str(text)
    text = re.sub(r'http\S+', '', text)                 # remove urls
    text = re.sub(r'[^a-zA-Z\s\.\,\!\?]', ' ', text)   # keep basic chars
    text = re.sub(r'\s+', ' ', text)                   # normalize spaces
    return text.strip()


# =========================
# LOAD DATASET
# =========================
df = pd.read_csv(
    'mypersonality_test.csv',
    encoding='latin1'   # tránh UnicodeDecodeError
)

df.columns = df.columns.str.strip()


# =========================
# GROUP BY USER
# =========================
df_myp = df.groupby('X.AUTHID').agg({
    'STATUS': lambda x: ' '.join(x.dropna().astype(str)),
    'sEXT': 'first',
    'sNEU': 'first',
    'sAGR': 'first',
    'sCON': 'first',
    'sOPN': 'first',
    'cEXT': 'first',
    'cNEU': 'first',
    'cAGR': 'first',
    'cCON': 'first',
    'cOPN': 'first',
}).reset_index()


# =========================
# CLEAN TEXT
# =========================
df_myp['TEXT_CLEAN'] = df_myp['STATUS'].apply(clean_text)

df_myp['text_length'] = df_myp['TEXT_CLEAN'].apply(
    lambda x: len(x.split())
)

df_myp = df_myp[df_myp['text_length'] >= 20].reset_index(drop=True)


# =========================
# SAVE CLEAN DATA
# =========================
df_myp.to_csv('mypersonality_clean.csv', index=False, encoding='utf-8-sig')


# =========================
# BASIC INFO
# =========================
print("\n=== myPERSONALITY DATASET ===")
print(f"Shape: {df_myp.shape}")

traits_s = ['sEXT', 'sNEU', 'sAGR', 'sCON', 'sOPN']

print("\nScore Statistics:")
print(df_myp[traits_s].describe())


# =========================
# PLOT DISTRIBUTION
# =========================
os.makedirs('outputs', exist_ok=True)

fig, axes = plt.subplots(1, 5, figsize=(20, 4))

for i, trait in enumerate(traits_s):
    df_myp[trait].hist(
        bins=15,
        ax=axes[i]
    )
    axes[i].set_title(trait)

plt.suptitle('myPersonality Score Distribution', fontweight='bold')
plt.tight_layout()

plt.savefig('outputs/myp_scores.png', dpi=150)
plt.show()


print(f"\nDone — {len(df_myp)} users retained after filtering")

# Tải từ: https://data.mendeley.com/datasets/3sndbd4p84/1
# Bấm "Download All" → giải nén → tìm file train/val/test

# Thử đọc (tên file có thể khác, kiểm tra sau khi giải nén)
df_train = pd.read_csv('train_set.csv')
df_val   = pd.read_csv('val_set.csv')
df_eval  = pd.read_csv('eval_set.csv')

print("=== MENDELEY REDDIT BIG FIVE ===")
print(f"Train: {df_train.shape}")
print(f"Val:   {df_val.shape}")
print(f"Eval:  {df_eval.shape}")
print(f"\nColumns: {df_train.columns.tolist()}")
print(df_train.head(2))
# Sau khi biết tên file, đọc vào:
# df_mendeley = pd.read_csv('TÊN_FILE.csv')
# print(df_mendeley.shape)
# print(df_mendeley.columns.tolist())

# Tải từ Kaggle: train.txt, val.txt, test.txt

def load_emotions(filepath):
    texts, labels = [], []
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(';')
            if len(parts) == 2:
                texts.append(parts[0])
                labels.append(parts[1])
    return pd.DataFrame({'text': texts, 'emotion': labels})

df_emo = load_emotions('train.txt')
print("\n=== EMOTIONS DATASET ===")
print(f"Shape: {df_emo.shape}")
print(df_emo['emotion'].value_counts())

# Plot
plt.figure(figsize=(8, 4))
df_emo['emotion'].value_counts().plot(kind='bar', color='coral')
plt.title('Emotions Dataset — Label Distribution')
plt.tight_layout()
plt.savefig('outputs/emotions_labels.png', dpi=150)
plt.show()

df_emo['TEXT_CLEAN'] = df_emo['text'].apply(clean_text)
df_emo.to_csv('outputs/emotions_clean.csv', index=False)
print(f"Emotions: {len(df_emo)} mẫu")

print("\n" + "="*50)
print("TỔNG KẾT CÁC BỘ DỮ LIỆU")
print("="*50)

summary = {
    'Dataset':   ['Essays', 'MBTI', 'myPersonality', 'Mendeley', 'Emotions'],
    'Loại nhãn': ['Big Five', 'MBTI→Big Five', 'Big Five', 'Big Five', 'Cảm xúc'],
    'Số mẫu':    [len(df_essays), len(df_mbti), len(df_myp), 'TBD', len(df_emo)],
    'Loại text': ['Bài luận', 'Forum posts', 'Facebook', 'Reddit', 'Social media'],
}

df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))
df_summary.to_csv('outputs/dataset_summary.csv', index=False)

# Cập nhật summary đúng số thật
summary = {
    'Dataset':    ['Essays', 'MBTI', 'myPersonality', 'Mendeley Reddit', 'Emotions'],
    'Loại nhãn':  ['Big Five', 'MBTI→Big Five', 'Big Five', 'Big Five', 'Cảm xúc'],
    'Số mẫu':     [2465, 8670, 45, 16047+2415+2415, 16000],
    'Loại text':  ['Bài luận', 'Forum posts', 'Facebook', 'Reddit', 'Social media'],
    'Dùng cho':   ['Main', 'Main', 'Phụ', 'Main', 'Mở rộng'],
}
df_summary = pd.DataFrame(summary)
print(df_summary.to_string(index=False))
df_summary.to_csv('outputs/dataset_summary.csv', index=False)

from sklearn.preprocessing import MinMaxScaler

trait_cols = ['agreeableness', 'openness',
              'conscientiousness', 'extraversion', 'neuroticism']

scaler = MinMaxScaler()
df_train[trait_cols] = scaler.fit_transform(df_train[trait_cols])
df_val[trait_cols]   = scaler.transform(df_val[trait_cols])
df_eval[trait_cols]  = scaler.transform(df_eval[trait_cols])

df_train.to_csv('outputs/mendeley_train_clean.csv', index=False)
df_val.to_csv('outputs/mendeley_val_clean.csv', index=False)
df_eval.to_csv('outputs/mendeley_eval_clean.csv', index=False)

print("Mendeley chuẩn hóa xong")
print(df_train[trait_cols].describe())