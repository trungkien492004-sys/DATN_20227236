import os
import json
import time
import pandas as pd
from pathlib import Path
from sklearn.metrics import classification_report, f1_score
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# PATHS (tương đối theo vị trí script -> clone về vẫn chạy)
# ============================================================

BASE_DIR   = Path(__file__).resolve().parent                       # Prompt/Code_Prompt
OUTPUT_DIR = BASE_DIR.parent / "Output_prompt" / "mbti_qwen"       # Prompt/Output_prompt/mbti_qwen
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = "mbti_clean_deleaked.csv"

def find_data_file(filename, root):
    matches = list(root.rglob(filename))
    if not matches:
        raise FileNotFoundError(
            f"Không tìm thấy '{filename}' trong {root}. "
            f"Kiểm tra lại tên file hoặc đặt file vào trong project."
        )
    return matches[0]

# BASE_DIR.parent.parent = root project (DATA DATN)
DATA_PATH = find_data_file(DATA_FILE, BASE_DIR.parent.parent)

# ============================================================
# SETUP
# ============================================================

load_dotenv(dotenv_path=BASE_DIR / "API.env")

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
)

MODEL = "qwen/qwen-2.5-7b-instruct"
TEXT_COL  = "TEXT_CLEAN"
LABEL_COL = "type"

df = pd.read_csv(DATA_PATH, engine="python")
df = df.dropna(subset=[TEXT_COL, LABEL_COL]).reset_index(drop=True)
df["sample_id"] = df.index
print(f"Loaded {len(df)} samples")
print(df[LABEL_COL].value_counts().head(8))

# File merge cuối (dùng để resume nếu không có checkpoint riêng)
MODEL_SLUG  = MODEL.replace("/", "-").replace(":", "-")
MERGED_FILE = OUTPUT_DIR / f"predictions_mbti_{MODEL_SLUG}.csv"

# ============================================================
# PROMPT TEMPLATES
# ============================================================

ZERO_SHOT_PROMPT = """You are an expert MBTI personality analyst.

Analyze the following text and predict the MOST likely MBTI personality type.

Possible MBTI types:
INTJ, INTP, ENTJ, ENTP,
INFJ, INFP, ENFJ, ENFP,
ISTJ, ISFJ, ESTJ, ESFJ,
ISTP, ISFP, ESTP, ESFP

TEXT:
\"\"\"{text}\"\"\"

Return ONLY valid JSON:
{{"type": "XXXX"}}"""


FEW_SHOT_PROMPT = """You are an expert MBTI personality analyst.
Predict the MOST likely MBTI type from the text.

EXAMPLE 1 (INTJ):
Text: "I enjoy analyzing abstract systems independently. I plan everything carefully and dislike inefficiency."
Output: {{"type": "INTJ"}}

EXAMPLE 2 (ENFP):
Text: "I love meeting new people and exploring spontaneous adventures. I get bored with routine."
Output: {{"type": "ENFP"}}

EXAMPLE 3 (INFJ):
Text: "I care deeply about helping others but need alone time to recharge. I see patterns and meaning everywhere."
Output: {{"type": "INFJ"}}

Now analyze this text:

TEXT:
\"\"\"{text}\"\"\"

Return ONLY valid JSON:
{{"type": "XXXX"}}"""


PROMPTS = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot":  FEW_SHOT_PROMPT,
}

MBTI_TYPES = [
    "INTJ","INTP","ENTJ","ENTP",
    "INFJ","INFP","ENFJ","ENFP",
    "ISTJ","ISFJ","ESTJ","ESFJ",
    "ISTP","ISFP","ESTP","ESFP"
]

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_mbti(prompt_template, text, prompt_name, retries=5):
    text_truncated = text
    prompt = prompt_template.format(text=text_truncated)

    max_tok = 30

    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=0.35,
                max_tokens=max_tok,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert MBTI analyst. Respond ONLY with valid JSON. No explanation, no markdown."
                    },
                    {"role": "user", "content": prompt}
                ]
            )
            raw = response.choices[0].message.content.strip()

            cleaned = raw.replace("```json", "").replace("```", "").strip()

            start = cleaned.find("{")
            end   = cleaned.rfind("}") + 1

            parsed = json.loads(cleaned[start:end])

            mbti_type = parsed.get("type", "").upper().strip()
            if mbti_type in MBTI_TYPES:
                return mbti_type
            return None

        except json.JSONDecodeError:
            print(f"  JSON error (attempt {attempt+1})")

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate_limit" in err:
                wait = min(2 ** attempt, 30)
                print(f"  Rate limit → wait {wait}s")
                time.sleep(wait)
            else:
                print(f"  Error: {e}")
                break

    return None

# ============================================================
# RUN EXPERIMENT với CHECKPOINT
# ============================================================

def load_prior_results(output_dir, id_col, strategy):
    """Quét mọi .csv trong OUTPUT_DIR, gom mẫu đã xong theo strategy (mọi tên file)."""
    done_ids, results = set(), []
    for f in Path(output_dir).glob("*.csv"):
        try:
            d = pd.read_csv(f)
        except Exception:
            continue
        if id_col not in d.columns or "prompt_strategy" not in d.columns:
            continue
        d = d[d["prompt_strategy"] == strategy]
        for r in d.to_dict("records"):
            if r[id_col] not in done_ids:
                done_ids.add(r[id_col])
                results.append(r)
    return done_ids, results


def run_experiment(df_input, prompt_name, prompt_template):
    print(f"\n{'='*60}")
    print(f"RUNNING: {prompt_name.upper()} | {MODEL} | {len(df_input)} samples")
    print(f"{'='*60}")

    ckpt_file  = OUTPUT_DIR / f"checkpoint_mbti_{prompt_name}_{MODEL_SLUG}.csv"
    final_file = OUTPUT_DIR / f"predictions_mbti_{prompt_name}_{MODEL_SLUG}.csv"

    all_ids = set(df_input["sample_id"].tolist())

    # --- Gom mọi kết quả cũ trong OUTPUT_DIR (mọi tên file) ---
    done_ids, results = load_prior_results(OUTPUT_DIR, "sample_id", prompt_name)
    print(f"  Resume: {len(done_ids)} done, {len(all_ids) - len(done_ids & all_ids)} remaining")

    for _, row in df_input.iterrows():
        sid = row["sample_id"]
        if sid in done_ids:
            continue

        print(f"  [{sid+1}/{len(df_input)}]", end=" ")
        pred = predict_mbti(prompt_template, str(row[TEXT_COL]), prompt_name)
        print(f"TRUE={row[LABEL_COL]} | PRED={pred}")

        results.append({
            "sample_id":       sid,
            "model":           MODEL,
            "true_type":       row[LABEL_COL],
            "pred_type":       pred,
            "prompt_strategy": prompt_name,
            "text_length":     len(str(row[TEXT_COL]).split()),
        })
        done_ids.add(sid)

        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(ckpt_file, index=False)
            print(f"  Checkpoint saved ({len(results)} done)")

        time.sleep(0.5)  # Mistral cần sleep hơn GPT

    df_result = pd.DataFrame(results)
    df_result.to_csv(ckpt_file, index=False)
    df_result.to_csv(final_file, index=False)
    print(f"  Done: {len(df_result)} samples")
    return df_result

# ============================================================
# RUN ALL
# ============================================================

all_results = []

for prompt_name, prompt_template in PROMPTS.items():
    df_result = run_experiment(df, prompt_name, prompt_template)
    all_results.append(df_result)

final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(MERGED_FILE, index=False)
print(f"\nSaved {MERGED_FILE}")

# ============================================================
# EVALUATION — 4 CHIỀU NHỊ PHÂN (khớp fine-tune) + 16 lớp tham chiếu
# ============================================================
from sklearn.metrics import accuracy_score, precision_score, recall_score

TRAITS = ["mEXT", "mOPN", "mAGR", "mCON"]
# Ánh xạ đã kiểm trên dữ liệu: bit=1 khi ký tự loại thuộc nhóm dưới đây
#   mEXT: pos0 == 'E'  | mOPN: pos1 == 'N' | mAGR: pos2 == 'F' | mCON: pos3 == 'P'
POS  = {"mEXT": 0, "mOPN": 1, "mAGR": 2, "mCON": 3}
CHAR = {"mEXT": "E", "mOPN": "N", "mAGR": "F", "mCON": "P"}

def type_to_bits(t):
    """Tách loại 4 ký tự thành 4 nhãn nhị phân. Trả None nếu loại không hợp lệ."""
    if not isinstance(t, str) or len(t.strip()) != 4:
        return {tr: None for tr in TRAITS}
    t = t.upper().strip()
    return {tr: int(t[POS[tr]] == CHAR[tr]) for tr in TRAITS}

print("\n" + "=" * 60)
print("EVALUATION — 4 CHIỀU NHỊ PHÂN")
print("=" * 60)

summary_rows = []

for strategy in PROMPTS.keys():
    subset = final_df[final_df["prompt_strategy"] == strategy].copy()
    valid  = subset.dropna(subset=["pred_type"]).copy()

    # Tách cả nhãn thật lẫn nhãn dự đoán thành 4 bit
    true_bits = (
        valid["true_type"]
        .apply(type_to_bits)
        .apply(pd.Series)
        .add_prefix("true_")
        .reset_index(drop=True)
    )

    pred_bits = (
        valid["pred_type"]
        .apply(type_to_bits)
        .apply(pd.Series)
        .add_prefix("pred_")
        .reset_index(drop=True)
    )

    valid = pd.concat(
        [
            valid.reset_index(drop=True),
            true_bits,
            pred_bits
        ],
        axis=1
    )
    coverage = len(valid) / len(subset)
    print(f"\n--- {strategy} | n_valid={len(valid)}/{len(subset)} (coverage {coverage:.1%}) ---")

    f1s, accs = [], []
    per_trait = {}
    for tr in TRAITS:
        yt = valid[f"true_{tr}"].astype(int)
        yp = valid[f"pred_{tr}"].astype(int)
        acc = accuracy_score(yt, yp)
        pre = precision_score(yt, yp, zero_division=0)
        rec = recall_score(yt, yp, zero_division=0)
        f1  = f1_score(yt, yp, zero_division=0)
        per_trait[tr] = (acc, pre, rec, f1)
        f1s.append(f1); accs.append(acc)
        print(f"   {tr}: Acc={acc:.4f}  P={pre:.4f}  R={rec:.4f}  F1={f1:.4f}")

    macro_f1  = sum(f1s) / len(f1s)
    macro_acc = sum(accs) / len(accs)

    # Exact match: đúng đồng thời cả 4 chiều (so loại đầy đủ)
    exact = (valid["true_type"].str.upper().str.strip()
             == valid["pred_type"].str.upper().str.strip()).mean()

    print(f"   --> Macro-Acc={macro_acc:.4f} | Macro-F1={macro_f1:.4f} | Exact-match(4 chiều)={exact:.4f}")

    row = {"strategy": strategy, "n_valid": len(valid),
           "coverage": round(coverage, 4),
           "macro_acc": round(macro_acc, 4),
           "macro_f1": round(macro_f1, 4),
           "exact_match": round(exact, 4)}
    for tr in TRAITS:
        a, p, r, f = per_trait[tr]
        row[f"{tr}_acc"] = round(a, 4)
        row[f"{tr}_f1"]  = round(f, 4)
    summary_rows.append(row)

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(OUTPUT_DIR / "summary_mbti_qwen.csv", index=False)
print("\nSaved summary_mbti_qwen.csv")
print(df_summary.to_string(index=False))
print("\nDONE!")

# ============================================================
# THỐNG KÊ BỔ SUNG — TỶ LỆ DỰ ĐOÁN "y" VÀ TUÂN THỦ ĐỊNH DẠNG
# ============================================================

print("\n" + "=" * 60)
print("TỶ LỆ DỰ ĐOÁN 'y' THEO TỪNG CHIỀU (phân phối biên)")
print("=" * 60)

bias_rows = []

for strategy in PROMPTS.keys():
    subset = final_df[final_df["prompt_strategy"] == strategy].copy()

    # --- Tuân thủ định dạng đầu ra ---
    n_total   = len(subset)
    n_valid   = subset["pred_type"].notna().sum()
    n_invalid = n_total - n_valid
    coverage  = n_valid / n_total

    print(f"\n--- {strategy} ---")
    print(f"  Tuân thủ định dạng: {n_valid}/{n_total} = {coverage:.1%}")
    print(f"  Không parse được  : {n_invalid} mẫu ({n_invalid/n_total:.1%})")

    # --- Tỷ lệ dự đoán "y" theo từng chiều ---
    valid = subset.dropna(subset=["pred_type"]).copy()

    pred_bits = (
        valid["pred_type"]
        .apply(type_to_bits)
        .apply(pd.Series)
        .add_prefix("pred_")
        .reset_index(drop=True)
    )
    valid = pd.concat([valid.reset_index(drop=True), pred_bits], axis=1)

    true_bits = (
        valid["true_type"]
        .apply(type_to_bits)
        .apply(pd.Series)
        .add_prefix("true_")
        .reset_index(drop=True)
    )
    valid = pd.concat([valid, true_bits], axis=1)

    row = {"strategy": strategy, "coverage": round(coverage, 4)}

    print(f"  {'Chiều':<8} {'Tỷ lệ pred=y':>14} {'Tỷ lệ true=y':>14}")
    for tr in TRAITS:
        pred_y_rate = valid[f"pred_{tr}"].mean()
        true_y_rate = valid[f"true_{tr}"].mean()
        print(f"  {tr:<8} {pred_y_rate:>13.1%} {true_y_rate:>13.1%}")
        row[f"{tr}_pred_y_rate"] = round(pred_y_rate, 4)
        row[f"{tr}_true_y_rate"] = round(true_y_rate, 4)

    bias_rows.append(row)

df_bias = pd.DataFrame(bias_rows)
df_bias.to_csv(OUTPUT_DIR / "bias_mbti_qwen.csv", index=False)
print("\nSaved bias_mbti_qwen.csv")
print(df_bias.to_string(index=False))

# ============================================================
# PHÂN TÍCH LOẠI MBTI ĐƯỢC DỰ ĐOÁN NHIỀU NHẤT
# ============================================================

print("\n" + "=" * 60)
print("PHÂN BỐ DỰ ĐOÁN THEO 16 LOẠI MBTI")
print("=" * 60)

for strategy in PROMPTS.keys():
    subset = final_df[final_df["prompt_strategy"] == strategy].copy()
    valid  = subset.dropna(subset=["pred_type"])
    print(f"\n--- {strategy} ---")
    dist = valid["pred_type"].value_counts(normalize=True).mul(100).round(2)
    print(dist.to_string())
    dist.to_csv(OUTPUT_DIR / f"pred_dist_mbti_{strategy}.csv", header=["percent"])