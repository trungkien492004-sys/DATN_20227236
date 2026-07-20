import os
import json
import time
import re
import pandas as pd
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# PATHS (tương đối theo vị trí script -> clone về vẫn chạy)
# ============================================================

BASE_DIR   = Path(__file__).resolve().parent                       # Prompt/Code_Prompt
OUTPUT_DIR = BASE_DIR.parent / "Output_prompt" / "essay_mistral"   # Prompt/Output_prompt/essay_mistral
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_FILE = "essays_clean.csv"

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


MODEL = "mistralai/mistral-small-3.1-24b-instruct"


df = pd.read_csv(DATA_PATH)
df = df[["#AUTHID", "TEXT_CLEAN", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]].dropna()
df = df.rename(columns={"TEXT_CLEAN": "text"})
TRAITS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]

# File merge cuối (dùng để resume nếu không có checkpoint riêng)
MODEL_SLUG  = MODEL.replace("/", "-").replace(":", "-")
MERGED_FILE = OUTPUT_DIR / f"predictions_{MODEL_SLUG}.csv"

# ============================================================
# PROMPT TEMPLATES
# ============================================================

ZERO_SHOT_PROMPT = """You are an expert psychologist.
Analyze the text below and predict the Big Five personality traits (OCEAN).

For each trait respond ONLY 'y' (high) or 'n' (low):
- cEXT: Extraversion
- cNEU: Neuroticism
- cAGR: Agreeableness
- cCON: Conscientiousness
- cOPN: Openness

TEXT:
\"\"\"{text}\"\"\"

Respond ONLY in this exact JSON format, no explanation:
{{"cEXT": "y", "cNEU": "n", "cAGR": "y", "cCON": "n", "cOPN": "y"}}"""

FEW_SHOT_PROMPT = """You are an expert psychologist specializing in personality assessment.
Analyze the text below and predict the Big Five personality traits (OCEAN).

For each trait respond ONLY 'y' (high) or 'n' (low):
- cEXT: Extraversion - sociable, talkative, energetic
- cNEU: Neuroticism - anxious, emotionally unstable, worrying
- cAGR: Agreeableness - cooperative, trusting, kind
- cCON: Conscientiousness - organized, disciplined, goal-oriented
- cOPN: Openness - curious, creative, open to new experiences

EXAMPLE 1:
Text: "I love going to parties and meeting new people. Being around others gives me so much energy!"
Output: {{"cEXT": "y", "cNEU": "n", "cAGR": "y", "cCON": "n", "cOPN": "y"}}

EXAMPLE 2:
Text: "I often worry about things going wrong. I prefer staying home alone rather than socializing."
Output: {{"cEXT": "n", "cNEU": "y", "cAGR": "n", "cCON": "y", "cOPN": "n"}}

TEXT TO ANALYZE:
\"\"\"{text}\"\"\"

Respond ONLY in this exact JSON format, no explanation:
{{"cEXT": "y", "cNEU": "n", "cAGR": "y", "cCON": "n", "cOPN": "y"}}"""

COT_PROMPT = """You are an expert psychologist specializing in personality assessment.
Analyze the text below and predict the Big Five personality traits (OCEAN).

Follow these steps:
1. Identify key linguistic signals in the text (word choice, topics, emotional tone)
2. Map each signal to the relevant Big Five dimension
3. Make a final prediction for each trait: 'y' (high) or 'n' (low)

Traits to predict:
- cEXT: Extraversion
- cNEU: Neuroticism
- cAGR: Agreeableness
- cCON: Conscientiousness
- cOPN: Openness

TEXT:
\"\"\"{text}\"\"\"

First briefly reason (2-3 sentences), then end with ONLY this JSON:
{{"cEXT": "y", "cNEU": "n", "cAGR": "y", "cCON": "n", "cOPN": "y"}}"""

PROMPTS = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot":  FEW_SHOT_PROMPT,
    "cot":       COT_PROMPT,
}

# ============================================================
# HÀM TRÍCH XUẤT JSON ROBUST
# ============================================================

def extract_json(raw_output: str):
    raw = raw_output.strip()

    # Cách 1: Tìm khối JSON lớn nhất
    match = re.search(r'\{[\s\S]*\}', raw)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    # Cách 2: Loại bỏ markdown
    cleaned = re.sub(r'```(?:json)?\s*', '', raw)
    cleaned = re.sub(r'\s*```', '', cleaned)

    try:
        return json.loads(cleaned)
    except:
        pass

    # Cách 3: Cắt thủ công từ { đến }
    try:
        start = cleaned.find('{')
        end = cleaned.rfind('}') + 1
        if start != -1 and end > start:
            return json.loads(cleaned[start:end])
    except:
        pass

    raise json.JSONDecodeError("Cannot extract JSON", raw, 0)


# ============================================================
# PREDICTION FUNCTION — Retry chỉ 3 lần
# ============================================================

def predict_ocean(prompt_template, text):
    prompt = prompt_template.format(text=text)
    max_attempts = 3
    attempt = 0

    while attempt < max_attempts:
        try:
            response = client.chat.completions.create(
                model=MODEL,
                max_tokens=500,
                temperature=0.35,
                messages=[
                    {"role": "system", "content": "You are an expert psychologist. Respond ONLY with valid JSON. No explanation, no markdown, no extra text."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw = response.choices[0].message.content.strip()
            parsed = extract_json(raw)

            result = {}
            for t in TRAITS:
                val = parsed.get(t)
                if isinstance(val, str):
                    v = val.lower().strip()
                    if v in ["y", "yes", "high", "1", "true"]:
                        result[t] = "y"
                    elif v in ["n", "no", "low", "0", "false"]:
                        result[t] = "n"
                    else:
                        result[t] = None
                elif isinstance(val, bool):
                    result[t] = "y" if val else "n"
                else:
                    result[t] = None

            return result

        except json.JSONDecodeError:
            print(f"  [Attempt {attempt+1}/{max_attempts}] JSON parse error → retrying")

        except Exception as e:
            err_str = str(e).lower()
            if "429" in err_str or "rate limit" in err_str or "quota" in err_str:
                wait = min(30 * (attempt + 1), 120)
                print(f"  [Attempt {attempt+1}/{max_attempts}] Rate limit, waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"  [Attempt {attempt+1}/{max_attempts}] Error: {type(e).__name__} → retrying")
                time.sleep(2)

        attempt += 1
        time.sleep(1.5)

    print(f"  ❌ Failed after {max_attempts} attempts")
    return {t: None for t in TRAITS}


# ============================================================
# MAIN LOOP với CHECKPOINT
# ============================================================

def run_experiment(df_input, prompt_name, prompt_template):
    print(f"\n{'='*50}")
    print(f"Running: {prompt_name.upper()} | {MODEL} | {len(df_input)} samples")
    print(f"{'='*50}")

    ckpt_file  = OUTPUT_DIR / f"checkpoint_{prompt_name}_{MODEL_SLUG}.csv"
    final_file = OUTPUT_DIR / f"predictions_{prompt_name}_{MODEL_SLUG}.csv"

    all_ids = set(df_input["#AUTHID"].tolist())

    # --- Check file cuối trước: đủ hết thì bỏ qua ---
    if os.path.exists(final_file):
        done_df = pd.read_csv(final_file)
        if "authid" in done_df.columns and all_ids.issubset(set(done_df["authid"].tolist())):
            print(f"  Skip | {prompt_name} | đã xong ({len(done_df)} mẫu), dùng file cuối")
            return done_df
        print(f"  File cuối {prompt_name} thiếu mẫu -> resume tiếp")

    # --- Chưa xong: nạp checkpoint / file merge cũ ---
    if os.path.exists(ckpt_file):
        done_df  = pd.read_csv(ckpt_file)
        done_ids = set(done_df["authid"].tolist())
        results  = done_df.to_dict("records")
        print(f"  Resume checkpoint: {len(done_ids)} done, {len(df_input)-len(done_ids)} remaining")

    elif os.path.exists(MERGED_FILE):
        merged_df = pd.read_csv(MERGED_FILE)
        strat_df  = merged_df[merged_df["prompt_strategy"] == prompt_name]
        done_ids  = set(strat_df["authid"].tolist())
        results   = strat_df.to_dict("records")
        print(f"  Resume từ file merge: {len(done_ids)} done, {len(df_input)-len(done_ids)} remaining")

    else:
        done_ids = set()
        results  = []

    for i, (_, row) in enumerate(df_input.iterrows()):
        if row["#AUTHID"] in done_ids:
            continue

        print(f"  [{i+1}/{len(df_input)}] {row['#AUTHID']}", end=" ")
        pred = predict_ocean(prompt_template, row["text"])
        print({t: pred[t] for t in TRAITS})

        results.append({
            "authid":          row["#AUTHID"],
            "model":           MODEL,
            "prompt_strategy": prompt_name,
            **{f"true_{t}": row[t] for t in TRAITS},
            **{f"pred_{t}": pred[t] for t in TRAITS},
        })

        if len(results) % 100 == 0:
            pd.DataFrame(results).to_csv(ckpt_file, index=False)
            print(f"  💾 Checkpoint saved ({len(results)} done)")

        time.sleep(0.5)

    df_result = pd.DataFrame(results)
    df_result.to_csv(ckpt_file, index=False)
    df_result.to_csv(final_file, index=False)
    return df_result


# ============================================================
# RUN & SAVE
# ============================================================

all_results = []

for prompt_name, prompt_template in PROMPTS.items():
    df_result = run_experiment(df, prompt_name, prompt_template)   # dùng df thay vì df_sample
    all_results.append(df_result)

final_df = pd.concat(all_results, ignore_index=True)
final_df.to_csv(MERGED_FILE, index=False)
print(f"\nDone! Saved to {MERGED_FILE}")
print(final_df.groupby("prompt_strategy").size())

# ============================================================
# ACCURACY CHECK
# ============================================================

print("\n--- Accuracy per trait per strategy ---")
for strategy in PROMPTS.keys():
    subset = final_df[final_df["prompt_strategy"] == strategy].copy()
    print(f"\n{strategy}:")
    for t in TRAITS:
        true_vals = subset[f"true_{t}"].map({"y": 1, "n": 0})
        pred_vals = subset[f"pred_{t}"].map({"y": 1, "n": 0})
        valid = true_vals.notna() & pred_vals.notna()
        if valid.sum() > 0:
            acc = (true_vals[valid] == pred_vals[valid]).mean()
            print(f"  {t}: {acc:.2%} ({valid.sum()} samples)")