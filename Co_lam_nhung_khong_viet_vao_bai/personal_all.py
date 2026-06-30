"""
=============================================================
  prompting_mypersonality.py
  Nhận diện Big Five từ myPersonality Dataset (45 users)
  Models: GPT-4.1-mini | Mistral-Small-24B | Qwen-2.5-7B
  Nhãn: y/n (binary) so sánh với cEXT/cNEU/cAGR/cCON/cOPN
=============================================================
"""

import os
import re
import json
import time
import pandas as pd
from sklearn.metrics import f1_score, classification_report
from openai import OpenAI
from dotenv import load_dotenv

# ============================================================
# SETUP
# ============================================================

load_dotenv(dotenv_path="API.env")

# Client GPT
client_gpt = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Client OpenRouter (Mistral + Qwen)
client_or = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    default_headers={
        "HTTP-Referer": "https://github.com/datn",
        "X-Title": "Personality Recognition DATN"
    }
)

MODELS = {
    "gpt-4.1-mini":          {"client": "gpt",       "id": "gpt-4.1-mini"},
    "mistral-small-3.1-24b": {"client": "openrouter", "id": "mistralai/mistral-small-3.1-24b-instruct"},
    "qwen-2.5-7b":           {"client": "openrouter", "id": "qwen/qwen-2.5-7b-instruct"},
}

TEXT_COL  = "TEXT_CLEAN"
TRAITS    = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
MAX_CHARS = 3000  # ~700 từ

# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv("mypersonality_clean.csv")
df = df.dropna(subset=[TEXT_COL] + TRAITS).reset_index(drop=True)
df["sample_id"] = df.index
print(f"Loaded: {len(df)} users")
print(f"Label distribution:")
for t in TRAITS:
    vc = df[t].value_counts()
    print(f"  {t}: y={vc.get('y',0)}, n={vc.get('n',0)}")

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

EXAMPLE 3:
Text: "I carefully plan my schedule and follow through on every commitment I make."
Output: {{"cEXT": "n", "cNEU": "n", "cAGR": "y", "cCON": "y", "cOPN": "n"}}

TEXT TO ANALYZE:
\"\"\"{text}\"\"\"

Respond ONLY in this exact JSON format, no explanation:
{{"cEXT": "y", "cNEU": "n", "cAGR": "y", "cCON": "n", "cOPN": "y"}}"""


COT_PROMPT = """You are an expert psychologist specializing in personality assessment.
Analyze the text below and predict the Big Five personality traits (OCEAN).

Follow these steps:
1. Identify key linguistic signals (word choice, emotional tone, topics, social references)
2. Map each signal to the relevant Big Five dimension
3. Make a final prediction for each trait: 'y' (high) or 'n' (low)

Traits:
- cEXT: Extraversion
- cNEU: Neuroticism
- cAGR: Agreeableness
- cCON: Conscientiousness
- cOPN: Openness

TEXT:
\"\"\"{text}\"\"\"

Write your reasoning (2-3 sentences), then end with ONLY this JSON on the last line:
{{"cEXT": "y", "cNEU": "n", "cAGR": "y", "cCON": "n", "cOPN": "y"}}"""


PROMPTS = {
    "zero_shot": ZERO_SHOT_PROMPT,
    "few_shot":  FEW_SHOT_PROMPT,
    "cot":       COT_PROMPT,
}

# ============================================================
# HELPER: EXTRACT JSON ROBUST
# ============================================================

def extract_json(raw, prompt_name):
    """Parse JSON từ output, xử lý cả CoT có text trước JSON"""
    if prompt_name == "cot":
        lines = raw.split("\n")
        for line in reversed(lines):
            if "{" in line and "}" in line:
                candidate = line[line.find("{"):line.rfind("}")+1]
                try:
                    return json.loads(candidate)
                except:
                    continue
        raise json.JSONDecodeError("No JSON in CoT output", raw, 0)
    else:
        match = re.search(r'\{[\s\S]*?\}', raw)
        if match:
            return json.loads(match.group(0))
        raise json.JSONDecodeError("No JSON found", raw, 0)


def normalize_traits(parsed):
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

# ============================================================
# PREDICTION FUNCTION
# ============================================================

def predict_ocean(model_key, prompt_template, text, prompt_name, retries=4):
    text_cut = text[:MAX_CHARS]
    prompt   = prompt_template.format(text=text_cut)

    model_cfg     = MODELS[model_key]
    client        = client_gpt if model_cfg["client"] == "gpt" else client_or
    model_id      = model_cfg["id"]
    max_tok       = 800 if prompt_name == "cot" else 50
    use_json_mode = (prompt_name != "cot") and (model_cfg["client"] == "gpt")

    for attempt in range(retries):
        try:
            if prompt_name == "cot" and model_cfg["client"] == "openrouter":
                max_tok  = 1500
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert psychologist. "
                            "Analyze texts and predict Big Five personality traits. "
                            "Always end your response with a JSON object on the last line."
                        )
                    },
                    {"role": "user", "content": prompt}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are an expert psychologist. Follow instructions exactly."},
                    {"role": "user",   "content": prompt}
                ]

            kwargs = dict(
                model=model_id,
                temperature=0,
                max_tokens=max_tok,
                messages=messages
            )
            if use_json_mode:
                kwargs["response_format"] = {"type": "json_object"}

            resp   = client.chat.completions.create(**kwargs)
            finish = resp.choices[0].finish_reason if resp.choices else None
            if finish == "length":
                print(f"    Truncated (finish_reason=length), retrying with more tokens...")
                max_tok = min(max_tok * 2, 3000)
                time.sleep(2)
                continue
            # Check response không rỗng
            if (not resp.choices or
                resp.choices[0].message is None or
                resp.choices[0].message.content is None):
                print(f"    Empty response (attempt {attempt+1})")
                time.sleep(3)
                continue

            raw = resp.choices[0].message.content.strip()
            if not raw:
                print(f"    Empty content (attempt {attempt+1})")
                time.sleep(3)
                continue

            parsed = extract_json(raw, prompt_name)
            return normalize_traits(parsed)

        except json.JSONDecodeError:
            print(f"    JSON error (attempt {attempt+1})")
            time.sleep(2)

        except Exception as e:
            err = str(e).lower()
            if "429" in err or "rate_limit" in err or "quota" in err:
                wait = min(30 * (attempt + 1), 120)
                print(f"    Rate limit → wait {wait}s")
                time.sleep(wait)
            elif "404" in err:
                print(f"    Model not found: {model_id}")
                return {t: None for t in TRAITS}
            else:
                print(f"    Error: {e}")
                time.sleep(3)

    return {t: None for t in TRAITS}

# ============================================================
# RUN EXPERIMENT
# ============================================================

def run_experiment(model_key, prompt_name, prompt_template):
    print(f"\n{'='*55}")
    print(f"MODEL: {model_key} | PROMPT: {prompt_name} | {len(df)} samples")
    print(f"{'='*55}")

    results = []

    for _, row in df.iterrows():
        sid = row["sample_id"]
        print(f"  [{sid+1}/{len(df)}]", end=" ")

        pred = predict_ocean(model_key, prompt_template,
                             str(row[TEXT_COL]), prompt_name)

        true_vals = {t: row[t] for t in TRAITS}
        print({t: pred[t] for t in TRAITS})

        results.append({
            "sample_id":       sid,
            "model":           model_key,
            "prompt_strategy": prompt_name,
            **{f"true_{t}": true_vals[t] for t in TRAITS},
            **{f"pred_{t}": pred[t]       for t in TRAITS},
        })

        time.sleep(0.5)

    return pd.DataFrame(results)
# ============================================================
# RUN ALL MODELS × ALL PROMPTS
# ============================================================

os.makedirs("outputs_myp", exist_ok=True)
all_results = []

for model_key in MODELS:
    for prompt_name, prompt_template in PROMPTS.items():

        fname = f"outputs_myp/{model_key}_{prompt_name}.csv"

        # Nếu file đã có thì load lại, không chạy nữa
        if os.path.exists(fname):
            print(f"  Skipping {model_key} {prompt_name} — file exists")
            all_results.append(pd.read_csv(fname))
            continue

        df_result = run_experiment(model_key, prompt_name, prompt_template)
        all_results.append(df_result)
        df_result.to_csv(fname, index=False)
        print(f"  Saved: {fname}")

# Sau vòng lặp for model_key...
# Thêm đoạn này trước EVALUATION

import glob

# Load tất cả file kết quả
all_files = glob.glob("outputs_myp/*.csv")
all_dfs   = []

for f in all_files:
    if "summary" not in f and "all" not in f:
        try:
            all_dfs.append(pd.read_csv(f))
        except:
            pass

if all_dfs:
    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df = final_df.drop_duplicates(
        subset=["sample_id", "model", "prompt_strategy"]
    )
    final_df.to_csv("outputs_myp/predictions_mypersonality_all.csv", index=False)
    print(f"Total rows: {len(final_df)}")
else:
    print("Không tìm thấy file kết quả!")
    final_df = pd.DataFrame()
# ============================================================
# EVALUATION
# ============================================================
print("\n" + "="*60)
print("EVALUATION RESULTS — myPersonality Dataset")
print("="*60)

# Debug: xem có gì trong final_df
print(f"Total rows: {len(final_df)}")
print(f"Models found: {final_df['model'].unique().tolist()}")
print(f"Strategies found: {final_df['prompt_strategy'].unique().tolist()}")

summary_rows = []

# Dùng unique values thực tế thay vì MODELS/PROMPTS
for model_key in final_df["model"].unique():
    for strategy in final_df["prompt_strategy"].unique():
        subset = final_df[
            (final_df["model"] == model_key) &
            (final_df["prompt_strategy"] == strategy)
        ].copy()

        if len(subset) == 0:
            continue

        print(f"\n[{model_key}] [{strategy}] ({len(subset)} samples)")

        trait_accs = []
        trait_f1s  = []

        for t in TRAITS:
            true_col = f"true_{t}"
            pred_col = f"pred_{t}"

            if true_col not in subset.columns or pred_col not in subset.columns:
                print(f"  {t}: column missing")
                continue

            true_vals = subset[true_col].map({"y": 1, "n": 0})
            pred_vals = subset[pred_col].map({"y": 1, "n": 0})
            valid     = true_vals.notna() & pred_vals.notna()

            if valid.sum() > 0:
                acc = (true_vals[valid] == pred_vals[valid]).mean()
                f1  = f1_score(true_vals[valid], pred_vals[valid],
                               average="macro", zero_division=0)
                trait_accs.append(acc)
                trait_f1s.append(f1)
                print(f"  {t}: acc={acc:.2%}  f1={f1:.3f}  ({valid.sum()} valid)")
            else:
                print(f"  {t}: N/A (no valid predictions)")

        avg_acc = sum(trait_accs)/len(trait_accs) if trait_accs else 0
        avg_f1  = sum(trait_f1s)/len(trait_f1s)   if trait_f1s  else 0
        print(f"  >>> Avg Accuracy: {avg_acc:.2%} | Avg F1-macro: {avg_f1:.3f}")

        summary_rows.append({
            "model":        model_key,
            "strategy":     strategy,
            "n_samples":    len(subset),
            "avg_accuracy": round(avg_acc, 4),
            "avg_f1_macro": round(avg_f1, 4),
        })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv("outputs_myp/summary_mypersonality.csv", index=False)
print("\n" + "="*60)
print("SUMMARY TABLE")
print("="*60)
print(df_summary.to_string(index=False))
print("\nDONE!")