import os
import re
import json
import asyncio
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR.parent / "Output_prompt" / "essay_gpt"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MERGED_FILE = OUTPUT_DIR / "predictions_gpt.csv"

# ============================================================
# SETUP
# ============================================================

load_dotenv(BASE_DIR / "API.env")

client = AsyncOpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

# ============================================================
# LOAD DATA
# ============================================================

DATA_FILE = "essays_clean.csv"

# Tự động tìm file data trong project (tương đối theo vị trí script)
# -> clone về máy khác vẫn chạy, không cần sửa path
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

df = pd.read_csv(DATA_PATH, encoding="latin-1")

df = df[
    ["#AUTHID", "TEXT_CLEAN", "cEXT", "cNEU", "cAGR", "cCON", "cOPN"]
].dropna()

df = df.rename(columns={"#AUTHID": "authid", "TEXT_CLEAN": "text"})

TRAITS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]

# ============================================================
# PROMPTS
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
    "few_shot": FEW_SHOT_PROMPT,
    "cot": COT_PROMPT
}

# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "gpt-4.1-mini"

CONCURRENCY = 5

CHECKPOINT_EVERY = 50

# ============================================================
# JSON PARSER
# ============================================================

def extract_json(raw_text):

    match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)

    if not match:
        raise json.JSONDecodeError(
            "No JSON found",
            raw_text,
            0
        )

    return json.loads(match.group(0))

# ============================================================
# NORMALIZE OUTPUT
# ============================================================

def normalize_prediction(parsed):

    result = {}

    for trait in TRAITS:

        value = parsed.get(trait)

        if value in ["y", "n"]:
            result[trait] = value
        else:
            result[trait] = None

    return result

# ============================================================
# GPT PREDICTION
# ============================================================

async def predict_ocean(prompt_template, text, semaphore, prompt_name, retries=5):
    prompt = prompt_template.format(text=text)

    # Tắt JSON mode cho COT
    use_json_mode = prompt_name != "cot"

    async with semaphore:
        for attempt in range(retries):
            try:
                response = await asyncio.wait_for(
                    client.chat.completions.create(
                        model=MODEL_NAME,
                        temperature=0.35,
                        max_tokens=500,           # tăng cho COT
                        response_format={"type": "json_object"} if use_json_mode else None,
                        messages=[{"role": "user", "content": prompt}]
                    ),
                    timeout=45.0
                )

                raw = response.choices[0].message.content.strip()
                parsed = extract_json(raw)
                return normalize_prediction(parsed)

            except asyncio.TimeoutError:
                print(f"⏰ Timeout | attempt {attempt+1}")
            except json.JSONDecodeError as e:
                print(f"❌ JSON parse failed | attempt {attempt+1} | Strategy: {prompt_name}")
                # In ra raw để debug
                if attempt == retries - 1:
                    print(f"Raw output: {raw[:300]}...")
            except Exception as e:
                err = str(e).lower()
                if "429" in err or "rate_limit" in err:
                    wait_time = min(2 ** attempt, 25)
                    print(f"Rate limit → wait {wait_time}s")
                    await asyncio.sleep(wait_time)
                else:
                    print(f"Error: {e}")
                    break

    return {trait: None for trait in TRAITS}

# ============================================================
# RUN EXPERIMENT
# ============================================================

async def run_experiment(
    df_input,
    prompt_name,
    prompt_template
):

    checkpoint_file = (
        OUTPUT_DIR / f"checkpoint_{prompt_name}.csv"
    )

    final_file = (
        OUTPUT_DIR / f"predictions_{prompt_name}.csv"
    )

    all_ids = set(df_input["authid"].tolist())

    # ========================================================
    # CHECK FINAL FILE FIRST
    # -> đủ hết rồi thì bỏ qua, không đụng checkpoint
    # ========================================================

    if os.path.exists(final_file):

        done_df = pd.read_csv(final_file)

        if all_ids.issubset(set(done_df["authid"].tolist())):

            print(
                f"Skip | {prompt_name} | "
                f"đã xong ({len(done_df)} mẫu), dùng file cuối"
            )

            return done_df

        print(
            f"File cuối {prompt_name} thiếu mẫu -> resume tiếp"
        )

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    if os.path.exists(checkpoint_file):

        checkpoint_df = pd.read_csv(
            checkpoint_file
        )

        completed_ids = set(
            checkpoint_df["authid"].tolist()
        )
        results = checkpoint_df.to_dict(
            "records"
        )

        print(
            f"Resume checkpoint | "
            f"{prompt_name} | "
            f"{len(completed_ids)} completed"
        )

    elif os.path.exists(MERGED_FILE):

        # Chưa có checkpoint riêng -> lấy từ file merge cũ (predictions_gpt.csv)
        merged_df = pd.read_csv(MERGED_FILE)

        strat_df = merged_df[
            merged_df["prompt_strategy"] == prompt_name
        ]

        completed_ids = set(strat_df["authid"].tolist())
        results = strat_df.to_dict("records")

        print(
            f"Resume từ file merge | "
            f"{prompt_name} | "
            f"{len(completed_ids)} completed"
        )

    else:

        completed_ids = set()

        results = []

    # ========================================================
    # FILTER REMAINING
    # ========================================================

    remaining_rows = []

    for row in df_input.to_dict("records"):

        if row["authid"] not in completed_ids:
            remaining_rows.append(row)

    print(
        f"{prompt_name} | "
        f"Remaining samples: "
        f"{len(remaining_rows)}"
    )

    # ========================================================
    # CONCURRENCY CONTROL
    # ========================================================

    semaphore = asyncio.Semaphore(
        CONCURRENCY
    )

    # ========================================================
    # WORKER
    # ========================================================

    async def worker(row):

        prediction = await predict_ocean(
            prompt_template=prompt_template,
            text=row["text"],
            semaphore=semaphore,
            prompt_name=prompt_name
        )

        return {

            "authid": row["authid"],

            "prompt_strategy": prompt_name,

            "true_cEXT": row["cEXT"],
            "true_cNEU": row["cNEU"],
            "true_cAGR": row["cAGR"],
            "true_cCON": row["cCON"],
            "true_cOPN": row["cOPN"],

            "pred_cEXT": prediction["cEXT"],
            "pred_cNEU": prediction["cNEU"],
            "pred_cAGR": prediction["cAGR"],
            "pred_cCON": prediction["cCON"],
            "pred_cOPN": prediction["cOPN"],
        }

    # ========================================================
    # CREATE TASKS
    # ========================================================

    tasks = [
        asyncio.create_task(worker(row))
        for row in remaining_rows
    ]

    completed_since_save = 0

    # ========================================================
    # PROCESS RESULTS
    # ========================================================

    for idx, task in enumerate(
        asyncio.as_completed(tasks),
        start=1
    ):

        result = await task

        results.append(result)

        completed_since_save += 1

        print(
            f"[{idx}/{len(tasks)}] "
            f"{result['authid']} "
            f"{ {t: result[f'pred_{t}'] for t in TRAITS} }"
        )

        # ====================================================
        # SAVE CHECKPOINT
        # ====================================================

        if completed_since_save >= CHECKPOINT_EVERY:

            pd.DataFrame(results).to_csv(
                checkpoint_file,
                index=False
            )

            print(
                f"Checkpoint saved | "
                f"{prompt_name} | "
                f"{len(results)} completed"
            )

            completed_since_save = 0

    # ========================================================
    # FINAL SAVE
    # ========================================================

    final_df = pd.DataFrame(results)

    final_df.to_csv(
        checkpoint_file,
        index=False
    )

    final_df.to_csv(
        final_file,
        index=False
    )

    return final_df

# ============================================================
# ACCURACY
# ============================================================

def calculate_accuracy(df_result):

    print("\n==============================")
    print("ACCURACY RESULTS")
    print("==============================")

    for strategy in PROMPTS.keys():

        subset = df_result[
            df_result["prompt_strategy"]
            == strategy
        ]

        print(f"\n{strategy}")

        for trait in TRAITS:

            true_values = subset[
                f"true_{trait}"
            ].map({"y": 1, "n": 0})

            pred_values = subset[
                f"pred_{trait}"
            ].map({"y": 1, "n": 0})

            valid_mask = (
                true_values.notna()
                &
                pred_values.notna()
            )

            valid_count = valid_mask.sum()

            if valid_count > 0:

                accuracy = (
                    true_values[valid_mask]
                    ==
                    pred_values[valid_mask]
                ).mean()

                print(
                    f"{trait}: "
                    f"{accuracy:.2%} "
                    f"({valid_count} samples)"
                )

# ============================================================
# MAIN
# ============================================================

async def main():

    print("=" * 60)
    print("STARTING BIG FIVE PREDICTION")
    print("=" * 60)

    all_results = []

    for prompt_name, prompt_template in PROMPTS.items():

        print("\n" + "=" * 60)
        print(f"RUNNING: {prompt_name}")
        print("=" * 60)

        result_df = await run_experiment(
            df_input=df,
            prompt_name=prompt_name,
            prompt_template=prompt_template
        )

        all_results.append(result_df)

    # ========================================================
    # MERGE ALL RESULTS
    # ========================================================

    final_df = pd.concat(
        all_results,
        ignore_index=True
    )

    final_df.to_csv(
        OUTPUT_DIR / "predictions_gpt.csv",
        index=False
    )
    print("\nSaved predictions_gpt.csv")

    calculate_accuracy(final_df)

    print("\nDONE!")

# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    asyncio.run(main())