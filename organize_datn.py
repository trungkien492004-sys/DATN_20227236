import os
import shutil

def clean_and_organize():
    base_dir = r"C:\Users\Kien\Downloads\DATA DATN"
    fine_tune_dir = os.path.join(base_dir, "Fine_tune")
    essay_dir = os.path.join(fine_tune_dir, "Essay")
    mbti_dir = os.path.join(fine_tune_dir, "MBTI")
    mbiti_khong_bao_cao_dir = os.path.join(fine_tune_dir, "Mbiti_khong_ghi_vao_bao_cao")

    print("=== 1. Organizing Essay Directory ===")
    essay_targets = ["mistral_sft", "mistral_ift", "qwen_sft", "qwen_ift"]
    for target in essay_targets:
        os.makedirs(os.path.join(essay_dir, target), exist_ok=True)
        print(f"Created/verified folder: Essay/{target}")

    # Mapping of Essay files to new locations
    essay_mapping = {
        # Mistral SFT
        os.path.join(essay_dir, "Mistral", "predictions_mistral_sft_essays.csv"): 
            os.path.join(essay_dir, "mistral_sft", "predictions_mistral_sft.csv"),
        os.path.join(essay_dir, "Mistral", "summary_mistral_sft_essays.csv"): 
            os.path.join(essay_dir, "mistral_sft", "summary_mistral_sft.csv"),
        os.path.join(essay_dir, "Mistral", "training_logs_mistral_sft_essays.csv"): 
            os.path.join(essay_dir, "mistral_sft", "training_logs_mistral_sft.csv"),
        os.path.join(essay_dir, "Mistral", "test_essays_mistral_ sft.csv"): 
            os.path.join(essay_dir, "mistral_sft", "test_essays_mistral_sft.csv"),
        os.path.join(essay_dir, "Mistral_SFT", "ESAY MISTRAL SFT.txt"): 
            os.path.join(essay_dir, "mistral_sft", "essay_mistral_sft.txt"),
        os.path.join(essay_dir, "Mistral_SFT", "essays_mistral_sft_cm.png"): 
            os.path.join(essay_dir, "mistral_sft", "essays_mistral_sft_cm.png"),
        os.path.join(essay_dir, "Mistral_SFT", "essays_mistral_sft_loss.png"): 
            os.path.join(essay_dir, "mistral_sft", "essays_mistral_sft_loss.png"),

        # Mistral IFT
        os.path.join(essay_dir, "Mistral", "predictions_mistral_ift_essays.csv"): 
            os.path.join(essay_dir, "mistral_ift", "predictions_mistral_ift.csv"),
        os.path.join(essay_dir, "Mistral", "summary_mistral_ift_essays.csv"): 
            os.path.join(essay_dir, "mistral_ift", "summary_mistral_ift.csv"),
        os.path.join(essay_dir, "Mistral", "training_logs_mistral_ift_essays.csv"): 
            os.path.join(essay_dir, "mistral_ift", "training_logs_mistral_ift.csv"),
        os.path.join(essay_dir, "Mistral", "test_essays_fixed (1) ift mistral.csv"): 
            os.path.join(essay_dir, "mistral_ift", "test_essays_mistral_ift.csv"),
        os.path.join(essay_dir, "Mistral_IFT", "ESAY MISTRAL IFT.txt"): 
            os.path.join(essay_dir, "mistral_ift", "essay_mistral_ift.txt"),
        os.path.join(essay_dir, "Mistral_IFT", "essays_mistral_ift_cm.png"): 
            os.path.join(essay_dir, "mistral_ift", "essays_mistral_ift_cm.png"),
        os.path.join(essay_dir, "Mistral_IFT", "essays_mistral_ift_loss.png"): 
            os.path.join(essay_dir, "mistral_ift", "essays_mistral_ift_loss.png"),

        # Qwen SFT
        os.path.join(essay_dir, "Qwen", "predictions_sft_essays.csv"): 
            os.path.join(essay_dir, "qwen_sft", "predictions_qwen_sft.csv"),
        os.path.join(essay_dir, "Qwen", "summary_sft_essays (1).csv"): 
            os.path.join(essay_dir, "qwen_sft", "summary_qwen_sft.csv"),
        os.path.join(essay_dir, "Qwen", "training_logs (1).csv"): 
            os.path.join(essay_dir, "qwen_sft", "training_logs_qwen_sft.csv"),
        os.path.join(essay_dir, "Qwen", "test_essays.csv"): 
            os.path.join(essay_dir, "qwen_sft", "test_essays_qwen_sft.csv"),
        os.path.join(essay_dir, "Qwen_SFT", "essay qwen SFT.txt"): 
            os.path.join(essay_dir, "qwen_sft", "essay_qwen_sft.txt"),
        os.path.join(essay_dir, "Qwen_SFT", "essays_qwen_sft_cm.png"): 
            os.path.join(essay_dir, "qwen_sft", "essays_qwen_sft_cm.png"),
        os.path.join(essay_dir, "Qwen_SFT", "essays_qwen_sft_loss.png"): 
            os.path.join(essay_dir, "qwen_sft", "essays_qwen_sft_loss.png"),

        # Qwen IFT
        os.path.join(essay_dir, "Qwen", "predictions_qwen_ift_essays.csv"): 
            os.path.join(essay_dir, "qwen_ift", "predictions_qwen_ift.csv"),
        os.path.join(essay_dir, "Qwen", "summary_qwen_ift_essays.csv"): 
            os.path.join(essay_dir, "qwen_ift", "summary_qwen_ift.csv"),
        os.path.join(essay_dir, "Qwen", "training_logs _qwen_ift.csv"): 
            os.path.join(essay_dir, "qwen_ift", "training_logs_qwen_ift.csv"),
        os.path.join(essay_dir, "Qwen", "test_essays _qwen_ift.csv"): 
            os.path.join(essay_dir, "qwen_ift", "test_essays_qwen_ift.csv"),
        os.path.join(essay_dir, "Qwen_IFT", "qwen IFT essay.txt"): 
            os.path.join(essay_dir, "qwen_ift", "essay_qwen_ift.txt"),
        os.path.join(essay_dir, "Qwen_IFT", "essays_qwen_ift_cm.png"): 
            os.path.join(essay_dir, "qwen_ift", "essays_qwen_ift_cm.png"),
        os.path.join(essay_dir, "Qwen_IFT", "essays_qwen_ift_loss.png"): 
            os.path.join(essay_dir, "qwen_ift", "essays_qwen_ift_loss.png"),
    }

    for src, dst in essay_mapping.items():
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved: {os.path.basename(src)} -> {os.path.relpath(dst, base_dir)}")

    # Clean up empty old folders in Essay
    old_essay_folders = ["Mistral", "Mistral_SFT", "Mistral_IFT", "Qwen", "Qwen_SFT", "Qwen_IFT"]
    for folder in old_essay_folders:
        folder_path = os.path.join(essay_dir, folder)
        if os.path.exists(folder_path):
            remaining_files = os.listdir(folder_path)
            if not remaining_files or remaining_files == ["desktop.ini"]:
                try:
                    shutil.rmtree(folder_path)
                    print(f"Removed old folder: Essay/{folder}")
                except Exception as e:
                    print(f"Error removing folder {folder}: {e}")

    print("\n=== 2. Handling Misplaced Files in MBTI ===")
    misplaced_file = os.path.join(mbti_dir, "mistral_sft", "qwen IFT essay.txt")
    correct_file_path = os.path.join(essay_dir, "qwen_ift", "essay_qwen_ift.txt")
    
    if os.path.exists(misplaced_file):
        if os.path.exists(correct_file_path):
            misplaced_size = os.path.getsize(misplaced_file)
            correct_size = os.path.getsize(correct_file_path)
            if abs(misplaced_size - correct_size) < 500:
                os.remove(misplaced_file)
                print("Removed duplicate misplaced file: MBTI/mistral_sft/qwen IFT essay.txt")
            else:
                backup_path = os.path.join(essay_dir, "qwen_ift", "essay_qwen_ift_backup.txt")
                shutil.move(misplaced_file, backup_path)
                print(f"Moved different misplaced file to: {os.path.relpath(backup_path, base_dir)}")
        else:
            shutil.move(misplaced_file, correct_file_path)
            print(f"Moved misplaced file to: {os.path.relpath(correct_file_path, base_dir)}")

    print("\n=== 3. Fixing Double Extensions in Mbiti_khong_ghi_vao_bao_cao ===")
    for root, dirs, files in os.walk(mbiti_khong_bao_cao_dir):
        for file in files:
            if file.endswith(".png.png"):
                old_path = os.path.join(root, file)
                new_path = os.path.join(root, file[:-4])
                shutil.move(old_path, new_path)
                print(f"Fixed double extension: {os.path.relpath(old_path, base_dir)} -> {os.path.basename(new_path)}")

    # Call organize_prompt logic
    organize_prompt(base_dir)

    print("\n=== 4. Cleaning desktop.ini files ===")
    for root, dirs, files in os.walk(base_dir):
        for file in files:
            if file.lower() == "desktop.ini":
                ini_path = os.path.join(root, file)
                try:
                    os.remove(ini_path)
                    print(f"Removed desktop.ini: {os.path.relpath(ini_path, base_dir)}")
                except Exception as e:
                    print(f"Failed to remove desktop.ini at {os.path.relpath(ini_path, base_dir)}: {e}")

    print("\n=== Clean and Organize Finished successfully ===")


def organize_prompt(base_dir):
    print("\n=== 5. Organizing Prompt Directory ===")
    prompt_dir = os.path.join(base_dir, "Prompt")
    code_prompt_dir = os.path.join(prompt_dir, "Code_Prompt")
    output_prompt_dir = os.path.join(prompt_dir, "Output_prompt")

    # A. Organize Code_Prompt (Standardizing filenames to snake_case and lowercase)
    print("--- 5a. Standardizing Code_Prompt Files ---")
    code_renames = {
        "GPT essay.py": "gpt_essay.py",
        "GPT_mbiti.py": "gpt_mbti.py",
        "mistral essay.py": "mistral_essay.py",
        "mistral_mbiti.py": "mistral_mbti.py",
        "qwen essay.py": "qwen_essay.py",
        "qwen_mbiti.py": "qwen_mbti.py",
    }
    
    for old_name, new_name in code_renames.items():
        old_path = os.path.join(code_prompt_dir, old_name)
        new_path = os.path.join(code_prompt_dir, new_name)
        if os.path.exists(old_path):
            os.rename(old_path, new_path)
            print(f"Renamed code: {old_name} -> {new_name}")
        else:
            print(f"Skipped (code not found): {old_name}")

    # B. Organize Output_prompt (Moving scattered outputs, renaming folders case-sensitively)
    print("--- 5b. Standardizing Output_prompt Directory & Files ---")
    
    # Rename target directories case-sensitively on Windows
    mbti_folders = ["MBTI_gpt", "MBTI_mistral", "MBTI_qwen", "MBTI_full_nhung_loi"]
    for folder in mbti_folders:
        src = os.path.join(output_prompt_dir, folder)
        if os.path.exists(src):
            tmp = os.path.join(output_prompt_dir, folder + "_tmp")
            dst = os.path.join(output_prompt_dir, folder.lower())
            os.rename(src, tmp)
            os.rename(tmp, dst)
            print(f"Renamed output folder: {folder} -> {folder.lower()}")

    # Ensure lowercase directories are created if not present
    for key in ["gpt", "mistral", "qwen"]:
        os.makedirs(os.path.join(output_prompt_dir, f"mbti_{key}"), exist_ok=True)

    # Move scattered CSV outputs to correct mbti_* subdirectories
    output_moves = {
        # GPT
        os.path.join(output_prompt_dir, "predictions_full_gpt.csv"): 
            os.path.join(output_prompt_dir, "mbti_gpt", "predictions_mbti_gpt.csv"),
        os.path.join(output_prompt_dir, "summary_mbti_gpt_new.csv"): 
            os.path.join(output_prompt_dir, "mbti_gpt", "summary_mbti_gpt.csv"),
        
        # Mistral
        os.path.join(output_prompt_dir, "predictions_full_mistral.csv"): 
            os.path.join(output_prompt_dir, "mbti_mistral", "predictions_mbti_mistral.csv"),
        os.path.join(output_prompt_dir, "summary_mbti_mistral_new.csv"): 
            os.path.join(output_prompt_dir, "mbti_mistral", "summary_mbti_mistral.csv"),
        
        # Qwen
        os.path.join(output_prompt_dir, "predictions_full_qwen.csv"): 
            os.path.join(output_prompt_dir, "mbti_qwen", "predictions_mbti_qwen.csv"),
        os.path.join(output_prompt_dir, "summary_mbti_qwen_new.csv"): 
            os.path.join(output_prompt_dir, "mbti_qwen", "summary_mbti_qwen.csv"),
    }

    for src, dst in output_moves.items():
        if os.path.exists(src):
            shutil.move(src, dst)
            print(f"Moved output: {os.path.basename(src)} -> {os.path.relpath(dst, base_dir)}")
        else:
            print(f"Skipped (output not found): {os.path.basename(src)}")

if __name__ == "__main__":
    clean_and_organize()
