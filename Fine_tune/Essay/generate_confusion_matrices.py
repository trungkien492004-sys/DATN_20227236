"""
Generate confusion matrix heatmaps AND loss curves for Essays dataset (4 models).
Requires: pip install matplotlib numpy pandas
Run: python generate_confusion_matrices.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import os

BASE = os.path.dirname(os.path.abspath(__file__))

# ── Confusion matrix data (TN, FP, FN, TP) ────────────────────────────────
CM_DATA = {
    "Mitral SFT": {
        "cEXT": [121, 116,  81, 175],
        "cNEU": [153,  90, 161,  89],
        "cAGR": [140,  94, 133, 126],
        "cCON": [159,  88, 135, 111],
        "cOPN": [155,  91, 135, 112],
    },
    "Mitral IFT": {
        "cEXT": [ 81, 155,  87, 169],
        "cNEU": [157,  86, 136, 113],
        "cAGR": [ 92, 142,  92, 166],
        "cCON": [110, 137, 104, 141],
        "cOPN": [111, 134, 104, 143],
    },
    "Qwen SFT": {
        "cEXT": [218,   9, 254,  12],
        "cNEU": [115, 144,  46, 188],
        "cAGR": [159,  61, 165, 108],
        "cCON": [169,  57, 170,  97],
        "cOPN": [181,  55, 136, 121],
    },
    "Qwen IFT": {
        "cEXT": [100, 127,  93, 173],
        "cNEU": [172,  87, 136,  98],
        "cAGR": [ 88, 132, 103, 170],
        "cCON": [ 90, 136, 102, 165],
        "cOPN": [ 79, 157,  98, 159],
    },
}

# ── Training log CSV paths ─────────────────────────────────────────────────
LOG_FILES = {
    "Mitral SFT": os.path.join(BASE, "Mitral", "training_logs_mistral_sft_essays.csv"),
    "Mitral IFT": os.path.join(BASE, "Mitral", "training_logs_mistral_ift_essays.csv"),
    "Qwen SFT":   os.path.join(BASE, "Qwen",   "training_logs_qwen_sft.csv"),
    "Qwen IFT":   os.path.join(BASE, "Qwen",   "training_logs _qwen_ift.csv"),
}

# ── Output folders ─────────────────────────────────────────────────────────
OUT_FOLDERS = {
    "Mitral SFT": os.path.join(BASE, "Mitral SFT"),
    "Mitral IFT": os.path.join(BASE, "Mitral IFT"),
    "Qwen SFT":   os.path.join(BASE, "Qwen SFT"),
    "Qwen IFT":   os.path.join(BASE, "Qwen IFT"),
}

TRAITS = ["cEXT", "cNEU", "cAGR", "cCON", "cOPN"]


# ══════════════════════════════════════════════════════════════════════════
# 1. Confusion matrix heatmaps
# ══════════════════════════════════════════════════════════════════════════
def plot_confusion_matrices(model_name, trait_data, out_path):
    n = len(TRAITS)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4.2))
    fig.suptitle(f"Ma trận nhầm lẫn — {model_name} trên Essays Dataset",
                 fontsize=13, fontweight='bold', y=1.01)

    for ax, trait in zip(axes, TRAITS):
        tn, fp, fn, tp = trait_data[trait]
        cm = np.array([[tn, fp], [fn, tp]], dtype=float)
        total = cm.sum()
        cm_norm = cm / total

        ax.imshow(cm_norm, interpolation='nearest', cmap='Blues',
                  vmin=0, vmax=cm_norm.max() * 1.1)
        ax.set_title(trait, fontsize=12, fontweight='bold')
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Dự đoán N", "Dự đoán Y"], fontsize=8)
        ax.set_yticklabels(["Thực tế N", "Thực tế Y"], fontsize=8)

        cell_labels = [["TN", "FP"], ["FN", "TP"]]
        for i in range(2):
            for j in range(2):
                count = int(cm[i, j])
                pct = 100 * count / total
                color = "white" if cm_norm[i, j] > 0.28 else "black"
                ax.text(j, i, f"{cell_labels[i][j]}\n{count}\n({pct:.1f}%)",
                        ha='center', va='center',
                        fontsize=9, color=color, fontweight='bold')

        tnr = tn / (tn + fp) * 100 if (tn + fp) > 0 else 0
        tpr = tp / (fn + tp) * 100 if (fn + tp) > 0 else 0
        ax.set_xlabel(f"TNR={tnr:.1f}%  TPR={tpr:.1f}%", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved CM: {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# 2. Loss curves from training logs
# ══════════════════════════════════════════════════════════════════════════
def plot_loss_curve(model_name, log_path, out_path):
    if not os.path.exists(log_path):
        print(f"  [SKIP] Log not found: {log_path}")
        return

    df = pd.read_csv(log_path)
    # Keep only rows with valid step-level loss (not the final summary row)
    df = df[df['step'].notna() & df['loss'].notna()].copy()
    df['step'] = pd.to_numeric(df['step'], errors='coerce')
    df['loss'] = pd.to_numeric(df['loss'], errors='coerce')
    df['epoch'] = pd.to_numeric(df['epoch'], errors='coerce')
    df = df.dropna(subset=['step', 'loss'])

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df['epoch'], df['loss'], color='steelblue', linewidth=1.5,
            label='Train Loss', alpha=0.85)

    # Smooth trend line
    if len(df) > 10:
        from numpy.polynomial import polynomial as P
        coef = P.polyfit(df['epoch'], df['loss'], 3)
        x_smooth = np.linspace(df['epoch'].min(), df['epoch'].max(), 300)
        y_smooth = P.polyval(x_smooth, coef)
        ax.plot(x_smooth, y_smooth, color='darkblue', linewidth=2,
                linestyle='--', label='Xu hướng (bậc 3)')

    # Epoch boundaries
    for ep in range(1, int(df['epoch'].max()) + 1):
        ax.axvline(x=ep, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)

    ax.set_xlabel('Epoch', fontsize=11)
    ax.set_ylabel('Training Loss', fontsize=11)
    ax.set_title(f'Đường cong Loss — {model_name} (Essays Dataset)',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved LC: {out_path}")


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    for model_name in CM_DATA:
        out_dir = OUT_FOLDERS[model_name]
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n[{model_name}]")

        # Confusion matrix (only if not already generated)
        cm_path = os.path.join(out_dir, "confusion_matrix.png")
        plot_confusion_matrices(model_name, CM_DATA[model_name], cm_path)

        # Loss curve
        lc_path = os.path.join(out_dir, "loss_curve.png")
        plot_loss_curve(model_name, LOG_FILES[model_name], lc_path)

    print("\n✓ Done! Upload to Chapter5/images/ on TeXPage:")
    mapping = [
        ("Mitral SFT/confusion_matrix.png", "essays_mistral_sft_cm.png"),
        ("Mitral IFT/confusion_matrix.png", "essays_mistral_ift_cm.png"),
        ("Qwen SFT/confusion_matrix.png",   "essays_qwen_sft_cm.png"),
        ("Qwen IFT/confusion_matrix.png",   "essays_qwen_ift_cm.png"),
        ("Mitral SFT/loss_curve.png",       "essays_mistral_sft_loss.png"),
        ("Mitral IFT/loss_curve.png",       "essays_mistral_ift_loss.png"),
        ("Qwen SFT/loss_curve.png",         "essays_qwen_sft_loss.png"),
        ("Qwen IFT/loss_curve.png",         "essays_qwen_ift_loss.png"),
    ]
    for src, dst in mapping:
        print(f"  {src:45s} → {dst}")
