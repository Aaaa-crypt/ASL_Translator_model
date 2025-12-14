# plot_relative_accuracy.py - UPDATED WITH NORMALIZED ACCURACY

import os
import json
import matplotlib.pyplot as plt

# ---------------- USER CONFIG ----------------
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\comparison"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "normalized_accuracy_comparison.png")
JSON_PATH = os.path.join(OUTPUT_DIR, "normalized_accuracy_metrics.json")
BAR_COLOR = "#1f77b4"

# ---------------- MODEL DATA ----------------
models_info = [
    {
        "model_name": "train_asl_model",
        "num_classes": 26,
        "accuracy": 0.28846153846153844
    },
    {
        "model_name": "train_from_sequences1",
        "num_classes": 2730,
        "accuracy": 0.0003351206541992724
    },
    {
        "model_name": "train_from_sequences5",
        "num_classes": 200,
        "accuracy": 0.0043196543119847775
    },
    {
        "model_name": "training_model3",
        "num_classes": 100,
        "accuracy": 0.0
    },
    {
        "model_name": "training_model4",
        "num_classes": 100,
        "accuracy": 0.02368421107530594
    },
    {
        "model_name": "training_model16",
        "num_classes": 200,
        "accuracy": 0.005201560445129871
    },
    {
        "model_name": "training_model19",
        "num_classes": 200,
        "accuracy": 0.09102730453014374
    },
    {
        "model_name": "model_kaggle",
        "num_classes": 47,
        "accuracy": 0.6100628972053528
    }
]

# ---------------- CALCULATE NORMALIZED ACCURACY ----------------
labels = []
normalized_scores = []
metrics_json = []

for m in models_info:
    labels.append(m["model_name"])
    acc = m["accuracy"]
    n_classes = m["num_classes"]
    
    # **Convert accuracy to percent**
    acc_percent = acc * 100
    # **Normalized accuracy formula from Coenen (2014)**
    prob_percent = 100 / n_classes
    normalized = ((acc_percent - prob_percent) / (100 - prob_percent)) * 100 / 100  # Scale to 0-1
    
    normalized_scores.append(normalized)
    
    metrics_json.append({
        "model_name": m["model_name"],
        "num_classes": n_classes,
        "accuracy": acc,
        **{"normalized_accuracy": normalized, "random_baseline": 1/n_classes}
    })

# ---------------- PLOT BAR GRAPH ----------------
plt.figure(figsize=(12,6))
bars = plt.bar(labels, normalized_scores, color=BAR_COLOR)
plt.title("Normalized Accuracy of ASL Models (Coenen, 2014)")
plt.ylabel("Normalized Accuracy (0 = chance, 1 = perfect)")
plt.ylim(-0.1, 1.1)  # Allow small negative values

# Annotate bars with normalized score
for bar, score in zip(bars, normalized_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
             f"{score:.2f}", ha='center', fontsize=10)

plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(BAR_PLOT_PATH, dpi=300, bbox_inches='tight')
plt.show()
print(f"✅ Normalized accuracy bar graph saved to: {BAR_PLOT_PATH}")

# ---------------- SAVE METRICS TO JSON ----------------
with open(JSON_PATH, "w") as f:
    json.dump(metrics_json, f, indent=2)
print(f"✅ Normalized accuracy metrics saved to: {JSON_PATH}")
