# plot_training_bar.py
import os
import json
import matplotlib.pyplot as plt

# -------- USER CONFIG --------
HISTORY_PATH = r"C:\Users\garga\Documents\Maturarbeit\05_re-extraction_dataset_again\history_seq_smart_v3.json"
OUTPUT_DIR = r"C:\Users\garga\Documents\Maturarbeit\05_re-extraction_dataset_again\data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
MAX_CLASSES = 100
BAR_PLOT_PATH = os.path.join(OUTPUT_DIR, "training_bar_graph_v3.png")

# Colors
TRAIN_COLOR = "#1f77b4"
VAL_COLOR = "#e28339"

# ---------------- LOAD HISTORY ----------------
with open(HISTORY_PATH, "r") as f:
    history = json.load(f)

# Convert to float just in case
history = {k: [float(v) for v in vals] for k, vals in history.items()}

# Extract final epoch metrics
train_acc = history.get("accuracy", [])
val_acc = history.get("val_accuracy", [])
train_loss = history.get("loss", [])
val_loss = history.get("val_loss", [])

final_train_acc = train_acc[-1] if train_acc else 0
final_val_acc = val_acc[-1] if val_acc else 0
final_train_loss = train_loss[-1] if train_loss else 0
final_val_loss = val_loss[-1] if val_loss else 0

# ---------------- PLOT BAR GRAPH ----------------
plt.figure(figsize=(8, 5))

labels = ["Train Acc", "Val Acc", "Train Loss", "Val Loss"]
values = [final_train_acc, final_val_acc, final_train_loss, final_val_loss]
colors = [TRAIN_COLOR, VAL_COLOR, TRAIN_COLOR, VAL_COLOR]

plt.bar(labels, values, color=colors)
plt.title("Final Training & Validation Metrics (Model v3)")
plt.ylabel("Value")
plt.ylim(0, max(values) * 1.2)

# Add value labels on bars
for i, v in enumerate(values):
    plt.text(i, v + 0.01, f"{v:.3f}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(BAR_PLOT_PATH)
plt.show()

print(f"✅ Training bar graph saved to: {BAR_PLOT_PATH}")
