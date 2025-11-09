# model_comparator_filtered.py
"""
Evaluate only models that meet a specific frame & class requirement.
Automatically skips models that do not match the criteria.
Saves per-model JSON (batch metrics) + PNG curves + summary.
"""

import os, json
from pathlib import Path
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model

# ======= CONFIG =======
TEST_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart\test"
OUT_DIR  = r"C:\Users\garga\Documents\Maturarbeit\test_data"
BATCH_SIZE  = 8

# Requirement parameters
REQ_FRAMES = 12
REQ_CLASSES = 100

MODELS = [
    r"C:\Users\garga\Documents\Maturarbeit\2\result_of_seq1\asl_seq_model.keras",
    r"C:\Users\garga\Documents\Maturarbeit\3\result_seq2\asl_seq_model.keras",
    r"C:\Users\garga\Documents\Maturarbeit\5\asl_seq_model_smart_v3.keras",
    r"C:\Users\garga\Documents\Maturarbeit\6\asl_seq_model_smart_v11_finetuned.keras",
    r"C:\Users\garga\Documents\Maturarbeit\7\asl_seq_model_smart_v3.keras",
    r"C:\Users\garga\Documents\Maturarbeit\8\asl_seq_model_smart_v3.keras",
    r"C:\Users\garga\Documents\Maturarbeit\9\asl_seq_model_smart_v6.keras",
]

def ensure_dir(p): Path(p).mkdir(parents=True, exist_ok=True)

def get_test_classes(test_dir, limit=None):
    classes = sorted([d for d in os.listdir(test_dir)
                      if os.path.isdir(os.path.join(test_dir, d))])
    if limit:
        classes = classes[:limit]
    return classes

def collect_test_clips(test_dir, class_names):
    clips = []
    for cls in class_names:
        cdir = os.path.join(test_dir, cls)
        for clip_folder in sorted(os.listdir(cdir)):
            clip_dir = os.path.join(cdir, clip_folder)
            if not os.path.isdir(clip_dir):
                continue
            frames = sorted([
                os.path.join(clip_dir, f)
                for f in os.listdir(clip_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            if frames:
                clips.append((frames, class_names.index(cls)))
    return clips

def check_model_requirements(model, req_frames, req_classes):
    """Check if model meets frame & class requirements."""
    try:
        input_shape = model.input_shape[1:]  # remove batch dim
        frames = int(input_shape[0])
        output_classes = int(model.output_shape[-1])
        meets_req = (frames == req_frames) and (output_classes == req_classes)
        return frames, output_classes, meets_req
    except Exception as e:
        print(f"⚠ Could not inspect model: {e}")
        return None, None, False

def adapt_eval(model, model_path, clips, class_names):
    input_shape = model.input_shape[1:]
    frames_per_clip = int(input_shape[0])
    img_size = (int(input_shape[1]), int(input_shape[2]))
    num_classes = len(class_names)

    print(f"   → Evaluating with {frames_per_clip} frames of size {img_size}")

    losses, accs = [], []
    for i, (frame_paths, label) in enumerate(clips[:min(200, len(clips))]):
        frames = []
        for f in range(frames_per_clip):
            idx = min(f, len(frame_paths) - 1)
            img = tf.io.read_file(frame_paths[idx])
            img = tf.image.decode_jpeg(img, channels=3)
            img = tf.image.resize(img, img_size)
            img = img / 255.0
            frames.append(img)
        X = tf.stack(frames)
        X_in = tf.expand_dims(X, axis=0)
        y_oh = tf.one_hot([label], depth=num_classes)

        try:
            res = model.test_on_batch(X_in, y_oh)
            losses.append(float(res[0]))
            accs.append(float(res[1]))
        except Exception as e:
            print(f"⚠️ Skipped clip {i} due to error: {e}")

    avg_acc = float(np.mean(accs)) if accs else 0.0
    avg_loss = float(np.mean(losses)) if losses else 0.0
    print(f"✅ {os.path.basename(model_path)} → Test accuracy: {avg_acc:.4f}")

    return {
        "name": os.path.basename(model_path).replace(".keras", ""),
        "model_path": model_path,
        "final_acc": avg_acc,
        "final_loss": avg_loss,
        "accs": accs,
        "losses": losses,
    }

def plot_save(res, out_dir):
    ensure_dir(out_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    jpath = os.path.join(out_dir, f"{res['name']}_test_{ts}.json")
    with open(jpath, "w") as f:
        json.dump(res, f, indent=2)

    if not res["accs"]:
        print(f"⚠️ No data to plot for {res['name']}")
        return jpath, None

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(res["losses"], label="Loss", color="tab:blue")
    ax1.set_ylabel("Loss"); ax1.set_xlabel("Batch")

    ax2 = ax1.twinx()
    ax2.plot(res["accs"], label="Accuracy", color="tab:orange")
    ax2.set_ylabel("Accuracy")

    plt.title(res["name"], fontsize=12, fontweight="bold")
    ax1.grid(True)
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right")

    ppath = os.path.join(out_dir, f"{res['name']}_plot_{ts}.png")
    plt.tight_layout()
    plt.savefig(ppath)
    plt.close()
    return jpath, ppath

def main():
    ensure_dir(OUT_DIR)
    cls = get_test_classes(TEST_DIR, REQ_CLASSES)
    clips = collect_test_clips(TEST_DIR, cls)
    summary = {}

    for mpath in MODELS:
        print(f"\n[INFO] Checking {mpath}")
        if not os.path.exists(mpath):
            print("   ⚠ Missing file, skipping.")
            continue

        try:
            model = load_model(mpath)
        except Exception as e:
            print(f"⚠ Failed to load {mpath}: {e}")
            continue

        frames, output_classes, meets_req = check_model_requirements(model, REQ_FRAMES, REQ_CLASSES)
        print(f"Model: {os.path.basename(mpath)}\n - Input frames: {frames}, Output classes: {output_classes}\n - Meets requirements: {meets_req}")

        if not meets_req:
            print("   ⚠ Skipping model (does not meet frame/class requirements).")
            continue

        # Evaluate model
        res = adapt_eval(model, mpath, clips, cls)
        j, p = plot_save(res, OUT_DIR)
        summary[res["name"]] = {
            "acc": res["final_acc"],
            "loss": res["final_loss"],
            "json": j,
            "png": p,
        }
        print(f"   ✅ Acc={res['final_acc']:.4f}  Loss={res['final_loss']:.4f}")

    # Save summary
    smry = os.path.join(OUT_DIR, "summary.json")
    with open(smry, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {smry}")

    # Comparison chart
    names = list(summary.keys())
    if names:
        accs = [summary[k]["acc"] for k in names]
        plt.figure(figsize=(10, 4))
        plt.bar(names, accs)
        plt.ylabel("Test Accuracy")
        plt.title(f"Model Comparison (Frames={REQ_FRAMES}, Classes={REQ_CLASSES})")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        comp = os.path.join(OUT_DIR, "comparison.png")
        plt.savefig(comp)
        plt.close()
        print(f"Comparison chart saved to {comp}")
    else:
        print("⚠ No models met the requirements; no comparison chart created.")

if __name__ == "__main__":
    main()
