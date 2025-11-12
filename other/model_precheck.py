# model_precheck.py
"""
Pre-check multiple .keras sequence models:
- Detect input frames & image size
- Detect number of output classes
- Only mark models that meet target requirements (frames=5, classes=10)
"""

import os
from tensorflow.keras.models import load_model

# ===== CONFIG =====
MODELS = [
    r"C:\Users\garga\Documents\Maturarbeit\2\result_of_seq1\asl_seq_model.keras",
    r"C:\Users\garga\Documents\Maturarbeit\3\result_seq2\asl_seq_model.keras",
    r"C:\Users\garga\Documents\Maturarbeit\5\asl_seq_model_smart_v3.keras",
    r"C:\Users\garga\Documents\Maturarbeit\6\asl_seq_model_smart_v11_finetuned.keras",
    r"C:\Users\garga\Documents\Maturarbeit\7\asl_seq_model_smart_v3.keras",
    r"C:\Users\garga\Documents\Maturarbeit\8\asl_seq_model_smart_v3.keras",
    r"C:\Users\garga\Documents\Maturarbeit\9\asl_seq_model_smart_v6.keras",
]

TARGET_FRAMES = 5
TARGET_CLASSES = 10

def check_model(model_path):
    if not os.path.exists(model_path):
        print(f"⚠ Model not found: {model_path}")
        return None

    try:
        model = load_model(model_path)
    except Exception as e:
        print(f"⚠ Failed to load {model_path}: {e}")
        return None

    # Inspect input shape
    inp_shape = model.input_shape  # (batch, frames, H, W, C)
    frames_per_clip = inp_shape[1] if len(inp_shape) == 5 else 1
    img_size = (inp_shape[2], inp_shape[3]) if len(inp_shape) == 5 else (inp_shape[1], inp_shape[2])

    # Inspect output shape
    out_shape = model.output_shape  # (batch, num_classes)
    num_classes = out_shape[1]

    # Check requirements
    meets_req = (frames_per_clip == TARGET_FRAMES) and (num_classes == TARGET_CLASSES)

    print(f"\nModel: {os.path.basename(model_path)}")
    print(f" - Input frames: {frames_per_clip}, Image size: {img_size}")
    print(f" - Output classes: {num_classes}")
    print(f" - Meets requirements (frames={TARGET_FRAMES}, classes={TARGET_CLASSES}): {meets_req}")

    return {
        "name": os.path.basename(model_path).replace(".keras",""),
        "path": model_path,
        "frames": frames_per_clip,
        "img_size": img_size,
        "num_classes": num_classes,
        "meets_req": meets_req
    }

def main():
    results = []
    for mpath in MODELS:
        res = check_model(mpath)
        if res:
            results.append(res)

    # Only models that meet requirements
    valid_models = [r for r in results if r["meets_req"]]
    print("\n✅ Models meeting requirements:")
    for r in valid_models:
        print(f" - {r['name']} ({r['frames']} frames, {r['num_classes']} classes)")

if __name__ == "__main__":
    main()
