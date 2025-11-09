import os, json

base = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"
splits = ["train", "val", "test"]
classes = {s: sorted([d for d in os.listdir(os.path.join(base, s)) if os.path.isdir(os.path.join(base, s, d))]) for s in splits}

intersection = sorted(list(set(classes["train"]) & set(classes["val"]) & set(classes["test"])))
print(f"✅ Classes common to all splits: {len(intersection)}")

with open("class_overlap.json", "w") as f:
    json.dump({"train": len(classes["train"]), "val": len(classes["val"]), "test": len(classes["test"]), "common": len(intersection)}, f, indent=2)
