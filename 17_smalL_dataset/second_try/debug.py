import os

SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\asl_sequences_mixed"

bad_files = []

for root, dirs, files in os.walk(SEQUENCES_DIR):
    for f in files:
        if f.lower().endswith(".jpg"):
            full = os.path.join(root, f)
            if os.path.getsize(full) == 0:
                bad_files.append(full)

print("BAD FILES FOUND:", len(bad_files))
for bf in bad_files:
    print("EMPTY FILE:", bf)
