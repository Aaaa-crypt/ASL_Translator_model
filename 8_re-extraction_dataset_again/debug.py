import os

SEQUENCES_DIR = r"C:\Users\garga\Documents\Maturarbeit\ASL_Citizen\ASL_Citizen\sequences_smart"

test_dir = os.path.join(SEQUENCES_DIR, "test")

print("\nTEST DIR:", test_dir)
print("Exists? ", os.path.isdir(test_dir))

if not os.path.isdir(test_dir):
    exit()

classes = sorted(os.listdir(test_dir))
print("\nFound classes:", len(classes))

for c in classes[:10]:
    print("  ", c)

# Check first class
if classes:
    first = os.path.join(test_dir, classes[0])
    print("\nFirst class path:", first)
    print("Exists? ", os.path.isdir(first))
    print("Contains:", os.listdir(first)[:10])

    # Check first sequence folder
    seqs = os.listdir(first)
    if seqs:
        seq_path = os.path.join(first, seqs[0])
        print("\nFirst sequence path:", seq_path)
        print("Exists? ", os.path.isdir(seq_path))
        print("Frames inside:", os.listdir(seq_path)[:10])
