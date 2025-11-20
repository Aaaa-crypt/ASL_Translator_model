import os

PROJECT_ROOT = r"C:\Users\garga\Documents\Maturarbeit"  # <-- your repo path
THRESHOLD_MB = 50  # change if needed

large_files = []

for root, dirs, files in os.walk(PROJECT_ROOT):
    for file in files:
        file_path = os.path.join(root, file)
        try:
            size_mb = os.path.getsize(file_path) / (1024 * 1024)
            if size_mb > THRESHOLD_MB:
                large_files.append((size_mb, file_path))
        except:
            pass

large_files.sort(reverse=True)

print("\n=== LARGE FILES (> {} MB) ===".format(THRESHOLD_MB))
for size, path in large_files:
    print(f"{size:.2f} MB  -  {path}")
