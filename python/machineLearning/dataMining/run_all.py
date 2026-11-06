"""
run_all.py - runs the entire pipeline in the correct order.
    python run_all.py
"""
import subprocess
import sys
import time

STEPS = [
    ("0. Generate dataset (skip if you already own a CSV)", "make_dataset.py"),
    ("1. Preprocess", "01_preprocess.py"),
    ("2. Classification", "02_classification.py"),
    ("3. Clustering", "03_clustering.py"),
    ("4. Association rules", "04_association.py"),
]

overall = time.time()
for label, script in STEPS:
    print("\n" + "#" * 78)
    print(f"# {label}   ->   {script}")
    print("#" * 78)
    t = time.time()
    r = subprocess.run([sys.executable, script])
    if r.returncode != 0:
        print(f"\n!! {script} FAILED (exit code {r.returncode}) - stopping.")
        sys.exit(r.returncode)
    print(f"-- {script} finished in {time.time()-t:.1f}s")

print("\n" + "#" * 78)
print(f"# ALL DONE in {time.time()-overall:.1f}s")
print("# Everything you need for the report is in ./outputs/")
print("#" * 78)
