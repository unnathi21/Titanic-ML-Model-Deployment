import subprocess
import sys
from pathlib import Path

steps = [
    ("Task 1: Training", "train_model.py"),
    ("Task 2: Online inference", "online_inference.py"),
    ("Task 3: Batch inference", "batch_inference.py"),
    ("Task 4: Streaming inference", "streaming_inference.py"),
]
if not Path("train.csv").exists():
    sys.exit("ERROR: train.csv is missing. Download it from https://www.kaggle.com/competitions/titanic/data and place it in this folder.")
for title, script in steps:
    print("\n" + "=" * 60 + "\n" + title + "\n" + "=" * 60, flush=True)
    result = subprocess.run([sys.executable, script])
    if result.returncode != 0:
        sys.exit(f"Stopped because {script} failed. Fix the error and rerun.")
print("\nAll four Python tasks completed. Next start the API with: uvicorn app:app --reload")
