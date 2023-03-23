import os

train_assignments = ["train_06_epochs", "train_07_imgsz", "train_05_close"]

os.system(f"rm whisper.txt")
for assignment in train_assignments:
    os.system(f"rm -rf runs/{assignment}/")
    os.system(f"echo running {assignment} >> whisper.txt")
    os.system(f"mkdir runs/{assignment}")
    os.system(f"python {assignment}.py > runs/{assignment}/{assignment}_1.out")