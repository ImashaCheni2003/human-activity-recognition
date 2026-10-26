from pathlib import Path

DATA_DIR = Path("UCI HAR Dataset")

required_files = [
    "activity_labels.txt",
    "features.txt",
    "train/X_train.txt",
    "train/y_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
]

print("HUMAN ACTIVITY RECOGNITION")
print("-" * 35)

for file in required_files:
    path = DATA_DIR / file

    if path.exists():
        print(f"[OK] {file}")
    else:
        print(f"[MISSING] {file}")

if all((DATA_DIR / file).exists() for file in required_files):
    print("\nDataset check passed! Ready for machine learning.")
else:
    print("\nPlease check the dataset folder location.")

    
from pathlib import Path
import pandas as pd

DATA_DIR = Path("UCI HAR Dataset")

# Load training data
X_train = pd.read_csv(
    DATA_DIR / "train" / "X_train.txt",
    sep=r"\s+",
    header=None
)

y_train = pd.read_csv(
    DATA_DIR / "train" / "y_train.txt",
    header=None
).squeeze("columns")

# Load testing data
X_test = pd.read_csv(
    DATA_DIR / "test" / "X_test.txt",
    sep=r"\s+",
    header=None
)

y_test = pd.read_csv(
    DATA_DIR / "test" / "y_test.txt",
    header=None
).squeeze("columns")

print("HUMAN ACTIVITY RECOGNITION - DATA SUMMARY")
print("-" * 45)
print(f"Training records: {X_train.shape[0]}")
print(f"Testing records:  {X_test.shape[0]}")
print(f"Number of features: {X_train.shape[1]}")
print(f"Training labels: {sorted(y_train.unique())}")
print(f"Testing labels:  {sorted(y_test.unique())}")
print(f"Missing values in training data: {X_train.isna().sum().sum()}")

print("\nDataset loaded successfully!")
