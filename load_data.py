
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
