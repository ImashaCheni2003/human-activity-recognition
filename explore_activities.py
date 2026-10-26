
from pathlib import Path
import pandas as pd

DATA_DIR = Path("UCI HAR Dataset")

# Read activity labels
labels = pd.read_csv(
    DATA_DIR / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["activity_id", "activity_name"]
)

# Read training labels
y_train = pd.read_csv(
    DATA_DIR / "train" / "y_train.txt",
    header=None,
    names=["activity_id"]
)

# Count records per activity
counts = y_train["activity_id"].value_counts().sort_index()

labels["training_records"] = labels["activity_id"].map(counts)

print("ACTIVITY DISTRIBUTION")
print("-" * 40)
print(labels.to_string(index=False))

print("\nTotal activities:", len(labels))
print("Total training records:", len(y_train))
