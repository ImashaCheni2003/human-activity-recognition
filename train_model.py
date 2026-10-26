
from pathlib import Path
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
import joblib

DATA_DIR = Path("UCI HAR Dataset")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

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

# Load test data
X_test = pd.read_csv(
    DATA_DIR / "test" / "X_test.txt",
    sep=r"\s+",
    header=None
)

y_test = pd.read_csv(
    DATA_DIR / "test" / "y_test.txt",
    header=None
).squeeze("columns")

# Load activity names
labels = pd.read_csv(
    DATA_DIR / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

activity_names = labels["activity"].tolist()

# Train Random Forest
print("Training Random Forest model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# Evaluate on the separate test set
predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("\nMODEL RESULTS")
print("-" * 40)
print(f"Test Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    labels=labels["id"].tolist(),
    target_names=activity_names,
    zero_division=0
))

print("CONFUSION MATRIX")
print(confusion_matrix(
    y_test,
    predictions,
    labels=labels["id"].tolist()
))

# Save trained model
joblib.dump(model, MODEL_DIR / "random_forest_model.joblib")
print("\nModel saved to models/random_forest_model.joblib")
