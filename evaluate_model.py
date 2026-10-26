
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
)
from sklearn.svm import LinearSVC

DATA_DIR = Path("UCI HAR Dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

X_train = pd.read_csv(
    DATA_DIR / "train" / "X_train.txt",
    sep=r"\s+",
    header=None,
)
X_test = pd.read_csv(
    DATA_DIR / "test" / "X_test.txt",
    sep=r"\s+",
    header=None,
)
y_train = pd.read_csv(
    DATA_DIR / "train" / "y_train.txt",
    header=None,
).squeeze("columns")
y_test = pd.read_csv(
    DATA_DIR / "test" / "y_test.txt",
    header=None,
).squeeze("columns")

activity_labels = pd.read_csv(
    DATA_DIR / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"],
)
names = activity_labels["activity"].tolist()
ids = activity_labels["id"].tolist()

print("Training the Linear SVM model...")
model = LinearSVC(random_state=42, max_iter=5000)
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(f"\nAccuracy: {accuracy_score(y_test, predictions):.4f}")
print(f"Macro F1: {f1_score(y_test, predictions, average='macro'):.4f}")

print("\nCLASSIFICATION REPORT")
print(classification_report(
    y_test,
    predictions,
    labels=ids,
    target_names=names,
    digits=4,
))

cm = confusion_matrix(y_test, predictions, labels=ids)
fig, ax = plt.subplots(figsize=(11, 8))
display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=names,
)
display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=35,
    values_format="d",
)
ax.set_title("Linear SVM - Human Activity Recognition")
plt.tight_layout()
fig.savefig(
    RESULTS_DIR / "linear_svm_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight",
)
plt.close(fig)

joblib.dump(model, "models/linear_svm_model.joblib")

print("\nSaved detailed confusion matrix.")
print("Saved model to models/linear_svm_model.joblib")
