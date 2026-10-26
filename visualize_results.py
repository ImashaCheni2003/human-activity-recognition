
from pathlib import Path
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

DATA_DIR = Path("UCI HAR Dataset")
MODEL_PATH = Path("models/random_forest_model.joblib")
OUTPUT_DIR = Path("results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Load test data and saved model
X_test = pd.read_csv(
    DATA_DIR / "test" / "X_test.txt",
    sep=r"\s+",
    header=None
)
y_test = pd.read_csv(
    DATA_DIR / "test" / "y_test.txt",
    header=None
).squeeze("columns")

model = joblib.load(MODEL_PATH)
predictions = model.predict(X_test)

# Load activity names in label order
labels = pd.read_csv(
    DATA_DIR / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

activity_ids = labels["id"].tolist()
activity_names = labels["activity"].tolist()

cm = confusion_matrix(
    y_test, predictions, labels=activity_ids
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=activity_names
)

fig, ax = plt.subplots(figsize=(11, 8))
display.plot(ax=ax, cmap="Blues", xticks_rotation=35, values_format="d")
ax.set_title("Human Activity Recognition - Random Forest")
ax.set_xlabel("Predicted Activity")
ax.set_ylabel("Actual Activity")
plt.tight_layout()

output_path = OUTPUT_DIR / "confusion_matrix.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

print(f"Confusion matrix saved to: {output_path}")
