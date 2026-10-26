
from pathlib import Path
import time

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import LinearSVC

DATA_DIR = Path("UCI HAR Dataset")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

# Load the original training and testing datasets
X_train = pd.read_csv(
    DATA_DIR / "train" / "X_train.txt",
    sep=r"\s+",
    header=None
)
X_test = pd.read_csv(
    DATA_DIR / "test" / "X_test.txt",
    sep=r"\s+",
    header=None
)
y_train = pd.read_csv(
    DATA_DIR / "train" / "y_train.txt",
    header=None
).squeeze("columns")
y_test = pd.read_csv(
    DATA_DIR / "test" / "y_test.txt",
    header=None
).squeeze("columns")

models = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Linear SVM": LinearSVC(random_state=42, max_iter=5000),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    ),
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    start_time = time.time()

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    elapsed = time.time() - start_time

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro F1 Score": f1_score(
            y_test, predictions, average="macro"
        ),
        "Training + Prediction Time (seconds)": round(elapsed, 2),
    })

results_df = pd.DataFrame(results).sort_values(
    by="Macro F1 Score", ascending=False
)

results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
print("\nSaved results to results/model_comparison.csv")
