
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

results_path = Path("results/model_comparison.csv")
output_dir = Path("results")
results = pd.read_csv(results_path)

metrics = ["Accuracy", "Macro F1 Score"]

ax = results.set_index("Model")[metrics].plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0.85, 1.0),
    rot=0
)

ax.set_title("Human Activity Recognition: Model Comparison")
ax.set_ylabel("Score")
ax.set_xlabel("Machine Learning Model")
ax.legend(loc="lower right")
ax.grid(axis="y", linestyle="--", alpha=0.4)

plt.tight_layout()

output_path = output_dir / "model_comparison.png"
plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close()

print(f"Comparison chart saved to: {output_path}")
