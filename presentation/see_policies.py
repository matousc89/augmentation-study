import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Define folder and files
root_folder = Path("../datasets/")
datasets = {
    "Saw": {"path": "saw"},
    "Stopper": {"path": "billets"},
    "Tubes": {"path": "tubes"},
}
policies = {
    "None": "policy_none.tsv",
    "Conservative TA": "policy_ta_co.tsv",
    "Exaggerated TA": "policy_ta_ex.tsv"
}

# Create subplots without sharing Y axis
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
colors = ['lightgray', 'skyblue', 'lightgreen']

# Table data collection
summary_rows = []

for ax, (dataset_name, info) in zip(axes, datasets.items()):
    group_data = []
    for policy_name, filename in policies.items():
        path = root_folder / info["path"] / "results" / filename
        df = pd.read_csv(path, sep='\t')
        values = df["test_map95"].values
        group_data.append(values)

        # Collect summary stats (mean ± std)
        mean = np.mean(values)
        std = np.std(values)
        summary_rows.append({
            "Dataset": dataset_name,
            "Policy": policy_name,
            "Mean ± Std": f"{mean:.3f} ± {std:.3f}"
        })

    # Boxplot
    box = ax.boxplot(group_data, patch_artist=True, showmeans=True, widths=0.6)
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    # Labels and formatting
    ax.set_title(dataset_name)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(policies.keys(), rotation=15)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

# Y-axis label only on the first subplot
axes[0].set_ylabel("Best mAP@0.95")

# Shared legend
fig.legend([
    plt.Line2D([0], [0], color='lightgray', lw=6),
    plt.Line2D([0], [0], color='skyblue', lw=6),
    plt.Line2D([0], [0], color='lightgreen', lw=6),
], ['None', 'TA', 'Exaggerated TA'], loc='upper center', ncol=4)

plt.tight_layout(rect=[0, 0, 1, 0.92])
plt.savefig("datasets_figs/a_boxes_policies.png")
plt.show()


# Print summary table
summary_df = pd.DataFrame(summary_rows)
print("\nSummary of mAP@0.95 statistics per dataset and policy (mean ± std):")
print(summary_df.to_string(index=False))
