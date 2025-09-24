import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np

# Define folder and files
root_folder = Path("../datasets/")
datasets = {
    "Saw": {"path": "saw"},
    "Stopper": {"path": "billets"},
    "Tubes": {"path": "tubes"},  # used to sort the policies
}
policies = {
    "None": "policy_none.tsv",
    "Rotate": "single_affine_rotate.tsv",
    "Scale": "single_affine_scale.tsv",
    "Shear": "single_affine_shear.tsv",
    "Translate": "single_affine_translate.tsv",
    "Coarse Dropout": "single_coarse_dropout.tsv",
    "Gauss Noise": "single_gaussnoise.tsv",
    "Grid Distortion": "single_grid_distortion.tsv",
    "Grid Dropout": "single_grid_dropout.tsv",
    "Grid Shuffle": "single_grid_shuffle.tsv",
    "Horizontal Flip": "single_horizontal_flip.tsv",
    "Vertical Flip": "single_vertical_flip.tsv",
}

# Step 1: Sort policy names by max mAP@0.95 from Tubes
policy_scores_tubes = {}
for policy_name, filename in policies.items():
    df = pd.read_csv(root_folder / datasets["Tubes"]["path"] / "results" / filename, sep='\t')
    policy_scores_tubes[policy_name] = df["test_map95"].max()

sorted_policies = sorted(policies.keys(), key=lambda k: policy_scores_tubes[k], reverse=True)

# Step 2: Create 3 vertical subplots (one per dataset)
fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)

summary_rows = []

for ax, (dataset_name, info) in zip(axes, datasets.items()):
    group_data = []
    for policy_name in sorted_policies:
        filename = policies[policy_name]
        path = root_folder / info["path"] / "results" / filename
        df = pd.read_csv(path, sep='\t')
        values = df["test_map95"].values
        group_data.append(values)

        # Collect summary stats
        mean = np.mean(values)
        std = np.std(values)
        summary_rows.append({
            "Dataset": dataset_name,
            "Policy": policy_name,
            "Mean ± Std": f"{mean:.3f} ± {std:.3f}"
        })

    # Vertical boxplot
    box = ax.boxplot(group_data, patch_artist=True, showmeans=True, widths=0.6)
    ax.set_xticks(np.arange(1, len(sorted_policies) + 1))
    ax.set_xticklabels(sorted_policies, rotation=15)
    ax.set_ylabel("mAP@0.95")
    ax.set_title(dataset_name)
    ax.grid(axis='y', linestyle=':', alpha=0.6)

# X-axis label at the bottom
axes[-1].set_xlabel("Augmentation Policy")

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig("datasets_figs/a_boxes_augmentations.png")
plt.show()

# Print summary table
summary_df = pd.DataFrame(summary_rows)
print("\nSummary of mAP@0.95 statistics per dataset and policy (mean ± std):")
print(summary_df.to_string(index=False))
