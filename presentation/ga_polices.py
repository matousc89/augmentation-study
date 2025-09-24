import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Paths and baselines
root_folder = Path("../datasets/")
datasets = {
    "Saw": {"path": "saw", "baseline": 0.4851},
    "Stopper": {"path": "billets", "baseline": 0.7436},
    "Tubes": {"path": "tubes", "baseline": 0.7597},
}
policies = {
    "None": "policy_none.tsv",
    "CoTA": "policy_ta_co.tsv",
    "ExTA": "policy_ta_ex.tsv"
}
colors = {'None': 'lightgray', 'CoTA': 'skyblue', 'ExTA': 'lightgreen'}

# Gather all normalized data grouped by policy
grouped_data = {k: [] for k in policies}
for dataset in datasets.values():
    baseline = dataset["baseline"]
    for policy_name, filename in policies.items():
        path = root_folder / dataset["path"] / "results" / filename
        df = pd.read_csv(path, sep='\t')
        relative = (df["test_map95"].values - baseline) / baseline
        grouped_data[policy_name].extend(relative)

# Create compact figure
plt.figure(figsize=(3.5, 2.5))
data = [grouped_data[key] for key in ["None", "CoTA", "ExTA"]]
bp = plt.boxplot(data, patch_artist=True, showmeans=True, widths=0.6)

# Color boxes
for patch, key in zip(bp['boxes'], ["None", "CoTA", "ExTA"]):
    patch.set_facecolor(colors[key])

# Axis formatting
plt.xticks([1, 2, 3], ["None", "CoTA", "ExTA"], fontsize=10)
plt.yticks(fontsize=10)
plt.ylabel("Relative improvement", fontsize=10)
plt.grid(axis='y', linestyle=':', alpha=0.6)

# Tight layout
plt.tight_layout()
plt.savefig("graphical_abstract_boxplot.png", dpi=600)
plt.show()