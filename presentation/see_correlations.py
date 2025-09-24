import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

# Define folder and files
root_folder = Path("../datasets/")
datasets = {
    "Saw": {"path": "saw"},
    "Stopper": {"path": "billets"},
    "Tubes": {"path": "tubes"},
}
policies = {
    "None": "policy_none.tsv",
    "Rotate": "single_affine_rotate.tsv",
    "Scale": "single_affine_scale.tsv",
    "Shear": "single_affine_shear.tsv",
    "Translate": "single_affine_translate.tsv",
    # "Grid Shuffle": "single_grid_shuffle.tsv",
    "Horizontal Flip": "single_horizontal_flip.tsv",
    "Vertical Flip": "single_vertical_flip.tsv",
}

# Collect correlation results
correlation_rows = []

for dataset_name, info in datasets.items():
    for policy_name, filename in policies.items():
        path = root_folder / info["path"] / "results" / filename
        df = pd.read_csv(path, sep='\t')

        if "test_map95" in df.columns and "val_map95" in df.columns:
            corr, pval = pearsonr(df["val_map95"], df["test_map95"])
            correlation_rows.append({
                "Dataset": dataset_name,
                "Policy": policy_name,
                "Pearson r": f"{corr:.3f}",
                "p-value": f"{pval:.3g}"
            })
        else:
            correlation_rows.append({
                "Dataset": dataset_name,
                "Policy": policy_name,
                "Pearson r": "N/A",
                "p-value": "N/A"
            })

# Create and print correlation table
correlation_df = pd.DataFrame(correlation_rows)
print("\nCorrelation between val_map95 and test_map95 (Pearson):")
print(correlation_df.to_string(index=False))