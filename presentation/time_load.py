import pandas as pd
from pathlib import Path
from datetime import datetime

# Add number of images and epochs per dataset
datasets = {
    "Saw": {"path": "saw", "baseline": 0.4851, "images": 240, "epochs": 20},
    "Stopper": {"path": "billets", "baseline": 0.7436, "images": 360, "epochs": 20},
    "Tubes": {"path": "tubes", "baseline": 0.7597, "images": 310, "epochs": 20},
}

policies = {
    "None": "policy_NONE.tsv",
    "Conservative TA": "policy_TA_FULL.tsv",
    "Exaggerated TA": "policy_TA_HUGE.tsv"
}

root_folder = Path("../datasets/")
summary_rows = []

for dataset_name, info in datasets.items():
    n_images = info["images"]
    n_epochs = info["epochs"]

    for policy_name, filename in policies.items():
        path = root_folder / info["path"] / "results" / filename
        df = pd.read_csv(path, sep='\t', parse_dates=["timefinished"])
        df = df.sort_values("timefinished")

        # Compute total training duration in seconds
        duration = (df["timefinished"].iloc[-1] - df["timefinished"].iloc[0]).total_seconds()
        time_per_image_per_epoch = duration / (len(df) * n_images * n_epochs)

        summary_rows.append({
            "Dataset": dataset_name,
            "Policy": policy_name,
            "Total time [min]": round(duration / 60, 1),
            "Avg time / img / epoch [s]": round(time_per_image_per_epoch, 4)
        })

# Show results
summary_df = pd.DataFrame(summary_rows)
print("\nTraining Time Analysis:")
print(summary_df.to_string(index=False))