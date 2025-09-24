import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from pathlib import Path
import pandas as pd
import seaborn as sns
from collections import defaultdict
import cv2 as cv
import glob
import os
import random



# Simulate the presence of YOLO-style annotations
# We'll mock up some data for demonstration purposes

def load_annotations(label_folder: Path, names_subset, split: str):
    boxes = []
    for name in names_subset:
        txt_file = label_folder / f"{name}.txt"
        if not txt_file.exists():
            continue
        with open(txt_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    cls, cx, cy, w, h = map(float, parts)
                    boxes.append((int(cls), cx, cy, w, h, split))
    return boxes

def accumulate_heatmap_data(boxes, bins=50):
    # Initialize heatmaps
    heatmap_xy = np.zeros((bins, bins))
    heatmap_wh = np.zeros((bins, bins))

    for _, cx, cy, w, h, split in boxes:
        xi = min(int(cx * bins), bins - 1)
        yi = min(int((1 - cy) * bins), bins - 1)  # Flip Y-axis
        wi = min(int(w * bins), bins - 1)
        hi = min(int(h * bins), bins - 1)
        heatmap_xy[yi, xi] += 1
        heatmap_wh[hi, wi] += 1

    return heatmap_xy, heatmap_wh


def plot_heatmap(data, title, xlabel, ylabel, flip_y=False):
    masked = np.ma.masked_where(data == 0, data)

    # Create colormap from white to yellow-red-black
    cmap = plt.get_cmap('hot', 256)
    newcolors = cmap(np.linspace(0, 1, 256))
    newcolors[0] = [1, 1, 1, 1]  # white for zero values
    custom_cmap = colors.ListedColormap(newcolors)

    if flip_y:
        masked = np.flipud(masked)
    plt.imshow(masked, cmap=custom_cmap,
               norm=colors.LogNorm(vmin=1, vmax=max(1, masked.max())),
               extent=[0, 1, 1, 0])  # flip Y axis here!

    # plt.colorbar(label="Log-scaled count")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # Ticks from 0% to 100%
    ticks = np.linspace(0, 1, 6)
    labels = [f"{int(x * 100)}%" for x in ticks]
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)

def plot_sample_image(data_path, name):
    size = 500
    image = cv.imread(str(data_path / f"images/{name}.jpg"))
    image = cv.resize(image, dsize=(size, size))

    label_path = data_path / f"labels/{name}.txt"
    if not label_path.exists():
        return

    with open(label_path) as fh:
        labels = fh.read().split("\n")

    for label in labels:
        if label:
            cls, xc, yc, w, h = label.split(" ")
            xc, yc, w, h = float(xc) * size, float(yc) * size, float(w) * size, float(h) * size
            x1, y1, x2, y2 = xc - w // 2, yc - h // 2, xc + w // 2, yc + h // 2
            color = (0, 255, 0) if cls == "0" else (125, 255, 255)
            cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 5)

    plt.title("Image scaled to 1:1 side ratio")
    plt.imshow(image)
    plt.xticks([])
    plt.yticks([])



folder = Path("../datasets/")

dataset_info = [
    ("Stopper", "billets/data", ["Billet", "Stopper"], "image_P1_43_1743017877"),
    ("Tubes", "tubes/data", ["Tube", ], "frame-1722585101-1-0"),
    ("Saw", "saw/data", ["Waste", "Billet"], "data-110-20240905_211101-735"),
]



dataset_splits = {}

for dataset_name, data_subfolder, labels, sample in dataset_info:
    data_dir = folder / data_subfolder
    pattern = os.path.join(data_dir, "images/*.jpg")
    names = [Path(x).stem for x in glob.glob(pattern)]
    random.shuffle(names)
    split = len(names) // 3
    splits = {
        'train': names[:split],
        'val': names[split:split * 2],
        'test': names[split * 2:]
    }

    dataset_splits[dataset_name] = splits



fig, axes = plt.subplots(3, 3, figsize=(12, 12))
plt.subplots_adjust(wspace=0.3, hspace=0.4)

for row_idx, (name, data_folder, labels, sample_name) in enumerate(dataset_info):
    dataset_full_path = folder / data_folder
    splits = dataset_splits[name]
    all_boxes = []

    for split_name, subset_names in splits.items():
        label_folder = dataset_full_path / "labels"
        boxes = load_annotations(label_folder, subset_names, split_name)
        all_boxes.extend(boxes)

    heatmap_xy, heatmap_wh = accumulate_heatmap_data(all_boxes, bins=30)
    df = pd.DataFrame(all_boxes, columns=["cls", "xc", "yc", "w", "h", "split"])

    print(f"Dataset: {name}, classes: {labels}")

    for split, split_set in df.groupby("split"):
        print(f"Split: {split}")
        for cls, cls_set in split_set.groupby('cls'):
            label = labels[int(cls)]
            w_data, h_data = cls_set["w"].values, cls_set["h"].values
            print(f"Class: {label} | Average count: {len(cls_set)/len(split_set):.2f} | width: {w_data.mean():.2f} ± {w_data.std():.2f} | height: {h_data.mean():.2f} ± {h_data.std():.2f}")

    plt.sca(axes[row_idx, 0])
    plot_sample_image(dataset_full_path, sample_name)
    axes[row_idx, 0].set_title(f"{name}: Sample Image")

    plt.sca(axes[row_idx, 1])
    plot_heatmap(heatmap_xy, f"{name}: Center Heatmap", "x [%]", "y [%]", flip_y=True)

    plt.sca(axes[row_idx, 2])
    plot_heatmap(heatmap_wh, f"{name}: Size Heatmap", "width [%]", "height [%]")


# Save and display
plt.tight_layout()
plt.savefig("datasets_figs/a_heatmaps_datasets.png")
plt.show()
