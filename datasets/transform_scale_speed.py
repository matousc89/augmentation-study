import albumentations as A
import numpy as np
import time
import matplotlib.pyplot as plt

# Create a random 500x500 RGB image
image = (np.random.rand(500, 500, 3) * 255).astype(np.uint8)

# Conservative and exaggerated transform dictionaries
ta_co_transforms = {
    "Affine Translate": A.Affine(translate_percent=0.1, shear=0, rotate=0, scale=1.0, p=1.0),
    "Affine Scale": A.Affine(scale=(0.8, 1.2), shear=0, rotate=0, translate_percent=0, p=1.0),
    "Affine Shear": A.Affine(shear=(-15, 15), scale=1, rotate=0, translate_percent=0, p=1.0),
    "Affine Rotate": A.Affine(rotate=(-15, 15), scale=1, shear=0, translate_percent=0, p=1.0),
    "Coarse Dropout": A.CoarseDropout(
        num_holes_range=(1, 10),
        hole_height_range=(0.05, 0.3),
        hole_width_range=(0.05, 0.3),
        p=1.0
    ),
    "Grid Dropout": A.GridDropout(ratio=0.5, unit_size_range=(10, 100), p=1.0),
    "Gaussian Blur": A.GaussianBlur(sigma_limit=(0, 5.0), p=1.0),
    "Gaussian Noise": A.GaussNoise(std_range=(0.0, 0.25), p=1.0),
    "Elastic Transform": A.ElasticTransform(alpha=500, sigma=250, p=1.0),
    "Grid Distortion": A.GridDistortion(num_steps=8, distort_limit=0.3, p=1.0),
    "Hue/Saturation/Value": A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
    "Horizontal Flip": A.HorizontalFlip(p=1.0),
    "Vertical Flip": A.VerticalFlip(p=1.0),
    "Grid Shuffle": A.RandomGridShuffle(grid=(1, 2), p=1.0),
    "No Operation": A.NoOp(p=1.0),
}

ta_ex_transforms = {
    "Affine Translate": A.Affine(translate_percent=0.2, shear=0, rotate=0, scale=1.0, p=1.0),
    "Affine Scale": A.Affine(scale=(0.6, 1.4), shear=0, rotate=0, translate_percent=0, p=1.0),
    "Affine Shear": A.Affine(shear=(-30, 30), scale=1, rotate=0, translate_percent=0, p=1.0),
    "Affine Rotate": A.Affine(rotate=(-30, 30), scale=1, shear=0, translate_percent=0, p=1.0),
    "Coarse Dropout": A.CoarseDropout(
        num_holes_range=(1, 20),
        hole_height_range=(0.1, 0.6),
        hole_width_range=(0.1, 0.6),
        p=1.0
    ),
    "Grid Dropout": A.GridDropout(ratio=0.5, unit_size_range=(10, 100), p=1.0),
    "Gaussian Blur": A.GaussianBlur(sigma_limit=(0, 10.0), p=1.0),
    "Gaussian Noise": A.GaussNoise(std_range=(0.0, 0.5), p=1.0),
    "Elastic Transform": A.ElasticTransform(alpha=1000, sigma=500, p=1.0),
    "Grid Distortion": A.GridDistortion(num_steps=8, distort_limit=0.6, p=1.0),
    "Hue/Saturation/Value": A.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=60, val_shift_limit=40, p=1.0),
    "Horizontal Flip": A.HorizontalFlip(p=1.0),
    "Vertical Flip": A.VerticalFlip(p=1.0),
    "Grid Shuffle": A.RandomGridShuffle(grid=(1, 2), p=1.0),
    "No Operation": A.NoOp(p=1.0),
}

def time_transform(transform, iterations=10):
    start = time.perf_counter()
    for _ in range(iterations):
        _ = transform(image=image)['image']
    end = time.perf_counter()
    return (end - start) * 1000 / iterations  # ms

# Collect timing data
co_times, ex_times = {}, {}
for name in ta_co_transforms:
    t_co = time_transform(ta_co_transforms[name])
    t_ex = time_transform(ta_ex_transforms[name])
    co_times[name] = t_co
    ex_times[name] = t_ex

# Sort by total time
sorted_names = sorted(co_times.keys(), key=lambda n: ex_times[n])
co_vals = [co_times[n] for n in sorted_names]
ex_vals = [ex_times[n] for n in sorted_names]
epsilon = 1.0  # in milliseconds
delta_perc = [100 * (ex_vals[i] - co_vals[i]) / (co_vals[i] + epsilon) for i in range(len(co_vals))]

# Plotting two subplots horizontally
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 10), sharey=True)

# Left plot: conservative times (log scale)
bars1 = ax1.barh(sorted_names, co_vals, label='Conservative TA', color='skyblue')
ax1.set_xlabel('Execution Time (ms)')
ax1.set_title('Conservative Transform Times')
ax1.set_xscale('log')
ax1.invert_yaxis()  # so fastest is at top
ax1.grid(True, axis='x', linestyle='--', alpha=0.4)

# Annotate with total exaggerated time on the left (bold)
for i, name in enumerate(sorted_names):
    ax1.text(0.01, i, f"{ex_vals[i]:.2f} ms", va='center', ha='left',
             fontsize=9, fontweight='bold', color='black',
             transform=ax1.get_yaxis_transform())

# Right plot: delta percentages
bars2 = ax2.barh(sorted_names, delta_perc, label='Exaggeration Overhead (%)', color='orange')
ax2.set_xlabel('Δ Time (%)')
ax2.set_title('Relative Overhead of Exaggerated Transform')
ax2.grid(True, axis='x', linestyle='--', alpha=0.4)

# Annotate with percentage
for i, val in enumerate(delta_perc):
    ax2.text(val + 1, i, f"{val:.0f}%", va='center', ha='left', fontsize=9)

plt.tight_layout()
plt.savefig("a_transform_time_split.png")
plt.show()
