import argparse
import csv
import cv2 as cv
import torch
import logging
import glob
from pathlib import Path
import os
import random

import albumentations as A
from datetime import datetime
import pandas as pd
import numpy as np

from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from augmentation_lists.single_preset import get_aug__grid_shuffle, get_aug__horizontal_flip, get_aug__vertical_flip, get_aug__affine_scale, get_aug__affine_translate 
from augmentation_lists.single_preset import get_aug__affine_rotate, get_aug__affine_shear, get_aug__elastic_transform, get_aug__grid_distortion
from augmentation_lists.single_preset import get_aug__huesatval, get_aug__gaussianblur, get_aug__gaussnoise
from augmentation_lists.single_preset import get_aug__coarse_dropout, get_aug__grid_dropout

import torch.nn.functional as F

# Parse command-line argument
parser = argparse.ArgumentParser()
parser.add_argument("experiment", choices=["billets", "saw", "tubes"], help="Which experiment to run.")
args = parser.parse_args()
EXPERIMENT = args.experiment

device = "cuda"
probability = 0.5
num_epochs = 1000
n_trials = 30

if EXPERIMENT == "billets":
    from billets.dataset import ItemDataset
    from billets.model import Model
    batch_size = 160
    num_classes = 3
elif EXPERIMENT == "saw":
    from saw.dataset import ItemDataset
    from saw.model import Model
    batch_size = 140
    num_classes = 3
elif EXPERIMENT == "tubes":
    from tubes.dataset import ItemDataset
    from tubes.model import Model
    batch_size = 120
    num_classes = 2



augmentation_fn = None

for label in ["vertical_flip", "horizontal_flip", "affine_scale", "affine_translate",
               "grid_shuffle", "affine_rotate", "affine_shear", 
               "grid_distortion", "huesatval", "gaussnoise",
               "coarse_dropout", "grid_dropout"]:


    logging.basicConfig(
        filename=f"{EXPERIMENT}/logs/single_{label}.txt",
        filemode="w",
        level=logging.INFO,   
        format='%(asctime)s\t%(message)s'
    )


    tsv_log_path = f"{EXPERIMENT}/results/single_{label}.tsv"
    with open(tsv_log_path, "w", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["run", "timefinished", "val_map95", "test_map95"])


    augmentation_fn = {
        # g1
        "vertical_flip": get_aug__vertical_flip,
        "horizontal_flip": get_aug__horizontal_flip,
        "affine_scale": get_aug__affine_scale,
        "affine_translate": get_aug__affine_translate,
        "grid_shuffle": get_aug__grid_shuffle,
        # g2
        "affine_rotate": get_aug__affine_rotate,
        "affine_shear": get_aug__affine_shear,
        "elastic_transform": get_aug__elastic_transform,
        "grid_distortion": get_aug__grid_distortion,
        # g3
        "huesatval": get_aug__huesatval,
        "gaussianblur": get_aug__gaussianblur,
        "gaussnoise": get_aug__gaussnoise,
        # g4
        # "remove_background": get_aug__remove_background,
        "coarse_dropout": get_aug__coarse_dropout,
        "grid_dropout": get_aug__grid_dropout,
    }[label]

    augmentations = augmentation_fn(probability)
    for run in range(n_trials):

        model = Model(num_classes=num_classes)

        model.eval()

        model.to(device)

        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

        pattern = os.path.join(EXPERIMENT, "data/images/*.jpg")
        names = [Path(x).stem for x in glob.glob(pattern)]

        split = len(names) // 3
        random.shuffle(names)
        train_names = names[:split]
        val_names = names[split:split * 2]
        test_names = names[split * 2:]

        train_dataset = ItemDataset(EXPERIMENT, train_names, augmentations=augmentations)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=train_dataset.collate_fn)

        val_dataset = ItemDataset(EXPERIMENT, val_names)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

        test_dataset = ItemDataset(EXPERIMENT,test_names)
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=val_dataset.collate_fn)

        metric = MeanAveragePrecision()

        best_map_val = 0
        best_map_test = 0
        for epoch in range(num_epochs):

            model.train()
            train_loss = 0
            for images, targets in train_dataloader:
                images = list(img.to(device) for img in images)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
                loss_dict = model(images, targets)
                loss = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            model.eval()
            with torch.no_grad():
                for images, targets in val_dataloader:
                    images = list(img.to(device) for img in images)
                    outputs = model(images)
                    outputs = [{k: v.cpu() for k, v in p.items()} for p in outputs]
                    torch.cuda.empty_cache()
                    metric.update(outputs, targets)
            result = metric.compute()
            val_map95 = result["map"].item()
            val_map50 = result["map_50"].item()
            metric.reset()


            with torch.no_grad():
                for images, targets in test_dataloader:
                    images = list(img.to(device) for img in images)
                    outputs = model(images)
                    outputs = [{k: v.cpu() for k, v in p.items()} for p in outputs]
                    torch.cuda.empty_cache()
                    metric.update(outputs, targets)
            result = metric.compute()
            test_map95 = result["map"].item()
            test_map50 = result["map_50"].item()
            metric.reset()

            lr_scheduler.step()

            logging.info(f"{epoch}\t{train_loss:.6f}\t{val_map95:.4f}\t{test_map95:.4f}")

            if val_map95 > best_map_val:
                best_map_val = val_map95
                best_map_test = test_map95

        with open(tsv_log_path, "a", newline="") as f:
            writer = csv.writer(f, delimiter="\t")
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            writer.writerow([run + 1, now_str, round(best_map_val, 6), round(best_map_test, 6)])




