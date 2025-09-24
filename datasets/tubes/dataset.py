import glob
import os
from pathlib import Path

import numpy as np
import matplotlib.pylab as plt
import cv2 as cv
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A


class ItemDataset(Dataset):

    def __init__(self, folder, names, augmentations=None):
        self._len = len(names)

        target_width, target_height = 608, 608

        self._augmentation = augmentations


        self._inputs = []
        self._targets = []

        for idx, name in enumerate(names):
            image_path = os.path.join(folder, "data", "images", name + ".jpg")
            label_path = os.path.join(folder, "data", "labels", name + ".txt")

            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
            image = cv.resize(image, (target_width, target_height))

            with open(label_path, "r") as f:
                label_data = f.read()

            boxes = []
            classes = []
            if label_data:
                for sample in label_data.split("\n"):
                    values = sample.strip().split(" ")
                    if len(values) < 3:
                        continue

                    cls, coordinates = int(values[0]), list(map(float, values[1:]))
                    classes.append(cls + 1)

                    # boxes[0] = boxes[0] * target_width
                    xc, yc, w, h = coordinates

                    bbox = (xc - (w / 2) , yc - (h / 2), xc + (w / 2), yc + (h / 2), )
                    bbox = np.array(bbox)
                    bbox = bbox * np.array([target_width, target_height, target_width, target_height])

                    boxes.append(torch.tensor(bbox).float())

            target = {
                "boxes": torch.stack(boxes) if boxes else torch.empty(0, 4),
                "labels": torch.tensor(classes, dtype=torch.int64),
            }

            image = np.expand_dims(image, axis=-1)

            self._inputs.append(image)
            self._targets.append(target)


    def collate_fn(self, batch):
        return tuple(zip(*batch))

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        image = self._inputs[idx]
        bboxes = self._targets[idx]["boxes"]
        labels = self._targets[idx]["labels"]

        if self._augmentation is not None:
            augmented = self._augmentation(image=image.copy(), bboxes=bboxes, class_labels=labels.tolist())
            image = augmented["image"]
            bboxes = augmented["bboxes"]
            labels = augmented["class_labels"]

        bboxes = torch.tensor(bboxes, dtype=torch.float32)  # Bounding boxes as float tensors
        labels = torch.tensor(labels, dtype=torch.long)  # Labels as integer tensors

        image = torch.tensor(image / 255).float()
        image = image.permute(2, 0, 1)

        return image, {"boxes": bboxes, "labels": labels}






if __name__ == "__main__":

    folder = "tubes"
    pattern = os.path.join(folder, "data/images/*.jpg")
    names = [Path(x).stem for x in glob.glob(pattern)]

 
    dataset = ItemDataset(folder, names)

    dataloader = DataLoader(dataset, batch_size=6, shuffle=False, collate_fn=dataset.collate_fn)

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(inputs[0].shape)

    #     target = targets[0]
    #     image = inputs[0].permute(1, 2, 0).numpy()

    #     for i, (x1, y1, x2, y2) in enumerate(target["boxes"]):

    #         h, w, _ = image.shape
    #         # x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
    #         # Choose color and draw rectangle
    #         color = (0, 255, 0)  # Green
    #         cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

    #     plt.imshow(image)
    #     plt.show()
