import time
import glob

import numpy as np
import torch
import torchvision
import cv2 as cv
from torch.utils.data import DataLoader
import matplotlib.pylab as plt

from model import Model


from dataset import ItemDataset

def collate_fn(batch):
    return tuple(zip(*batch))


device = "cuda"
batch_size = 1


model = Model(num_classes=2)



# model_data = "checkpoints/model_v1.pt"
# model.load_state_dict(torch.load(model_data))
model.to(device)
model.eval()

#
# total_params = 0
# trainable_params = 0
# for name, param in model.named_parameters():
#     total_params += param.numel()
#     if param.requires_grad:
#         trainable_params += param.numel()
#         print(f"Trainable: {name}")
#     else:
#         print(f"Frozen:    {name}")
#
# print(f"\nTotal parameters: {total_params:,}")
# print(f"Trainable parameters: {trainable_params:,}")
# print(f"Frozen parameters: {total_params - trainable_params:,}")




val_dataset = ItemDataset("data/val")
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


for images, val_targets in val_dataloader:


    t0 = time.time()
    val_inputs = list(img.to(device) for img in images)


    outputs = model(val_inputs)
    t_end = time.time() - t0
    print(f"Estimated time: {t_end}")

#     for image, output, target in zip(val_inputs, outputs, val_targets):


#         image = image.detach().cpu().permute(1, 2, 0).numpy()

#         boxes = output['boxes'].detach().cpu().numpy()
#         scores = output['scores'].detach().cpu().numpy()
#         labels = output['labels'].detach().cpu().numpy()

#         # boxes = target['boxes'].detach().cpu().numpy()

#         # print(boxes, scores, labels)

#         # Draw bounding boxes on the image
#         for i, (x1, y1, x2, y2) in enumerate(boxes):
#             score = scores[i]
#             label = labels[i]

#             # Convert normalized coordinates to pixel values
#             h, w, _ = image.shape
#             # x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)

#             # Choose color and draw rectangle
#             color = (0, 255, 0)  # Green

#             cv.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)

#             # Put label text
#             text = f"Class {label}: {score:.2f}"
#             cv.putText(image, text, (int(x1), int(y1) - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

#         # Show the image
#         cv.imshow("Detections", image)
#         cv.waitKey(0)
#         cv.destroyAllWindows()




