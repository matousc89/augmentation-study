import cv2 as cv
import torch
import logging

from dataset import ItemDataset
from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from model import Model

def collate_fn(batch):
    return tuple(zip(*batch))

device = "cuda"
batch_size = 160
num_epochs = 10000
reverse_split = False


logging.basicConfig(
    filename=f"logs/log_train_base_{reverse_split}.txt",
    filemode="w",
    level=logging.INFO,   
    format='%(asctime)s\t%(message)s'
)




model = Model(num_classes=3)
model.eval()

model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.AdamW(params, lr=1e-4)
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)


if reverse_split:
    train_folder = "data/val"
    val_folder = "data/train"
else:
    train_folder = "data/train"
    val_folder = "data/val"

train_dataset = ItemDataset(train_folder, augmentations=None)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
val_dataset = ItemDataset(val_folder, augmentations=None)   
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

metric = MeanAveragePrecision()

best_map = 0
for epoch in range(num_epochs):

    model.train()
    train_loss = 0
    val_loss = 0

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
            # targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            outputs = [{k: v.cpu() for k, v in p.items()} for p in outputs]
            # targets = [{k: v.cpu() for k, v in t.items()} for t in targets]
            torch.cuda.empty_cache()
            metric.update(outputs, targets)

    result = metric.compute()
    map95 = result["map"].item()
    map50 = result["map_50"].item()

    metric.reset()

    lr_scheduler.step()

    #print(f"{epoch}\t{map50:.4f}\t{map95:.4f}\t{train_loss:.6f}")

    logging.info(f"{epoch}\t{map50:.4f}\t{map95:.4f}\t{train_loss:.6f}")

    if map95 > best_map:
        best_map = map95
        torch.save(model.state_dict(), f"checkpoints/model_v2_{reverse_split}.pt")
        logging.info("Saving model")



