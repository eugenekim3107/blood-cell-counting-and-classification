import torch
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import CellDataset
import os
from commonFunc import (
IOU,
non_max_suppression,
mean_average_precision,
cellboxes_to_boxes,
get_bboxes,
)
from loss import YoloLoss

# Hyperparemters etc.
lr = 2e-5
batch_size = 16
weight_decay = 0
epochs = 100
load_model = False
load_model_file = "model.pth.tar"
transform = True

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x,y) in enumerate(loop):
        out = model(x)
        loss = loss_fn(out,y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update the progress bar
        loop.set_postfix(loss = loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

def main():
    model = Yolov1(split_size=15, num_boxes=2, num_classes=6)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = YoloLoss()

    dir_name = "cellData"
    file_name = "annotations.json"
    # 345 images
    dataset = CellDataset(annotation_file=os.path.join(dir_name, file_name),
                          transform=True, S=15)
    train_set, test_set = torch.utils.data.random_split(dataset, [275, 70])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(epochs):
        pred_boxes, target_boxes = get_bboxes(
            train_loader, model, iou_threshold=0.8, threshold=0.4
        )
        mean_avg_prec = mean_average_precision(pred_boxes, target_boxes, iou_threshold=0.8)
        print(f"Train mAP: {mean_avg_prec}")

if __name__ == "__main__":
    main()


