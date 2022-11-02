import torch
import os
import pandas as pd
from PIL import Image
import json

# Image shape = (960, 1280, 3)
# 6 classes
# ['red blood cell', 'ring', 'gametocyte', 'schizont', 'trophozoite', 'difficult']

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, annotation_file, S=7, B=2, C=20, transform=None):
        self.annotations = json.load(annotation_file)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.class_map = {'red blood cell':0, 'ring':1, 'gametocyte':2, 'schizont':3, 'trophozoite':4, 'difficult':5}

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        boxes=[]
        img = self.annotations[index]
        image_name = img["image_name"]
        for cell in img["objects"]:
            class_label = self.class_map[cell["type"]]
            x, y, width, height = int(cell["bbox"]["x"]), \
                                  int(cell["bbox"]["y"]), \
                                  int(cell["bbox"]["w"]), \
                                  int(cell["bbox"]["h"])
            boxes.append([class_label, x, y, width, height])

        image = Image.open(image_name)
        boxes = torch.tensor(boxes)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        


