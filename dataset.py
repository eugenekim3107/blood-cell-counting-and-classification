import torch
import json
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import cv2
import numpy as np

# Image shape = (960, 1280, 3)
# 6 classes
# ['red blood cell', 'ring', 'gametocyte', 'schizont', 'trophozoite', 'difficult']

class CellDataset(Dataset):
    def __init__(self, annotation_file, transform, S=7, B=2, C=6, dir_name="cellData", img_h=960, img_w=1280):
        with open(annotation_file) as file:
            self.annotations = json.load(file)
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C
        self.dir_name = dir_name
        self.img_h = img_h
        self.img_w = img_w
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
        image = cv2.imread(os.path.join(self.dir_name,image_name), cv2.IMREAD_COLOR)
        boxes = torch.tensor(boxes)

        if self.transform:
            image = image.reshape((960, 1280, 3))
            image = (image - np.min(image))/ np.max(image)
            image = torch.from_numpy(image)

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))

        for box in boxes:
            class_label, x, y, width, height = box.tolist()
            class_label = int(class_label)
            x_norm = (x + (width / 2)) / self.img_w
            y_norm = (y + (height / 2)) / self.img_h
            i = int(self.S * y_norm)
            j = int(self.S * x_norm)
            x_cell = self.S * x_norm - j
            y_cell = self.S * y_norm - i

            if label_matrix[i,j,self.C] == 0:
                label_matrix[i,j,self.C] = 1
                box_coord = torch.tensor(
                    [x_cell, y_cell, width, height]
                )
                label_matrix[i, j, self.C+1:self.C+5] = box_coord
                label_matrix[i, j, class_label] = 1
        return image, label_matrix

def main():
    dir_name = "cellData"
    file_name = "annotations.json"
    # 345 images
    dataset = CellDataset(annotation_file=os.path.join(dir_name,file_name),transform=True, S=15)
    batch_size = 1
    train_set, test_set = torch.utils.data.random_split(dataset, [275, 70])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True)
    for (image,label) in train_loader:
        img = image[0].numpy()
        img = (img * 255).astype(np.uint8)
        lab = label[0]
        height_cell = 960 / 15
        width_cell = 1280 / 15
        for i in range(15):
            for j in range(15):
                bbox = lab[i,j,6:11]
                if bbox[0] == 1:
                    temp_height = height_cell * i
                    temp_width = width_cell * j
                    x = bbox[1] * (1280 / 15)
                    y = bbox[2] * (960 / 15)
                    please_x = int(temp_width + x)
                    please_y = int(temp_height + y)
                    left_top = (int(please_x - (bbox[3]/2)), int(please_y - (bbox[4]/2)))
                    right_bottom = (int(please_x + (bbox[3]/2)), int(please_y + (bbox[4]/2)))
                    cv2.circle(img, (please_x, please_y), radius=3, color=(0, 0, 255), thickness=4)
                    cv2.rectangle(img, left_top, right_bottom, (0, 255, 0),
                             thickness=2)
        cv2.imwrite("please.jpg", img)
        break

if __name__ == '__main__':
    main()

