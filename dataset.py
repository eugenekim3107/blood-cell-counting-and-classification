import torch
from PIL import Image
import json
from torch.utils.data import Dataset
import os
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# Image shape = (960, 1280, 3)
# 6 classes
# ['red blood cell', 'ring', 'gametocyte', 'schizont', 'trophozoite', 'difficult']

class VOCDataset(Dataset):
    def __init__(self, annotation_file, dir_name="cellData", S=7, B=2, C=6, img_h=960, img_w=1280, transform=None):
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
        image = Image.open(os.path.join(self.dir_name,image_name))
        boxes = torch.tensor(boxes)
        if self.transform:
            image, boxes = self.transform(image, boxes)

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

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, bboxes):
        for t in self.transforms:
            img, bboxes = t(img), bboxes

        return img, bboxes

def main():
    dir_name = "cellData"
    file_name = "annotations.json"
    # 345 images
    transform = Compose([transforms.Resize((960, 1280)), transforms.ToTensor()])
    dataset = VOCDataset(os.path.join(dir_name,file_name), transform=transform)
    batch_size = 30
    train_set, test_set = torch.utils.data.random_split(dataset, [275, 70])
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True)
    for (image,label) in train_loader:
        print(image.shape, label.shape)
        break

if __name__ == '__main__':
    main()

