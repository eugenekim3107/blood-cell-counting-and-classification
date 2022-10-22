# //  Created by Qazi Ammar Arshad on 16/07/2020.
# //  Copyright © 2020 Qazi Ammar Arshad. All rights reserved.
"""
This code first detect the cells in image and then check the accuracy against the ground truth.
"""

import cv2
import json

# replace these paths with yours
images_path = "/Users/qaziammararshad/Documents/MSCS/Malaria_project/IML_Malaria/"
annotation_path = "/Users/qaziammararshad/Documents/MSCS/Malaria_project/annotations.json"
save_annotated_img_path = "/Users/qaziammararshad/Documents/MSCS/Malaria_project/annotated/"

# %%
with open(annotation_path) as annotation_path:
    ground_truth = json.load(annotation_path)

# %%
# iterate through all images and find TF and FP.
for single_image_ground_truth in ground_truth:

    image_name = single_image_ground_truth['image_name']
    objects = single_image_ground_truth['objects']
    image = cv2.imread(images_path + image_name)

    for bbox in objects:
        cell_type = bbox['type']
        x = int(bbox['bbox']['x'])
        y = int(bbox['bbox']['y'])
        h = int(bbox['bbox']['h'])
        w = int(bbox['bbox']['w'])
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 5)

    cv2.imwrite(save_annotated_img_path + image_name, image)
