from PIL import Image
import numpy as np
from matplotlib import pyplot as plt

def read_img(path, grayscale=True):
    img = Image.open(path)
    if grayscale:
        img = img.convert('L')
    else:
        img = img.convert('RGB')
    return np.array(img) / 255


def save_img(img, path, normalize=True):
    if normalize:
        img = img - img.min()
        img = img / img.max()
    else:
        img = np.clip(img, 0., 1.)
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img)
    img.save(path)
