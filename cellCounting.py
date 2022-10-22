import numpy as np
import scipy.ndimage
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from commonFunc import read_img, save_img


# Gradient of Image Function
def detect_edge(image, sobel=False):
    image_copy = image.copy()
    if sobel:
        Ix = scipy.ndimage.convolve(image_copy, np.array([[1,0,-1],[2,0,-2],[1,0,-1]]))
        Iy = scipy.ndimage.convolve(image_copy, np.array([[1,2,1],[0,0,0],[-1,-2,-1]]))
    else:
        Ix = scipy.ndimage.convolve(image_copy, np.array([[1,0,-1]]))
        Iy = scipy.ndimage.convolve(image_copy, np.array([[1,0,-1]]).T)
    grad_magnitude = np.sqrt((Ix**2) + (Iy**2))
    return Ix, Iy, grad_magnitude

# Harris Corner Detector
def harris_detector(image, alpha=0.05):
    image_copy = image.copy()
    x_prime = scipy.ndimage.sobel(image_copy, axis=0, mode = 'constant')
    y_prime = scipy.ndimage.sobel(image_copy, axis=1, mode = 'constant')
    x2_prime = scipy.ndimage.gaussian_filter(x_prime**2, sigma=0.85,mode = 'constant')
    xy_prime = scipy.ndimage.gaussian_filter(x_prime*y_prime, sigma=0.85, mode = 'constant')
    y2_prime = scipy.ndimage.gaussian_filter(y_prime**2, sigma=0.85, mode = 'constant')
    # determinant - (alpha * trace)
    output = ((x2_prime * y2_prime) - (xy_prime**2)) - alpha*((x2_prime+y2_prime)**2)
    return output

def main():
    folder = "cellData"
    path0 = os.path.join(folder, "PA171690.JPG")
    path1 = os.path.join(folder, "PA171690.JPG")
    path2 = os.path.join(folder, "PA171690.JPG")
    image0 = read_img(path0, grayscale=True)
    image1 = read_img(path1, grayscale=True)
    image2 = read_img(path2, grayscale=True)

    os.makedirs('gradientImages', exist_ok=True)

    # Save gradient images
    Ix = detect_edge(image0, sobel=False)[0]
    Iy = detect_edge(image0, sobel=False)[1]
    grad_mag = detect_edge(image0, sobel=False)[2]
    save_img(Ix, "gradientImages/IxOriginal.jpg")
    save_img(Iy, "gradientImages/IyOriginal.jpg")
    save_img(grad_mag, "gradientImages/gradOriginal.jpg")

    os.makedirs('cornerScore', exist_ok=True)

    # Corner Detection
    harris = harris_detector(image0)
    save_img(harris, "cornerScore/harrisCornerOriginal.jpg")

    #


if __name__ == '__main__':
    main()