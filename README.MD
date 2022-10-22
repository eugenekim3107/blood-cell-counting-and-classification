
# Cell Counting and Classification

There are two main tasks in this project: Count the number of blood cells in a given image and classify them as healthy of infected. Standard computer vision methods will be used to count the number of cells and deep learning algorithms will be used to classify these cells.

## Roadmap

- Task 1: Count the number of blood cells

    - Image Preprocessing: Convolutional filters such as Gaussian filter, Sobel filter, etc.

    - Cell Counting: Laplacian of Gaussian (Difference of Gaussian) and possibly other methods.

    - Evaluation: Test method on multiple images and determine reliability to be used for Task 2

- Task 2: Classify blood cells into either healthy or infected

    - Image Preprocessing: Convolutional filters such as Gaussian filter, Sobel filter, etc. (maybe different from task 1)

    - Modeling: Experiment with different deep learning models and use the results of Task 1 as a feature to the model.

    - Evaluation: Test model with k-fold cross-validation and determine scores based on scoring metric.
## Task 1: Count the Number of Blood Cells

- Image Preprocessing

X Derivative Filter of Image
![Visual](https://i.postimg.cc/T3vZsrV6/Screen-Shot-2022-10-23-at-3-12-37-AM.png)

Gradient Magnitude of Image
![Visual](https://i.postimg.cc/jjc0PLJr/Screen-Shot-2022-10-23-at-3-12-25-AM.png)

- Cell Counting

Scale Space of Image (Difference of Gaussian)
![Visual](https://i.postimg.cc/dQyFSQbG/Screen-Shot-2022-10-23-at-3-13-18-AM.png)

Cell Detection
![Visual](https://i.postimg.cc/gJ5YLnf2/Screen-Shot-2022-10-23-at-3-13-28-AM.png)
