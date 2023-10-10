import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2


def get_plt_image(img, title=None):
    """ Function to get a plt.figure that displays an array-like image
    Args:
        1) img - image array-like object
        2) title - the title that will be displayed above the image
    Returns:
        plt.figure object
    """
    fig, ax = plt.subplots()
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')
    return fig


def get_plt_images(images, titles=None, figsize=(12, 4)):
    """ Function to get a plt.figure that displays array-like images in a row
        Args:
            1) images - the list of array-like images
            2) titles - the list of titles that will be displayed above the images (len(titles) == len(images))
            3) figsize - the size (width, height) of each image
        Returns:
            plt.figure object
        """
    fig, axes = plt.subplots(1, len(images), figsize=figsize)
    if titles is not None:
        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].set_title(titles[i])
            axes[i].axis('off')
    else:
        for i in range(len(images)):
            axes[i].imshow(images[i])
            axes[i].axis('off')

    return fig


def preprocess_resnet_image(image_path, target_size=(224, 224)):
    """ Function for reading and preparing an image for use in the ResNet model
    Args:
        1) mage_path - image file path (including the name of the image)
        2) target_size - the size of the image used to train the model (the original image will be resized to this size)
    Returns:
        Tuple containing:
            1) image - the read image in the form of a numpy array;
            2) x - the image prepared for use in the model
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img,  cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return (img, x)


if __name__ == "__main__":
    print("Hello World from preprocess_image!")
