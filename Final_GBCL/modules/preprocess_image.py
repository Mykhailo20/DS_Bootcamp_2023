import numpy as np
from tensorflow.keras.applications.resnet50 import preprocess_input
import cv2


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
