import numpy as np
import tensorflow as tf

import cv2

import os

import streamlit as st

from modules import preprocess_image


def configure_tensorflow():
    """ Function to configure tensorflow for use in an application
    Returns:
         None
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def configure_streamlit(layout="centered"):
    """ Function to customize the appearance of the Streamlit page
    Args:
        1) layout - the layout parameter of the st.set_page_config method
    Returns:
        None
    """
    st.set_page_config(layout=layout)


@st.cache_data
def get_classes_images(images_path):
    """ Function for reading images of each of the garbage classes
    Args:
        1) images_path: the path to the images folder (the folder that contains the subfolders of each class)
    Returns:
        class_images_dict - dictionary: key name of garbage class;
                                             value - list of array-like_images
    """
    class_images_dict = {}
    classes_folders = os.listdir(images_path)
    for folder in classes_folders:
        class_images_dict[folder] = []
        folder_path = os.path.join(images_path, folder)
        image_names = os.listdir(folder_path)
        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            img = cv2.imread(image_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            class_images_dict[folder].append(img)
    return class_images_dict


if __name__ == '__main__':
    # configure_tensorflow()
    configure_streamlit(layout="wide")
    st.header("Online Garbage Classifier")
    images_path = 'images/classes_images'
    target_size = (224, 224)

    test_folders = os.listdir(images_path)
    if st.checkbox("Display examples of garbage images for each class"):
        classes_images_dict = get_classes_images(images_path=images_path)
        for garbage_class_name in classes_images_dict.keys():
            with st.expander(garbage_class_name):
                plt_images = preprocess_image.get_plt_images(images=classes_images_dict[garbage_class_name])
                st.pyplot(plt_images)



