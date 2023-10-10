import numpy as np
import tensorflow as tf
from PIL import Image

import cv2

import os

import streamlit as st

from modules import preprocess_image
from modules import nn_model


def configure_streamlit(layout="centered", css_filepath=None):
    """ Function to customize the appearance of the Streamlit page
    Args:
        1) layout - the layout parameter of the st.set_page_config method
    Returns:
        None
    """
    st.set_page_config(layout=layout)
    if css_filepath is not None:
        with open(css_filepath) as file:
            st.markdown(f'<style>{file.read()}</style>', unsafe_allow_html=True)


@st.cache_resource
def load_nn_model(model_filepath):
    """ Function for loading a neural network model
        Args:
            1) model_filepath: model file path (including HDF5 file name)
        Returns:
            nn_model
    """
    model = nn_model.get_nn_model(model_filepath=model_filepath)
    return model


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


def main():
    configure_streamlit(layout="wide", css_filepath='styles/styles.css')
    nn_model.configure_tensorflow()
    garbage_classification_model = load_nn_model(model_filepath=
                                                 'models/6_resnet152_garbage_classification_6_classes_model.h5')
    target_size = (224, 224)
    classes_path = 'images/classes_images'
    class_labels = os.listdir(path=classes_path)

    # Streamlit
    st.header("Online Garbage Classifier")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, width=300, caption="Uploaded Image")

        pil_image = Image.open(uploaded_file)
        image_array = np.array(pil_image)
        _, x = preprocess_image.preprocess_resnet_image(img=image_array, target_size=target_size)

        predictions, predicted_class_index = nn_model.classify_image(model=garbage_classification_model, image=x)
        col1, col2, col3 = st.columns([0.38, 0.35, 0.27])
        with col2:
            st.subheader(f"Predicted class: {class_labels[predicted_class_index]}")
            st.subheader(f"Confidence: {predictions.max():.2f}")

    images_path = 'images/classes_images'
    st.markdown('<hr class="above-hr">', unsafe_allow_html=True)
    if st.checkbox("Display examples of garbage images for each class"):
        classes_images_dict = get_classes_images(images_path=images_path)
        for garbage_class_name in classes_images_dict.keys():
            with st.expander(garbage_class_name):
                plt_images = preprocess_image.get_plt_images(images=classes_images_dict[garbage_class_name])
                st.pyplot(plt_images)


if __name__ == '__main__':
    main()



