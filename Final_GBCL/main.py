import numpy as np
import tensorflow as tf

import cv2

import os

import streamlit as st

from modules import preprocess_image


def configure_tensorflow():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    # configure_tensorflow()
    print('Hello, World!')
    st.header("Online Garbage Classifier")
    images_path = 'test_images'
    target_size = (224, 224)

    test_folders = os.listdir(images_path)
    with st.expander("images"):
        for folder in test_folders:
            folder_path = os.path.join(images_path, folder)
            image_names = os.listdir(folder_path)
            for image_name in image_names:
                image_path = os.path.join(folder_path, image_name)
                img = cv2.imread(image_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                plt_img = preprocess_image.get_plt_image(img=img, title=image_name)
                st.pyplot(plt_img)


