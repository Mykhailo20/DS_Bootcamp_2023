import numpy as np
import tensorflow as tf
import os


def configure_tensorflow():
    """ Function to configure tensorflow for use in an application
    Returns:
         None
    """
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def get_nn_model(model_filepath):
    """ Function for loading a neural network model
    Args:
        1) model_filepath: model file path (including HDF5 file name)
    Returns:
        nn_model
    """
    try:
        nn_model = tf.keras.models.load_model(model_filepath)
        return nn_model
    except Exception as e:
        print(f"Error occurred while loading the model: {str(e)}")


def classify_image(model, image):
    """ Function for image classification using a neural network model
    Args:
        1) model - the model that will perform the classification
        2) image - an image that has been previously prepared for a model and needs to be classified
    Returns:
        Tuple containing:
        1) predictions - probabilities of the image belonging to a certain target class
        2) predicted_class_index - index of the target class to which this image belongs according to the model's
        prediction results
    """
    predictions = model.predict(image)
    predicted_class_index = np.argmax(predictions)
    return predictions, predicted_class_index
