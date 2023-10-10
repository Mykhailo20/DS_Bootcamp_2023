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
