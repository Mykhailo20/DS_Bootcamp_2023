import tensorflow as tf
import os


def configure_tensorflow():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    configure_tensorflow()
    print('Hello, World!')
    model = tf.keras.models.load_model('models/6_resnet152_garbage_classification_6_classes_model.h5')
    print(f"type(model) = {type(model)}")

