import numpy as np
import tensorflow as tf
import os

from modules import preprocess_image


def configure_tensorflow():
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


if __name__ == '__main__':
    configure_tensorflow()
    print('Hello, World!')
    model = tf.keras.models.load_model('models/6_resnet152_garbage_classification_6_classes_model.h5')
    print(f"type(model) = {type(model)}")
    source_path = 'test_images'
    target_size = (224, 224)
    class_labels = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

    test_folders = os.listdir(source_path)

    for folder in test_folders:
        folder_path = os.path.join(source_path, folder)
        image_names = os.listdir(folder_path)
        for image_name in image_names:
            image_path = os.path.join(folder_path, image_name)
            img, x = preprocess_image.preprocess_resnet_image(image_path=image_path, target_size=target_size)

            # Make predictions
            predictions = model.predict(x)
            predicted_class_index = np.argmax(predictions)
            predicted_class_label = class_labels[predicted_class_index]
            print(f'\nThe model predicts: {predicted_class_label}')

            image_title = f"{os.path.basename(image_path)}: {predicted_class_label}"
            print(f"image_title = {image_title}")


