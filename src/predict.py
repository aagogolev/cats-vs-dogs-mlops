import tensorflow as tf
import numpy as np
from PIL import Image

def load_and_prep_image(image_path, img_shape=150):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.resize(img, [img_shape, img_shape])
    img = img/255.
    return img

def predict(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img = load_and_prep_image(image_path)
    pred = model.predict(tf.expand_dims(img, axis=0))
    return "Dog" if pred > 0.5 else "Cat"
