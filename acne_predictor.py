import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
import tensorflow as tf
import keras

print(tf.__version__)
print(keras.__version__)



import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "EfficientNet_B0_model.keras"
)

model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False
)



classes = [
    "Blackheads",
    "Cyst",
    "Papules",
    "Pustules",
    "Whiteheads"

]

def predict_image(face_crop):
    img = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB

    )
    img = Image.fromarray(img)
    img = img.resize((224 , 224))
    img = np.array(img)
    img = img/255.0
    img = np.expand_dims(img, axis = 0)
    pred = model.predict(img, verbose = 0)
    class_id = np.argmax(pred)
    confidence = np.max(pred)
    return {

        "acne_type" : classes[class_id],
        "acne_probability":float(confidence*100)
    }
