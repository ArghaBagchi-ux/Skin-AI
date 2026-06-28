import tensorflow as tf
import numpy as np
import cv2
from PIL import Image

# Load model
model = tf.keras.models.load_model(
    r"D:\SkinAI\model\slin_type_model.keras",
    compile=False
)

# Class names
classes = [
    "Dry",
    "Oily",
    "Normal"
]


def predict_skin_type(face_crop):
    """
    face_crop: OpenCV image (BGR format)
    Returns skin type prediction dictionary
    """

    # Convert BGR -> RGB
    img = cv2.cvtColor(
        face_crop,
        cv2.COLOR_BGR2RGB
    )

    # Convert to PIL Image
    img = Image.fromarray(img)

    # Resize to model input size
    img = img.resize((224, 224))

    # Convert back to NumPy
    img = np.array(img)

    # Normalize
    img = img / 255.0

    # Add batch dimension
    img = np.expand_dims(
        img,
        axis=0
    )

    # Prediction
    pred = model.predict(
        img,
        verbose=0
    )[0]

    return {
        "dry": float(pred[0] * 100),
        "oily": float(pred[1] * 100),
        "normal": float(pred[2] * 100),
        "skin_type": classes[np.argmax(pred)]
    }