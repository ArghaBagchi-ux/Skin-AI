from flask import Flask, render_template, request
import cv2
import os
import json

from acne_predictor import predict_image
#from skin_type_predictior import predict_skin_type
from skin_type_predictior import predict_skin_type
from skin_segmentation import segment_skin
from score_engine import calculate_skin_score
from recommendation_engine import get_recommendations

app = Flask(__name__)

UPLOAD_FOLDER="uploads"
app.config["UPLOAD_FOLDER"]=UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER,exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():

    print("Step 1")

    file = request.files["image"]

    print("Step 2")

    path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)

    file.save(path)

    print("Saved:", path)

    img = cv2.imread(path)

    if img is None:
        return "OpenCV could not read the uploaded image."

    print("Image shape:", img.shape)

    segmented = segment_skin(img)
    print("Segmentation OK")

    acne_result = predict_image(segmented)
    print("Acne prediction:", acne_result)

    skin_result = predict_skin_type(segmented)
    print("Skin prediction:", skin_result)

    score = calculate_skin_score(acne_result["acne_probability"])
    print("Score:", score)

    recommendation = get_recommendations(
        acne_result["acne_probability"],
        skin_result["skin_type"]
    )
    print("Recommendation:", recommendation)

    return render_template("result.html", report={
        "Acne Type": acne_result["acne_type"],
        "Probability": round(acne_result["acne_probability"], 2),
        "Skin Type": skin_result["skin_type"],
        "Skin Score": score,
        "Recommendation": recommendation
    })

    
import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)