import os
import time
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Setup
UPLOAD_FOLDER = "uploaded_images"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model
model = MobileNetV2(weights="imagenet")

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Prediction function
def predict_image(img: Image.Image):
    img_resized = img.resize((224, 224))
    x = img_to_array(img_resized)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=3)[0]
    return decoded

# Home Page (UI)
@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    image_preview = None

    if request.method == "POST":
        if "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(file_path)

            img = Image.open(file_path).convert("RGB")
            decoded = predict_image(img)
            label, conf = decoded[0][1], float(decoded[0][2]) * 100
            image_preview = uploaded_file.filename

            results.append({
                "Filename": uploaded_file.filename,
                "Predicted Label": label,
                "Confidence (%)": round(conf, 2),
                "Scan Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Source": "Uploaded"
            })

        elif "camera" in request.form:
            cap = cv2.VideoCapture(0)
            ret, frame = cap.read()
            cap.release()

            if ret:
                img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                decoded = predict_image(img)
                label, conf = decoded[0][1], float(decoded[0][2]) * 100
                image_preview = "camera_snapshot.jpg"
                img.save(os.path.join(app.config["UPLOAD_FOLDER"], image_preview))

                results.append({
                    "Filename": "Camera Snapshot",
                    "Predicted Label": label,
                    "Confidence (%)": round(conf, 2),
                    "Scan Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Source": "Camera"
                })

    # HTML + CSS
    html = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f9f9f9;
            }
            h1 {
                font-size: 32px;
                margin-top: 20px;
                color: #333;
            }
            h2 {
                margin-top: 30px;
                color: #444;
            }
            form {
                margin: 20px auto;
            }
            input[type="file"], input[type="submit"], button {
                font-size: 18px;
                padding: 10px 15px;
                margin-top: 10px;
            }
            table {
                margin: 20px auto;
                border-collapse: collapse;
                width: 80%;
                font-size: 18px;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 12px;
                text-align: center;
            }
            th {
                background-color: #eee;
            }
            img {
                margin-top: 20px;
                max-width: 400px;
                border-radius: 12px;
                box-shadow: 0px 4px 10px rgba(0,0,0,0.2);
            }
        </style>
    </head>
    <body>
        <h1>Machine Recognition System (Flask)</h1>

        <h2>Upload an Image</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*"><br>
            <input type="submit" value="Upload & Predict">
        </form>

        <h2>Camera Scan</h2>
        <form method="POST">
            <button type="submit" name="camera">Capture from Camera</button>
        </form>

        {% if image_preview %}
            <h2>Preview</h2>
            <img src="{{ url_for('static', filename='../uploaded_images/' + image_preview) }}">
        {% endif %}

        {% if results %}
            <h2>Results</h2>
            <table>
                <tr>
                    <th>Filename</th>
                    <th>Predicted Label</th>
                    <th>Confidence (%)</th>
                    <th>Scan Time</th>
                    <th>Source</th>
                </tr>
                {% for row in results %}
                <tr>
                    <td>{{ row["Filename"] }}</td>
                    <td>{{ row["Predicted Label"] }}</td>
                    <td>{{ row["Confidence (%)"] }}</td>
                    <td>{{ row["Scan Time"] }}</td>
                    <td>{{ row["Source"] }}</td>
                </tr>
                {% endfor %}
            </table>
        {% endif %}
    </body>
    </html>
    """

    return render_template_string(html, results=results, image_preview=image_preview)


if __name__ == "__main__":
    app.run(debug=True)
