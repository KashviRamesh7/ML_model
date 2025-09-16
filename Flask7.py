import os
import time
import cv2
import numpy as np
from PIL import Image
from flask import Flask, render_template_string, request
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from collections import Counter

# Setup
UPLOAD_FOLDER = os.path.join("static", "uploaded_images")
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
    decoded = decode_predictions(preds, top=1)[0]  # top-1 only
    return decoded[0][1], float(decoded[0][2]) * 100  # label, confidence

@app.route("/", methods=["GET", "POST"])
def index():
    results = []
    image_preview = None

    if request.method == "POST":
        # Upload from file
        if "file" in request.files and request.files["file"].filename != "":
            uploaded_file = request.files["file"]
            file_path = os.path.join(app.config["UPLOAD_FOLDER"], uploaded_file.filename)
            uploaded_file.save(file_path)

            img = Image.open(file_path).convert("RGB")
            label, conf = predict_image(img)
            image_preview = uploaded_file.filename

            results.append({
                "Filename": uploaded_file.filename,
                "Predicted Label": label,
                "Confidence (%)": round(conf, 2),
                "Scan Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "Source": "Uploaded"
            })

        # Capture from camera (multiple frames)
        elif "capture_camera" in request.form:
            cap = cv2.VideoCapture(0)
            predictions = []
            frames_captured = 5  # capture 5 frames

            for i in range(frames_captured):
                ret, frame = cap.read()
                if ret:
                    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    label, conf = predict_image(img)
                    predictions.append((label, conf))
                time.sleep(0.5)  # wait 0.5s between frames

            cap.release()

            if predictions:
                # Majority vote for label
                labels = [p[0] for p in predictions]
                most_common_label = Counter(labels).most_common(1)[0][0]

                # Average confidence for that label
                avg_conf = np.mean([p[1] for p in predictions if p[0] == most_common_label])

                # Save last frame as preview
                image_preview = "camera_snapshot.jpg"
                img.save(os.path.join(app.config["UPLOAD_FOLDER"], image_preview))

                results.append({
                    "Filename": "Camera Snapshot",
                    "Predicted Label": most_common_label,
                    "Confidence (%)": round(avg_conf, 2),
                    "Scan Time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "Source": "Camera (5-frame avg)"
                })

    # HTML UI
    html = """
    <html>
    <head>
        <style>
            body {
                font-family: Arial, sans-serif;
                text-align: center;
                background-color: #f0f2f5;
                margin: 0;
                padding: 0;
            }
            h1 {
                font-size: 42px;
                margin-top: 30px;
                color: #2c3e50;
            }
            h2 {
                margin-top: 40px;
                font-size: 28px;
                color: #34495e;
            }
            form {
                margin: 25px auto;
            }
            input[type="file"], input[type="submit"], button {
                font-size: 20px;
                padding: 12px 18px;
                margin-top: 15px;
                border-radius: 8px;
                border: none;
                cursor: pointer;
                background-color: #3498db;
                color: white;
            }
            input[type="file"] {
                background: none;
                color: black;
                border: 1px solid #ccc;
                cursor: pointer;
            }
            table {
                margin: 40px auto;
                border-collapse: collapse;
                width: 90%;
                font-size: 20px;
            }
            th, td {
                border: 1px solid #ccc;
                padding: 15px;
                text-align: center;
            }
            th {
                background-color: #2980b9;
                color: white;
            }
            td {
                background-color: #ecf0f1;
            }
            img {
                margin-top: 30px;
                max-width: 600px;
                border-radius: 16px;
                box-shadow: 0px 6px 15px rgba(0,0,0,0.3);
            }
        </style>
    </head>
    <body>
        <h1>🤖 Machine Recognition System (Flask)</h1>

        <h2>📤 Upload an Image</h2>
        <form method="POST" enctype="multipart/form-data">
            <input type="file" name="file" accept="image/*"><br>
            <input type="submit" value="Upload & Predict">
        </form>

        <h2>📸 Camera Scan</h2>
        <form method="POST">
            <button type="submit" name="capture_camera">📷 Capture from Camera</button>
        </form>

        {% if image_preview %}
            <h2>Preview</h2>
            <img src="{{ url_for('static', filename='uploaded_images/' + image_preview) }}">
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
    # Important for deployment
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
