import os
from datetime import datetime
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import cv2

# ---------- Configuration ----------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
OUTPUT_FOLDER = os.path.join(BASE_DIR, "static", "outputs")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["OUTPUT_FOLDER"] = OUTPUT_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB

# Haar cascade (prefer built-in; fallback to models/)
CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    CASCADE_PATH = os.path.join(BASE_DIR, "models", "haarcascade_frontalface_default.xml")

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXT

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/detect", methods=["POST"])
def detect():
    if "image" not in request.files:
        return "No file part", 400

    file = request.files["image"]
    if file.filename == "":
        return "No selected file", 400

    if not allowed_file(file.filename):
        return "File type not allowed", 400

    filename = secure_filename(file.filename)
    name, ext = os.path.splitext(filename)
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
    save_name = f"{name}_{timestamp}{ext}"
    upload_path = os.path.join(app.config["UPLOAD_FOLDER"], save_name)
    file.save(upload_path)

    # Read image with OpenCV
    img = cv2.imread(upload_path)
    if img is None:
        return "Failed to read image", 500

    # Optional: resize large images for speed (keeps aspect ratio)
    max_dim = 1200
    h, w = img.shape[:2]
    scale = min(max_dim / max(h, w), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Save output
    out_name = f"out_{save_name}"
    out_path = os.path.join(app.config["OUTPUT_FOLDER"], out_name)
    cv2.imwrite(out_path, img)

    return render_template("result.html",
                           original_img=url_for("static", filename=f"uploads/{save_name}"),
                           output_img=url_for("static", filename=f"outputs/{out_name}"),
                           face_count=len(faces))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
