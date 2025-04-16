from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import random
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/api/transform", methods=["POST"])
def transform():
    file = request.files.get("file")
    target_emotion = request.form.get("target_emotion")

    if not file or not target_emotion:
        return jsonify({"error": "Missing file or target_emotion"}), 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    detected_emotion = random.choice(["happy", "sad", "angry", "calm"])

    return jsonify(
        {
            "detected_emotion": detected_emotion,
            "file_url": f"http://127.0.0.1:5000/uploads/{filename}",
        }
    )


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


if __name__ == "__main__":
    app.run(debug=True)
