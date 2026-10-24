import os
import uuid
import traceback
from flask import Flask, render_template, request, jsonify
from PIL import Image
from detector import detect_tomatoes, detect_pil

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # maksimal 5 MB


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada file yang dikirim."}), 400

    file = request.files["image"]

    if file.filename == "":
        return jsonify({"success": False, "message": "Belum ada gambar yang dipilih."}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "message": "Format harus JPG atau PNG."}), 400

    extension = file.filename.rsplit(".", 1)[1].lower()
    new_name = f"{uuid.uuid4().hex}.{extension}"
    save_path = os.path.join(UPLOAD_FOLDER, new_name)
    file.save(save_path)

    try:
        detections, image_width, image_height, mode = detect_tomatoes(save_path)
    except Exception:
        traceback.print_exc()
        return jsonify({
            "success": False,
            "message": "Terjadi kesalahan saat deteksi. Cek terminal server."
        }), 500

    counts = {}
    for item in detections:
        counts[item["label"]] = counts.get(item["label"], 0) + 1

    return jsonify({
        "success": True,
        "message": "Gambar berhasil diunggah ke server.",
        "filename": new_name,
        "detections": detections,
        "counts": counts,
        "image_width": image_width,
        "image_height": image_height,
        "is_dummy": mode == "dummy"
    })
@app.route("/predict_frame", methods=["POST"])
def predict_frame():
    if "image" not in request.files:
        return jsonify({"success": False, "message": "Tidak ada frame yang dikirim."}), 400

    try:
        det_conf = float(request.form.get("det_conf", 0.6))
    except ValueError:
        det_conf = 0.6
    det_conf = min(max(det_conf, 0.05), 0.95)

    try:
        img = Image.open(request.files["image"].stream).convert("RGB")
        detections, image_width, image_height, mode = detect_pil(img, det_conf)
    except Exception:
        traceback.print_exc()
        return jsonify({"success": False, "message": "Gagal memproses frame."}), 500

    return jsonify({
        "success": True,
        "detections": detections,
        "image_width": image_width,
        "image_height": image_height,
        "is_dummy": mode == "dummy"
    })

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({"success": False, "message": "Ukuran gambar maksimal 5 MB."}), 413


if __name__ == "__main__":
    # Mode publik (tunnel/hosting) mematikan debug demi keamanan
    public_mode = os.environ.get("APP_PUBLIC") == "1"
    app.run(debug=not public_mode)