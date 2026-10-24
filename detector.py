import os
from PIL import Image, ImageOps

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DET_PATH = os.path.join(BASE_DIR, "models", "detektor_best.pt")
CLS_PATH = os.path.join(BASE_DIR, "models", "klasifikasi_best.pt")

# Pengaturan yang sama dengan notebook AI Engineer
DET_CONF = 0.35
DET_IMGSZ = 640
CLS_IMGSZ = 224
CROP_MARGIN = 0.05

_det_model = None
_cls_model = None


def models_available():
    return os.path.exists(DET_PATH) and os.path.exists(CLS_PATH)


def _load_models():
    global _det_model, _cls_model
    if _det_model is None or _cls_model is None:
        from ultralytics import YOLO
        _det_model = YOLO(DET_PATH)
        _cls_model = YOLO(CLS_PATH)
    return _det_model, _cls_model


def _crop(img, box, margin=CROP_MARGIN):
    x1, y1, x2, y2 = box
    w, h = img.size
    mx, my = (x2 - x1) * margin, (y2 - y1) * margin
    left, top = max(0, int(x1 - mx)), max(0, int(y1 - my))
    right, bottom = min(w, int(x2 + mx)), min(h, int(y2 + my))
    return img.crop((left, top, right, bottom))


def _detect_real(img, det_conf=None):
    """Tahap 1: deteksi lokasi tomat. Tahap 2: klasifikasi tiap potongan."""
    det, cls = _load_models()

    conf = DET_CONF if det_conf is None else det_conf
    result = det.predict(img, conf=conf, imgsz=DET_IMGSZ, verbose=False)[0]
    if len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy().tolist()
    det_scores = result.boxes.conf.cpu().numpy().tolist()

    crops, kept = [], []
    for i, box in enumerate(boxes):
        crop = _crop(img, box)
        if crop.size[0] >= 16 and crop.size[1] >= 16:
            crops.append(crop)
            kept.append(i)

    detections = []
    if crops:
        cls_results = cls.predict(crops, imgsz=CLS_IMGSZ, verbose=False)
        for i, r in zip(kept, cls_results):
            probs = r.probs.data.cpu().numpy()
            k = int(probs.argmax())
            x1, y1, x2, y2 = boxes[i]
            detections.append({
                "label": cls.names[k],
                "confidence": float(probs[k]),
                "det_confidence": float(det_scores[i]),
                "box": [int(x1), int(y1), int(x2), int(y2)],
            })
    return detections


def _detect_dummy(img):
    """DUMMY: hasil palsu untuk menguji alur aplikasi (dipakai jika model belum ada)."""
    width, height = img.size
    return [
        {
            "label": "Ripe",
            "confidence": 0.90,
            "box": [int(width * 0.10), int(height * 0.15),
                    int(width * 0.40), int(height * 0.60)],
        },
        {
            "label": "Unripe",
            "confidence": 0.85,
            "box": [int(width * 0.55), int(height * 0.25),
                    int(width * 0.85), int(height * 0.75)],
        },
    ]
def detect_pil(img, det_conf=None):
    """
    Deteksi dari objek gambar PIL (dipakai untuk frame kamera).
    Mengembalikan: (daftar deteksi, lebar, tinggi, mode)
    """
    img = img.convert("RGB")
    width, height = img.size

    if models_available():
        return _detect_real(img, det_conf), width, height, "model"
    return _detect_dummy(img), width, height, "dummy"


def detect_tomatoes(image_path, det_conf=None):
    """Deteksi dari file gambar di disk (dipakai untuk upload)."""
    with Image.open(image_path) as raw:
        img = ImageOps.exif_transpose(raw).convert("RGB")
    return detect_pil(img, det_conf)