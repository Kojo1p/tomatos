import time

import cv2
from PIL import Image

from detector import models_available, _detect_real

# Warna kotak (format BGR milik OpenCV)
COLORS = {
    "healthy": (74, 157, 42),
    "reject": (18, 156, 243),
    "ripe": (40, 40, 214),
    "unripe": (209, 111, 29),
}


def main():
    if not models_available():
        print("File model tidak ditemukan di folder models.")
        return

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Kamera tidak bisa dibuka.")
        print("Tutup aplikasi lain yang memakai kamera, lalu cek izin kamera Windows.")
        return

    print("Kamera terbuka. Frame pertama agak lama karena model sedang dimuat.")

    detections = []
    frame_count = 0
    fps = 0.0
    last_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Gagal membaca gambar dari kamera.")
            break

        # Deteksi dilakukan tiap 3 frame supaya tampilan tidak berat
        if frame_count % 3 == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            detections = _detect_real(Image.fromarray(rgb))
        frame_count += 1

        for item in detections:
            x1, y1, x2, y2 = item["box"]
            color = COLORS.get(item["label"].lower(), (128, 128, 128))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            text = f'{item["label"]} {item["confidence"] * 100:.0f}%'
            cv2.putText(frame, text, (x1, max(15, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - last_time, 1e-6))
        last_time = now

        info = f"FPS: {fps:.1f} | Tomat: {len(detections)} | Q = keluar"
        cv2.putText(frame, info, (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Tes Kamera Tomat", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()