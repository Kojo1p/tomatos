
const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("previewContainer");
const detectButton = document.getElementById("detectButton");
const statusText = document.getElementById("status");
const resultPanel = document.getElementById("resultPanel");
const totalText = document.getElementById("totalText");
const countList = document.getElementById("countList");
const detailBody = document.querySelector("#detailTable tbody");
const modeBadge = document.getElementById("modeBadge");

function showResults(result) {
    modeBadge.textContent = result.is_dummy ? "DUMMY" : "MODEL AI";
    modeBadge.classList.toggle("badge-model", !result.is_dummy);
    totalText.textContent =
        "Total objek terdeteksi: " + result.detections.length;

    countList.replaceChildren();
    for (const label in result.counts) {
        const li = document.createElement("li");
        li.textContent = label + ": " + result.counts[label];
        countList.appendChild(li);
    }

    detailBody.replaceChildren();
    result.detections.forEach(function (item, index) {
        const row = document.createElement("tr");
        const values = [
            index + 1,
            item.label,
            (item.confidence * 100).toFixed(1) + "%",
            item.det_confidence !== undefined
                ? (item.det_confidence * 100).toFixed(1) + "%"
                : "-"
        ];
        values.forEach(function (value) {
            const cell = document.createElement("td");
            cell.textContent = value;
            row.appendChild(cell);
        });
        detailBody.appendChild(row);
    });

    resultPanel.classList.remove("hidden");
}
   const BOX_COLORS = {
       "healthy": "#2a9d4a",
       "reject": "#f39c12",
       "ripe": "#d62828",
       "unripe": "#1d6fd1"
   };

let lastResult = null;

function clearBoxes() {
    previewContainer.querySelectorAll(".detection-box").forEach(function (box) {
        box.remove();
    });
}

function drawBoxes(result) {
    clearBoxes();

    const image = previewContainer.querySelector("img");
    if (!image || !result) {
        return;
    }

    const scaleX = image.clientWidth / result.image_width;
    const scaleY = image.clientHeight / result.image_height;

    result.detections.forEach(function (item) {
        const [x1, y1, x2, y2] = item.box;
        const color = BOX_COLORS[item.label.toLowerCase()] || "#6c757d";

        const box = document.createElement("div");
        box.className = "detection-box";
        box.style.left = (image.offsetLeft + x1 * scaleX) + "px";
        box.style.top = (image.offsetTop + y1 * scaleY) + "px";
        box.style.width = ((x2 - x1) * scaleX) + "px";
        box.style.height = ((y2 - y1) * scaleY) + "px";
        box.style.borderColor = color;

        const tag = document.createElement("span");
        tag.className = "detection-label";
        tag.style.background = color;
        tag.textContent =
            item.label + " " + (item.confidence * 100).toFixed(0) + "%";

        box.appendChild(tag);
        previewContainer.appendChild(box);
    });
}
window.addEventListener("resize", function () {
    if (lastResult) {
        drawBoxes(lastResult);
    }
});

imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    previewContainer.replaceChildren();
    resultPanel.classList.add("hidden");
    lastResult = null;
    detectButton.disabled = true;

    if (!file) {
        statusText.textContent = "Model AI belum diintegrasikan.";
        previewContainer.textContent =
            "Pratinjau gambar akan muncul di sini.";
        return;
    }

    if (!["image/jpeg", "image/png"].includes(file.type)) {
        statusText.textContent = "Pilih gambar JPG atau PNG.";
        imageInput.value = "";
        return;
    }

    const image = document.createElement("img");
    image.alt = "Pratinjau gambar tomat";
    image.src = URL.createObjectURL(file);

    previewContainer.appendChild(image);
    detectButton.disabled = false;

    statusText.textContent =
        "Gambar siap. Model AI belum diintegrasikan.";
});

detectButton.addEventListener("click", async function () {
    const file = imageInput.files[0];

    if (!file) {
        statusText.textContent = "Pilih gambar dulu.";
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    detectButton.disabled = true;
    statusText.textContent = "Mengunggah gambar...";

    try {
        const response = await fetch("/upload", {
            method: "POST",
            body: formData
        });
        const result = await response.json();
        if (result.success) {
               if (result.is_dummy) {
                   statusText.textContent =
                       result.message + " Terdeteksi " + result.detections.length +
                       " objek (hasil DUMMY, bukan dari model asli).";
               } else {
                   statusText.textContent =
                       "Selesai. Model mendeteksi " + result.detections.length +
                       " tomat (hasil estimasi model).";
               }
            showResults(result);
            lastResult = result;
            drawBoxes(result);
        } else {
            statusText.textContent = result.message;
        }
    } catch (error) {
        statusText.textContent = "Terjadi kesalahan di halaman: " + error.message;
        console.error(error);
    } finally {
        detectButton.disabled = false;
    }
});
const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

tabButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        if (button.dataset.target !== "tab-camera" && cameraRunning) {
            stopCamera();
        }
        tabButtons.forEach(function (b) { b.classList.remove("active"); });
        tabButtons.forEach(function (b) { b.classList.remove("active"); });
        tabPanels.forEach(function (p) { p.classList.add("hidden"); });

        button.classList.add("active");
        document.getElementById(button.dataset.target).classList.remove("hidden");

        // Gambar ulang kotak karena ukuran gambar bisa berubah saat tab disembunyikan
        if (button.dataset.target === "tab-upload" && lastResult) {
            drawBoxes(lastResult);
        }
    });
});
const cameraVideo = document.getElementById("cameraVideo");
const cameraOverlay = document.getElementById("cameraOverlay");
const cameraButton = document.getElementById("cameraButton");
const cameraStatus = document.getElementById("cameraStatus");
const cameraCounts = document.getElementById("cameraCounts");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");

let cameraStream = null;
let cameraRunning = false;

function wait(ms) {
    return new Promise(function (resolve) {
        setTimeout(resolve, ms);
    });
}

confSlider.addEventListener("input", function () {
    confValue.textContent = confSlider.value;
});

function drawCameraBoxes(result) {
    cameraOverlay.replaceChildren();
    const counts = {};

    result.detections.forEach(function (item) {
        const [x1, y1, x2, y2] = item.box;
        const color = BOX_COLORS[item.label.toLowerCase()] || "#6c757d";

        const box = document.createElement("div");
        box.className = "detection-box";
        box.style.left = (x1 / result.image_width * 100) + "%";
        box.style.top = (y1 / result.image_height * 100) + "%";
        box.style.width = ((x2 - x1) / result.image_width * 100) + "%";
        box.style.height = ((y2 - y1) / result.image_height * 100) + "%";
        box.style.borderColor = color;

        const tag = document.createElement("span");
        tag.className = "detection-label";
        tag.style.background = color;
        tag.textContent =
            item.label + " " + (item.confidence * 100).toFixed(0) + "%";

        box.appendChild(tag);
        cameraOverlay.appendChild(box);

        counts[item.label] = (counts[item.label] || 0) + 1;
    });

    const parts = Object.keys(counts).map(function (label) {
        return label + ": " + counts[label];
    });
    cameraCounts.textContent = parts.length
        ? "Terdeteksi: " + parts.join(", ")
        : "Tidak ada tomat terdeteksi.";
}

async function cameraLoop() {
    const captureCanvas = document.createElement("canvas");
    const context = captureCanvas.getContext("2d");

    while (cameraRunning) {
        const videoWidth = cameraVideo.videoWidth;
        const videoHeight = cameraVideo.videoHeight;

        if (!videoWidth || !videoHeight) {
            await wait(100);
            continue;
        }

        // Perkecil frame (maks. lebar 640) supaya pengiriman dan deteksi lebih cepat
        const scale = Math.min(1, 640 / videoWidth);
        captureCanvas.width = Math.round(videoWidth * scale);
        captureCanvas.height = Math.round(videoHeight * scale);
        context.drawImage(cameraVideo, 0, 0, captureCanvas.width, captureCanvas.height);

        const blob = await new Promise(function (resolve) {
            captureCanvas.toBlob(resolve, "image/jpeg", 0.8);
        });

        const formData = new FormData();
        formData.append("image", blob, "frame.jpg");
        formData.append("det_conf", confSlider.value / 100);

        const started = performance.now();

        try {
            const response = await fetch("/predict_frame", {
                method: "POST",
                body: formData
            });
            const result = await response.json();

            if (!cameraRunning) {
                break;
            }

            if (result.success) {
                drawCameraBoxes(result);
                const ms = Math.round(performance.now() - started);
                cameraStatus.textContent =
                    "Kamera aktif | " + ms + " ms per frame (sekitar " +
                    (1000 / ms).toFixed(1) + " deteksi per detik)";
            } else {
                cameraStatus.textContent = result.message;
                await wait(1000);
            }
        } catch (error) {
            cameraStatus.textContent = "Terjadi kesalahan: " + error.message;
            console.error(error);
            await wait(1000);
        }
    }
}

async function startCamera() {
    cameraButton.disabled = true;

    try {
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: { ideal: "environment" },
                width: { ideal: 1280 },
                height: { ideal: 720 }
            },
            audio: false
        });
    } catch (error) {
        cameraStatus.textContent =
            "Kamera tidak bisa dibuka: " + error.message +
            ". Cek izin kamera dan pastikan tidak dipakai aplikasi lain.";
        cameraButton.disabled = false;
        return;
    }

    cameraVideo.srcObject = cameraStream;
    cameraRunning = true;
    cameraButton.textContent = "Hentikan Kamera";
    cameraButton.disabled = false;
    cameraStatus.textContent =
        "Kamera aktif. Frame pertama agak lama karena model sedang dimuat.";
    cameraLoop();
}

function stopCamera() {
    cameraRunning = false;

    if (cameraStream) {
        cameraStream.getTracks().forEach(function (track) {
            track.stop();
        });
        cameraStream = null;
    }

    cameraVideo.srcObject = null;
    cameraOverlay.replaceChildren();
    cameraCounts.textContent = "";
    cameraButton.textContent = "Mulai Kamera";
    cameraStatus.textContent = "Kamera dimatikan.";
}

cameraButton.addEventListener("click", function () {
    if (cameraRunning) {
        stopCamera();
    } else {
        startCamera();
    }
});