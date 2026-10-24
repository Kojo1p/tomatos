// ===== Pengaturan (disamakan dengan detector.py) =====
const DET_IMGSZ = 640;
const CLS_IMGSZ = 224;
const CROP_MARGIN = 0.05;
const NMS_IOU = 0.7;
const MAX_BOXES = 30;
const UPLOAD_DET_CONF = 0.35;   // samakan dengan DET_CONF di detector.py
const CLASS_NAMES = ["healthy", "reject", "ripe", "unripe"];

const BOX_COLORS = {
    "healthy": "#2a9d4a",
    "reject": "#f39c12",
    "ripe": "#d62828",
    "unripe": "#1d6fd1"
};

// ===== Elemen halaman =====
const modelStatus = document.getElementById("modelStatus");
const imageInput = document.getElementById("imageInput");
const previewContainer = document.getElementById("previewContainer");
const detectButton = document.getElementById("detectButton");
const statusText = document.getElementById("status");
const resultPanel = document.getElementById("resultPanel");
const totalText = document.getElementById("totalText");
const countList = document.getElementById("countList");
const detailBody = document.querySelector("#detailTable tbody");

const cameraVideo = document.getElementById("cameraVideo");
const cameraOverlay = document.getElementById("cameraOverlay");
const cameraButton = document.getElementById("cameraButton");
const cameraStatus = document.getElementById("cameraStatus");
const cameraCounts = document.getElementById("cameraCounts");
const confSlider = document.getElementById("confSlider");
const confValue = document.getElementById("confValue");

const tabButtons = document.querySelectorAll(".tab-btn");
const tabPanels = document.querySelectorAll(".tab-panel");

let lastResult = null;
let cameraStream = null;
let cameraRunning = false;

function wait(ms) {
    return new Promise(function (resolve) {
        setTimeout(resolve, ms);
    });
}

function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
}

// ===== Memuat model =====
let detSession = null;
let clsSession = null;
let modelsPromise = null;

function loadModels() {
    if (typeof ort === "undefined") {
        modelStatus.textContent = "Gagal memuat onnxruntime-web. Cek koneksi internet.";
        return Promise.reject(new Error("onnxruntime-web tidak termuat"));
    }

    if (!modelsPromise) {
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
        ort.env.wasm.numThreads = 1;

        modelsPromise = (async function () {
            modelStatus.textContent = "Memuat model AI (sekitar 17 MB)...";
            const options = { executionProviders: ["wasm"] };
            detSession = await ort.InferenceSession.create("models/detektor_best.onnx", options);
            clsSession = await ort.InferenceSession.create("models/klasifikasi_best.onnx", options);
            modelStatus.textContent = "Model siap. Semua proses berjalan di perangkat ini.";
        })();

        modelsPromise.catch(function (error) {
            modelStatus.textContent = "Gagal memuat model: " + error.message;
            modelsPromise = null;
        });
    }
    return modelsPromise;
}

// ===== Pra-proses dan pasca-proses =====
const detCanvas = document.createElement("canvas");
const detCtx = detCanvas.getContext("2d", { willReadFrequently: true });
const clsCanvas = document.createElement("canvas");
const clsCtx = clsCanvas.getContext("2d", { willReadFrequently: true });

function canvasToTensorData(ctx, size) {
    const pixels = ctx.getImageData(0, 0, size, size).data;
    const area = size * size;
    const data = new Float32Array(3 * area);
    for (let i = 0; i < area; i++) {
        data[i] = pixels[i * 4] / 255;
        data[area + i] = pixels[i * 4 + 1] / 255;
        data[2 * area + i] = pixels[i * 4 + 2] / 255;
    }
    return data;
}

function iou(a, b) {
    const x1 = Math.max(a.x1, b.x1);
    const y1 = Math.max(a.y1, b.y1);
    const x2 = Math.min(a.x2, b.x2);
    const y2 = Math.min(a.y2, b.y2);
    const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
    const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
    const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
    return inter / (areaA + areaB - inter + 1e-9);
}

function nms(boxes, threshold) {
    boxes.sort(function (p, q) { return q.score - p.score; });
    const kept = [];
    for (const box of boxes) {
        let keep = true;
        for (const other of kept) {
            if (iou(box, other) > threshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            kept.push(box);
        }
        if (kept.length >= MAX_BOXES) {
            break;
        }
    }
    return kept;
}

function toProbabilities(values) {
    let sum = 0;
    let inRange = true;
    for (const v of values) {
        sum += v;
        if (v < 0 || v > 1) {
            inRange = false;
        }
    }
    if (inRange && Math.abs(sum - 1) < 0.01) {
        return Array.from(values);
    }
    const max = Math.max(...values);
    const exps = Array.from(values).map(function (v) { return Math.exp(v - max); });
    const total = exps.reduce(function (a, b) { return a + b; }, 0);
    return exps.map(function (v) { return v / total; });
}

// Tahap 1: cari lokasi tomat
async function detectBoxes(source, width, height, confThreshold) {
    const ratio = Math.min(DET_IMGSZ / width, DET_IMGSZ / height);
    const newW = Math.round(width * ratio);
    const newH = Math.round(height * ratio);
    const left = Math.round((DET_IMGSZ - newW) / 2 - 0.1);
    const top = Math.round((DET_IMGSZ - newH) / 2 - 0.1);

    detCanvas.width = DET_IMGSZ;
    detCanvas.height = DET_IMGSZ;
    detCtx.fillStyle = "rgb(114,114,114)";
    detCtx.fillRect(0, 0, DET_IMGSZ, DET_IMGSZ);
    detCtx.imageSmoothingEnabled = true;
    detCtx.imageSmoothingQuality = "high";
    detCtx.drawImage(source, 0, 0, width, height, left, top, newW, newH);

    const input = new ort.Tensor(
        "float32",
        canvasToTensorData(detCtx, DET_IMGSZ),
        [1, 3, DET_IMGSZ, DET_IMGSZ]
    );
    const feeds = {};
    feeds[detSession.inputNames[0]] = input;
    const outputs = await detSession.run(feeds);
    const output = outputs[detSession.outputNames[0]];

    const data = output.data;
    const count = output.dims[2];
    const candidates = [];

    for (let i = 0; i < count; i++) {
        const score = data[4 * count + i];
        if (score < confThreshold) {
            continue;
        }
        const cx = data[i];
        const cy = data[count + i];
        const w = data[2 * count + i];
        const h = data[3 * count + i];

        const x1 = clamp((cx - w / 2 - left) / ratio, 0, width);
        const y1 = clamp((cy - h / 2 - top) / ratio, 0, height);
        const x2 = clamp((cx + w / 2 - left) / ratio, 0, width);
        const y2 = clamp((cy + h / 2 - top) / ratio, 0, height);

        if (x2 - x1 < 1 || y2 - y1 < 1) {
            continue;
        }
        candidates.push({ x1: x1, y1: y1, x2: x2, y2: y2, score: score });
    }

    return nms(candidates, NMS_IOU);
}

// Tahap 2: klasifikasi potongan gambar
async function classifyCrop(source, width, height, box) {
    const mx = (box.x2 - box.x1) * CROP_MARGIN;
    const my = (box.y2 - box.y1) * CROP_MARGIN;
    const left = Math.max(0, Math.trunc(box.x1 - mx));
    const top = Math.max(0, Math.trunc(box.y1 - my));
    const right = Math.min(width, Math.trunc(box.x2 + mx));
    const bottom = Math.min(height, Math.trunc(box.y2 + my));
    const cw = right - left;
    const ch = bottom - top;

    if (cw < 16 || ch < 16) {
        return null;
    }

    // Perkecil sisi terpendek ke 224, lalu potong bagian tengah (seperti Ultralytics)
    const scale = CLS_IMGSZ / Math.min(cw, ch);
    const drawW = Math.round(cw * scale);
    const drawH = Math.round(ch * scale);
    const offX = Math.round((drawW - CLS_IMGSZ) / 2);
    const offY = Math.round((drawH - CLS_IMGSZ) / 2);

    clsCanvas.width = CLS_IMGSZ;
    clsCanvas.height = CLS_IMGSZ;
    clsCtx.imageSmoothingEnabled = true;
    clsCtx.imageSmoothingQuality = "high";
    clsCtx.drawImage(source, left, top, cw, ch, -offX, -offY, drawW, drawH);

    const input = new ort.Tensor(
        "float32",
        canvasToTensorData(clsCtx, CLS_IMGSZ),
        [1, 3, CLS_IMGSZ, CLS_IMGSZ]
    );
    const feeds = {};
    feeds[clsSession.inputNames[0]] = input;
    const outputs = await clsSession.run(feeds);
    const probs = toProbabilities(outputs[clsSession.outputNames[0]].data);

    let best = 0;
    for (let i = 1; i < probs.length; i++) {
        if (probs[i] > probs[best]) {
            best = i;
        }
    }
    return { label: CLASS_NAMES[best], confidence: probs[best] };
}

async function runDetection(source, width, height, confThreshold) {
    await loadModels();

    const boxes = await detectBoxes(source, width, height, confThreshold);
    const detections = [];

    for (const box of boxes) {
        const result = await classifyCrop(source, width, height, box);
        if (!result) {
            continue;
        }
        detections.push({
            label: result.label,
            confidence: result.confidence,
            det_confidence: box.score,
            box: [Math.trunc(box.x1), Math.trunc(box.y1), Math.trunc(box.x2), Math.trunc(box.y2)]
        });
    }

    return { detections: detections, image_width: width, image_height: height };
}

// ===== Tampilan hasil (tab Upload) =====
function showResults(result) {
    const counts = {};
    result.detections.forEach(function (item) {
        counts[item.label] = (counts[item.label] || 0) + 1;
    });

    totalText.textContent = "Total objek terdeteksi: " + result.detections.length;

    countList.replaceChildren();
    for (const label in counts) {
        const li = document.createElement("li");
        li.textContent = label + ": " + counts[label];
        countList.appendChild(li);
    }

    detailBody.replaceChildren();
    result.detections.forEach(function (item, index) {
        const row = document.createElement("tr");
        const values = [
            index + 1,
            item.label,
            (item.confidence * 100).toFixed(1) + "%",
            (item.det_confidence * 100).toFixed(1) + "%"
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

// ===== Upload gambar =====
imageInput.addEventListener("change", function () {
    const file = imageInput.files[0];

    previewContainer.replaceChildren();
    resultPanel.classList.add("hidden");
    lastResult = null;
    detectButton.disabled = true;

    if (!file) {
        statusText.textContent = "Pilih gambar untuk memulai.";
        previewContainer.textContent = "Pratinjau gambar akan muncul di sini.";
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
    statusText.textContent = "Gambar siap. Klik Deteksi Tomat.";
});

detectButton.addEventListener("click", async function () {
    const image = previewContainer.querySelector("img");

    if (!image) {
        statusText.textContent = "Pilih gambar dulu.";
        return;
    }

    detectButton.disabled = true;
    statusText.textContent = "Memproses gambar (pertama kali akan memuat model)...";

    try {
        const started = performance.now();
        const result = await runDetection(
            image, image.naturalWidth, image.naturalHeight, UPLOAD_DET_CONF
        );
        const ms = Math.round(performance.now() - started);

        showResults(result);
        lastResult = result;
        drawBoxes(result);

        statusText.textContent =
            "Selesai. Model mendeteksi " + result.detections.length +
            " tomat dalam " + ms + " ms (hasil estimasi model).";
    } catch (error) {
        statusText.textContent = "Terjadi kesalahan: " + error.message;
        console.error(error);
    } finally {
        detectButton.disabled = false;
    }
});

// ===== Kamera =====
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
    while (cameraRunning) {
        const videoWidth = cameraVideo.videoWidth;
        const videoHeight = cameraVideo.videoHeight;

        if (!videoWidth || !videoHeight) {
            await wait(100);
            continue;
        }

        const started = performance.now();

        try {
            const result = await runDetection(
                cameraVideo, videoWidth, videoHeight, confSlider.value / 100
            );

            if (!cameraRunning) {
                break;
            }

            drawCameraBoxes(result);
            const ms = Math.round(performance.now() - started);
            cameraStatus.textContent =
                "Kamera aktif | " + ms + " ms per frame (sekitar " +
                (1000 / ms).toFixed(1) + " deteksi per detik)";
        } catch (error) {
            cameraStatus.textContent = "Terjadi kesalahan: " + error.message;
            console.error(error);
            await wait(1000);
        }

        await wait(10);
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

// ===== Pindah tab =====
tabButtons.forEach(function (button) {
    button.addEventListener("click", function () {
        if (button.dataset.target !== "tab-camera" && cameraRunning) {
            stopCamera();
        }

        tabButtons.forEach(function (b) { b.classList.remove("active"); });
        tabPanels.forEach(function (p) { p.classList.add("hidden"); });

        button.classList.add("active");
        document.getElementById(button.dataset.target).classList.remove("hidden");

        if (button.dataset.target === "tab-upload" && lastResult) {
            drawBoxes(lastResult);
        }
    });
});