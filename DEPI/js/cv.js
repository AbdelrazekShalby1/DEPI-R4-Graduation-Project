const API_BASE_URL = "";

const MODEL_ENDPOINTS = {
    brain_classification: "/api/cv/brain-classification/predict",
    brain: "/api/cv/brain-detection/predict",
    chest: "/api/cv/chest/predict",
    covid: "/api/cv/chest/predict",
};

function getCurrentPageKey() {
    const path = window.location.pathname;
    return path.substring(path.lastIndexOf("/") + 1).replace(".html", "");
}

function isHealthyLabel(label) {
    const normalized = String(label || "").toLowerCase();
    return normalized.includes("normal") || normalized.includes("no tumor");
}

const imageInput = document.getElementById("imageInput");
const previewImage = document.getElementById("previewImage");
let detectionCanvas = document.getElementById("detectionCanvas");
const previewBox = document.querySelector(".preview-box");
const uploadArea = document.querySelector(".upload-area");
let lastDetectionPayload = null;

if (!detectionCanvas && previewImage?.parentElement) {
    let previewFrame = previewImage.parentElement;
    if (!previewFrame.classList.contains("preview-frame")) {
        previewFrame = document.createElement("div");
        previewFrame.className = "preview-frame";
        previewImage.parentElement.insertBefore(previewFrame, previewImage);
        previewFrame.appendChild(previewImage);
    }

    detectionCanvas = document.createElement("canvas");
    detectionCanvas.id = "detectionCanvas";
    detectionCanvas.className = "detection-canvas";
    previewFrame.appendChild(detectionCanvas);
}

function clearDetectionOverlay() {
    if (!detectionCanvas) return;

    const context = detectionCanvas.getContext("2d");
    context.clearRect(0, 0, detectionCanvas.width, detectionCanvas.height);
}

function getTumorDetections(data) {
    const detections = Array.isArray(data?.detections) ? data.detections : [];

    return detections.filter((detection) => {
        const box = detection?.box;
        const label = detection?.class_name || detection?.label || detection?.prediction;

        return (
            Array.isArray(box) &&
            box.length === 4 &&
            box.every((value) => Number.isFinite(Number(value))) &&
            !isHealthyLabel(label)
        );
    });
}

function drawDetectionBoxes(data) {
    lastDetectionPayload = data || null;

    if (!detectionCanvas || !previewImage || !previewImage.complete || !previewImage.naturalWidth) {
        return;
    }

    const displayWidth = previewImage.clientWidth;
    const displayHeight = previewImage.clientHeight;
    if (!displayWidth || !displayHeight) {
        clearDetectionOverlay();
        return;
    }

    const pixelRatio = window.devicePixelRatio || 1;
    detectionCanvas.style.width = displayWidth + "px";
    detectionCanvas.style.height = displayHeight + "px";
    detectionCanvas.width = Math.round(displayWidth * pixelRatio);
    detectionCanvas.height = Math.round(displayHeight * pixelRatio);

    const context = detectionCanvas.getContext("2d");
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, displayWidth, displayHeight);

    const detections = getTumorDetections(data);
    if (!detections.length) return;

    const sourceWidth = Number(data?.image_size?.width) || previewImage.naturalWidth;
    const sourceHeight = Number(data?.image_size?.height) || previewImage.naturalHeight;
    const scaleX = displayWidth / sourceWidth;
    const scaleY = displayHeight / sourceHeight;
    const lineWidth = Math.max(2, Math.min(5, Math.round(displayWidth / 180)));
    const fontSize = Math.max(12, Math.min(18, Math.round(displayWidth / 34)));

    context.lineWidth = lineWidth;
    context.strokeStyle = "#ef4444";
    context.fillStyle = "#ef4444";
    context.font = "600 " + fontSize + "px Poppins, Arial, sans-serif";
    context.textBaseline = "top";

    detections.forEach((detection) => {
        const [rawX1, rawY1, rawX2, rawY2] = detection.box.map(Number);
        const x1 = Math.max(0, Math.min(displayWidth, rawX1 * scaleX));
        const y1 = Math.max(0, Math.min(displayHeight, rawY1 * scaleY));
        const x2 = Math.max(0, Math.min(displayWidth, rawX2 * scaleX));
        const y2 = Math.max(0, Math.min(displayHeight, rawY2 * scaleY));
        const width = Math.max(0, x2 - x1);
        const height = Math.max(0, y2 - y1);

        if (width < 2 || height < 2) return;

        context.strokeRect(x1, y1, width, height);

        const label = detection.class_name || "Tumor";
        const confidence =
            detection.confidence === null || detection.confidence === undefined
                ? ""
                : " " + Math.round(Number(detection.confidence)) + "%";
        const text = label + confidence;
        const paddingX = 7;
        const paddingY = 5;
        const textWidth = context.measureText(text).width;
        const labelWidth = textWidth + paddingX * 2;
        const labelHeight = fontSize + paddingY * 2;
        const labelX = Math.max(0, Math.min(x1, displayWidth - labelWidth));
        const labelY = y1 - labelHeight - 2 >= 0 ? y1 - labelHeight - 2 : y1 + 2;

        context.fillRect(labelX, labelY, labelWidth, labelHeight);
        context.fillStyle = "#ffffff";
        context.fillText(text, labelX + paddingX, labelY + paddingY);
        context.fillStyle = "#ef4444";
    });
}

if (previewImage) {
    previewImage.addEventListener("load", () => {
        drawDetectionBoxes(lastDetectionPayload);
    });
}

window.addEventListener("resize", () => {
    drawDetectionBoxes(lastDetectionPayload);
});

if (imageInput) {
    imageInput.addEventListener("change", function () {
        const file = this.files[0];
        if (!file) return;

        const reader = new FileReader();
        reader.onload = function (event) {
            lastDetectionPayload = null;
            clearDetectionOverlay();
            previewImage.src = event.target.result;
            previewBox.style.display = "block";
            uploadArea.style.display = "none";
        };
        reader.readAsDataURL(file);
    });
}

const analyzeBtn = document.querySelector(".analyze-btn");
const floatingChatBtn = document.querySelector(".chat-btn");

if (floatingChatBtn) {
    floatingChatBtn.addEventListener("click", () => {
        window.location.href = "nlp.html";
    });
}

if (analyzeBtn) {
    analyzeBtn.addEventListener("click", async () => {
        const result = document.getElementById("predictionResult");
        const probability = document.getElementById("predictionProbability");

        if (!imageInput.files.length) {
            alert("Please select an image first.");
            return;
        }

        const endpoint = MODEL_ENDPOINTS[getCurrentPageKey()];
        if (!endpoint) {
            result.innerHTML = "Model not connected";
            result.style.color = "#f59e0b";
            probability.innerHTML = "This clinic does not have a configured endpoint.";
            return;
        }

        result.innerHTML = "Analyzing...";
        result.style.color = "#2563eb";
        probability.innerHTML = "Please wait...";
        drawDetectionBoxes(null);

        const originalBtnText = analyzeBtn.innerHTML;
        analyzeBtn.disabled = true;
        analyzeBtn.innerHTML = "Analyzing...";

        try {
            const formData = new FormData();
            formData.append("file", imageInput.files[0]);

            const response = await fetch(API_BASE_URL + endpoint, {
                method: "POST",
                body: formData,
            });

            const data = await response.json().catch(() => ({}));
            if (!response.ok || data.error) {
                throw new Error(data.error || data.message || "Server error: " + response.status);
            }

            const predictedClass = data.predicted_class || data.prediction;
            const confidence = data.confidence ?? data.probability;

            result.innerHTML = predictedClass;
            result.style.color = isHealthyLabel(predictedClass) ? "#22c55e" : "#dc2626";
            probability.innerHTML =
                confidence === null || confidence === undefined
                    ? "Confidence : --"
                    : "Confidence : " + confidence + "%";
            drawDetectionBoxes(data);
        } catch (err) {
            result.innerHTML = "Error";
            result.style.color = "#dc2626";
            probability.innerHTML = err.message;
            drawDetectionBoxes(null);
        } finally {
            analyzeBtn.disabled = false;
            analyzeBtn.innerHTML = originalBtnText;
        }
    });
}
