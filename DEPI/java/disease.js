const ML_ENDPOINTS = {
    kidney: "/api/ml/kidney/predict",
    liver: "/api/ml/liver/predict",
    lung: "/api/ml/lung/predict",
    alz: "/api/ml/alzheimer/predict",
    epilepsy: "/api/ml/epileptic/predict",
};

function getCurrentPageKey() {
    const path = window.location.pathname;
    return path.substring(path.lastIndexOf("/") + 1).replace(".html", "");
}

function slugify(value) {
    return String(value || "")
        .trim()
        .toLowerCase()
        .replace(/[^a-z0-9]+/g, "_")
        .replace(/^_+|_+$/g, "");
}

function getFieldKey(field) {
    if (field.name) return field.name;
    if (field.id) return field.id;

    const group = field.closest(".input-group");
    const label = group ? group.querySelector("label") : null;
    return slugify(label ? label.innerText : field.placeholder);
}

function getFieldValue(field) {
    const value = field.value.trim();
    if (field.tagName === "SELECT" && value.toLowerCase().startsWith("select ")) {
        return "";
    }
    return value;
}

function collectFormData(form) {
    const data = {};
    const fields = form.querySelectorAll("input, select, textarea");

    fields.forEach((field) => {
        const key = getFieldKey(field);
        if (!key) return;
        data[key] = getFieldValue(field);
    });

    return data;
}

function validateForm(form) {
    let valid = true;
    form.querySelectorAll("input, select, textarea").forEach((field) => {
        const value = getFieldValue(field);
        if (value === "") {
            valid = false;
            field.style.borderColor = "#ef4444";
        } else {
            field.style.borderColor = "#d1d5db";
        }
    });
    return valid;
}

function setResult(result, probability, text, color, probabilityText) {
    result.innerHTML = text;
    result.style.color = color;
    if (probability) {
        probability.innerHTML = probabilityText;
    }
}

function isHealthyPrediction(label) {
    const normalized = String(label || "").toLowerCase();
    return (
        normalized.includes("healthy") ||
        normalized.includes("normal") ||
        normalized.includes("no ") ||
        normalized.includes("low ")
    );
}

document.addEventListener("DOMContentLoaded", () => {
    const floatingChatBtn = document.querySelector(".chat-btn");
    if (floatingChatBtn) {
        floatingChatBtn.addEventListener("click", () => {
            window.location.href = "nlp.html";
        });
    }

    const form = document.querySelector("form");
    const predictBtn = document.querySelector(".predict-btn");
    const result = document.getElementById("predictionResult");
    const probability =
        document.getElementById("predictionProbability") ||
        document.getElementById("probability");

    if (!form || !predictBtn || !result) return;

    form.addEventListener("submit", async (event) => {
        event.preventDefault();

        if (!validateForm(form)) {
            alert("Please fill in all fields.");
            return;
        }

        const endpoint = ML_ENDPOINTS[getCurrentPageKey()];
        if (!endpoint) {
            setResult(
                result,
                probability,
                "Model not connected",
                "#f59e0b",
                "This page does not have a configured endpoint."
            );
            return;
        }

        const originalText = predictBtn.innerHTML;
        predictBtn.disabled = true;
        predictBtn.innerHTML = "Predicting...";
        setResult(result, probability, "Predicting...", "#2563eb", "Please wait...");

        try {
            const response = await fetch(endpoint, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(collectFormData(form)),
            });
            const data = await response.json().catch(() => ({}));

            if (!response.ok || data.error) {
                throw new Error(data.error || data.message || "Server error: " + response.status);
            }

            const prediction = data.prediction || data.predicted_class;
            const confidence = data.probability ?? data.confidence;
            const color = isHealthyPrediction(prediction) ? "#16a34a" : "#dc2626";
            const probabilityText =
                confidence === null || confidence === undefined
                    ? "Confidence : --"
                    : "Confidence : " + confidence + "%";

            setResult(result, probability, prediction, color, probabilityText);
        } catch (error) {
            setResult(result, probability, "Connection Error", "#dc2626", error.message);
        } finally {
            predictBtn.disabled = false;
            predictBtn.innerHTML = originalText;
        }
    });
});
