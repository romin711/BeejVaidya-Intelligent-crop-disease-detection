const API_URL = "http://127.0.0.1:8000/predict";

const imageInput = document.getElementById("imageInput");
const imagePreview = document.getElementById("imagePreview");
const previewPlaceholder = document.getElementById("previewPlaceholder");
const detectBtn = document.getElementById("detectBtn");
const statusMessage = document.getElementById("statusMessage");
const resultCard = document.getElementById("resultCard");

const predictionEl = document.getElementById("prediction");
const confidenceEl = document.getElementById("confidence");
const confidenceLevelEl = document.getElementById("confidenceLevel");
const treatmentEl = document.getElementById("treatment");
const preventionEl = document.getElementById("prevention");
const explanationEl = document.getElementById("explanation");
const processingTimeEl = document.getElementById("processingTime");

let selectedFile = null;


/* IMAGE UPLOAD + PREVIEW */

imageInput.addEventListener("change", function (event) {
    const file = event.target.files[0];

    if (!file) {
        selectedFile = null;
        imagePreview.style.display = "none";
        previewPlaceholder.style.display = "block";
        return;
    }

    selectedFile = file;

    const reader = new FileReader();
    reader.onload = function (e) {
        imagePreview.src = e.target.result;
        imagePreview.style.display = "block";
        previewPlaceholder.style.display = "none";
    };

    reader.readAsDataURL(file);
});


/* DETECT BUTTON CLICK */

detectBtn.addEventListener("click", async function () {

    if (!selectedFile) {
        statusMessage.textContent = "Please upload an image first.";
        return;
    }

    statusMessage.textContent = "Detecting disease...";
    detectBtn.disabled = true;
    resultCard.style.display = "none";

    try {

        const formData = new FormData();
        formData.append("file", selectedFile);

        const response = await fetch(API_URL, {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText);
        }

        const data = await response.json();

        console.log("API response:", data);


        /* DISPLAY RESULTS */

        predictionEl.textContent = data.prediction || "-";

        confidenceEl.textContent =
            typeof data.confidence === "number"
                ? data.confidence.toFixed(3)
                : "-";

        confidenceLevelEl.textContent = data.confidence_level || "-";

        treatmentEl.textContent = data.treatment || "-";

        preventionEl.textContent = data.prevention || "-";

        explanationEl.textContent =
            data.explanation || data.message || "-";

        processingTimeEl.textContent =
            typeof data.processing_time_ms === "number"
                ? data.processing_time_ms + " ms"
                : "-";


        resultCard.style.display = "block";
        statusMessage.textContent = "Prediction completed.";

    } catch (error) {

        console.error("Error:", error);
        statusMessage.textContent = "Error: " + error.message;

    } finally {

        detectBtn.disabled = false;

    }
});