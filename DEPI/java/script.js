/*=====================================
            HOME PAGE
=====================================*/

// Machine Learning Clinic
const mlClinic = document.getElementById("mlClinic");

if (mlClinic) {
    mlClinic.addEventListener("click", () => {
        window.location.href = "ml.html";
    });
}

// Computer Vision Clinic
const cvClinic = document.getElementById("cvClinic");

if (cvClinic) {
    cvClinic.addEventListener("click", () => {
        window.location.href = "cv.html";
    });
}

// NLP Clinic
const nlpClinic = document.getElementById("nlpClinic");

if (nlpClinic) {
    nlpClinic.addEventListener("click", () => {
        window.location.href = "nlp.html";
    });
}

const floatingChatBtn = document.querySelector(".chat-btn");

if (floatingChatBtn) {
    floatingChatBtn.addEventListener("click", () => {
        window.location.href = "nlp.html";
    });
}


/*=====================================
        MACHINE LEARNING PAGE
=====================================*/

// Kidney
const kidneyCard = document.getElementById("kidneyCard");

if (kidneyCard) {
    kidneyCard.addEventListener("click", () => {
        window.location.href = "kidney.html";
    });
}

// Liver
const liverCard = document.getElementById("liverCard");

if (liverCard) {
    liverCard.addEventListener("click", () => {
        window.location.href = "liver.html";
    });
}

// Lung
const lungCard = document.getElementById("lungCard");

if (lungCard) {
    lungCard.addEventListener("click", () => {
        window.location.href = "lung.html";
    });
}

// Alzheimer's
const alzCard = document.getElementById("alzCard");

if (alzCard) {
    alzCard.addEventListener("click", () => {
        window.location.href = "alz.html";
    });
}

// Epilepsy
const epilepsyCard = document.getElementById("epilepsyCard");

if (epilepsyCard) {
    epilepsyCard.addEventListener("click", () => {
        window.location.href = "epilepsy.html";
    });
}


/*=====================================
        COMPUTER VISION PAGE
=====================================*/

// Brain Tumor Classification
const brainClassificationCard = document.getElementById("brainClassificationCard");

if (brainClassificationCard) {
    brainClassificationCard.addEventListener("click", () => {
        window.location.href = "brain_classification.html";
    });
}

// Brain Tumor detection
const brainCard = document.getElementById("brainCard");

if (brainCard) {
    brainCard.addEventListener("click", () => {
        window.location.href = "brain.html";
    });
}


// Chest X-Ray
const chestCard = document.getElementById("chestCard");

if (chestCard) {
    chestCard.addEventListener("click", () => {
        window.location.href = "chest.html";
    });
}
