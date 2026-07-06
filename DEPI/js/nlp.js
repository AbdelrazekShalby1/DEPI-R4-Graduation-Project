const CHATBOT_API_URL = "";

let sessionId = null;
let isSending = false;

const chatBody = document.getElementById("chatBody");
const userInput = document.getElementById("userInput");
const sendBtn = document.getElementById("sendBtn");
const suggestionBtns = document.querySelectorAll(".suggestion-btn");
const floatingChatBtn = document.querySelector(".chat-btn");

if (floatingChatBtn) {
    floatingChatBtn.addEventListener("click", () => {
        document.querySelector(".chat-container")?.scrollIntoView({ behavior: "smooth" });
        userInput?.focus();
    });
}

function getTime() {
    const now = new Date();
    return now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function addUserMessage(message) {
    const html = `
    <div class="message">
        <div class="avatar">
            <i class="fa-solid fa-user"></i>
        </div>
        <div class="bubble">
            ${message}
            <div class="time">${getTime()}</div>
        </div>
    </div>
    `;

    chatBody.insertAdjacentHTML("beforeend", html);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function addAIMessage(message) {
    const formatted = (message || "").replace(/\n/g, "<br>");

    const html = `
    <div class="message ai-message">
        <div class="avatar">
            <i class="fa-solid fa-robot"></i>
        </div>
        <div class="bubble">
            ${formatted}
            <div class="time">${getTime()}</div>
        </div>
    </div>
    `;

    chatBody.insertAdjacentHTML("beforeend", html);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function showTyping() {
    const html = `
    <div class="message ai-message" id="typingIndicator">
        <div class="avatar">
            <i class="fa-solid fa-robot"></i>
        </div>
        <div class="bubble">
            <i class="fa-solid fa-ellipsis fa-fade"></i>
        </div>
    </div>
    `;
    chatBody.insertAdjacentHTML("beforeend", html);
    chatBody.scrollTop = chatBody.scrollHeight;
}

function removeTyping() {
    const el = document.getElementById("typingIndicator");
    if (el) el.remove();
}

async function ensureSession() {
    if (sessionId) return sessionId;

    const formData = new FormData();
    formData.append("user_id", "");

    const response = await fetch(CHATBOT_API_URL + "/chat/start", {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error("Failed to start session: " + response.status);
    }

    const data = await response.json();
    sessionId = data.session_id;
    return sessionId;
}

async function sendMessage() {
    const message = userInput.value.trim();
    if (message === "" || isSending) return;

    addUserMessage(message);
    userInput.value = "";

    isSending = true;
    sendBtn.disabled = true;
    showTyping();

    try {
        await ensureSession();

        const formData = new FormData();
        formData.append("session_id", sessionId);
        formData.append("message", message);

        const response = await fetch(CHATBOT_API_URL + "/chat/message", {
            method: "POST",
            body: formData,
        });

        removeTyping();

        if (!response.ok) {
            throw new Error("Server error: " + response.status);
        }

        const data = await response.json();
        addAIMessage(data.text);
    } catch (err) {
        removeTyping();
        addAIMessage("Connection error: " + err.message);
    } finally {
        isSending = false;
        sendBtn.disabled = false;
    }
}

if (sendBtn) {
    sendBtn.addEventListener("click", sendMessage);
}

if (userInput) {
    userInput.addEventListener("keypress", function (event) {
        if (event.key === "Enter") {
            sendMessage();
        }
    });
}

suggestionBtns.forEach((button) => {
    button.addEventListener("click", () => {
        userInput.value = button.innerText;
        sendMessage();
    });
});
