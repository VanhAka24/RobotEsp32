document.getElementById("send-btn").addEventListener("click", function() {
    const userInput = document.getElementById("user-input").value;
    if (userInput.trim() === "") return;
    displayMessage(userInput, "user");

    fetch("/chat", {
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ message: userInput })
    })
    .then(response => response.json())
    .then(data => {
        displayMessage(data.response, "bot");
    })
    .catch(error => {
        console.error("Error:", error);
    });

    document.getElementById("user-input").value = "";
});

function displayMessage(message, sender) {
    const messageDiv = document.createElement("div");
    messageDiv.classList.add(sender);
    messageDiv.innerHTML = message;
    document.getElementById("chat-box").appendChild(messageDiv);

    document.getElementById("chat-box").scrollTop = document.getElementById("chat-box").scrollHeight;
}

document.getElementById("user-input").addEventListener("keypress", function(event) {
    if (event.key === "Enter") {
        event.preventDefault();
        document.getElementById("send-btn").click();
    }
});
