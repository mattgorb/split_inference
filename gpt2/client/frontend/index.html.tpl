<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AnjunaLabs</title>
    <style>
        body {
            font-family: "Segoe UI", Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 0;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            min-height: 100vh;
            background-color: #121212; /* Dark background */
            color: #e0e0e0; /* Light text */
        }
        #chat-container {
            width: 100%;
            max-width: 600px;
            height: 70vh;
            display: flex;
            flex-direction: column;
            gap: 10px;
            padding: 20px;
            background: #1e1e1e; /* Slightly lighter dark */
            border-radius: 10px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.3);
            overflow-y: auto;
            border: 1px solid #333; /* Subtle border */
        }
        .message {
            padding: 12px 16px;
            border-radius: 10px;
            margin: 5px 0;
            max-width: 80%;
            word-wrap: break-word;
        }
        .user-message {
            align-self: flex-end;
            background: #4caf50; /* Soft green for user */
            color: #fff;
        }
        .bot-message {
            align-self: flex-start;
            background: #333; /* Soft gray for bot */
            color: #e0e0e0;
        }
        #input-container {
            width: 100%;
            max-width: 600px;
            display: flex;
            gap: 10px;
            margin-top: 20px;
            padding: 0 20px;
        }
        #message-input {
            flex: 1;
            padding: 14px 16px;
            font-size: 16px;
            background: #1e1e1e; /* Dark input background */
            color: #e0e0e0; /* Light text */
            border: 1px solid #333;
            border-radius: 8px;
            outline: none;
            box-shadow: inset 0 1px 3px rgba(0, 0, 0, 0.3);
            transition: border-color 0.3s;
        }
        #message-input:focus {
            border-color: #777; /* Gray border on focus */
        }
        #send-button {
            padding: 14px 20px;
            background: #444; /* Neutral gray button */
            color: #fff;
            font-size: 16px;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: background 0.3s;
        }
        #send-button:hover {
            background: #555; /* Slightly lighter gray on hover */
        }
    </style>
</head>
<body>
    <div id="chat-container"></div>
    <div id="input-container">
        <input type="text" id="message-input" placeholder="Type your message here..." />
        <button id="send-button">Send</button>
    </div>

    <script>
        const chatContainer = document.getElementById('chat-container');
        const messageInput = document.getElementById('message-input');
        const sendButton = document.getElementById('send-button');

        const serverUrl = "${server_url}"; 
        const username = "${username}";
        const password = "${password}";

        function appendMessage(content, isUser = true) {
            const message = document.createElement('div');
            message.className = `message $${isUser ? 'user-message' : 'bot-message'}`;
            message.textContent = content;
            chatContainer.appendChild(message);
            chatContainer.scrollTop = chatContainer.scrollHeight;
        }

        async function sendMessage() {
            const userMessage = messageInput.value.trim();
            if (!userMessage) return;

            appendMessage(userMessage, true);
            messageInput.value = '';

            try {
                const data = {
                    prompt: userMessage,
                    max_length: 512,
                    temperature: 1,
                };

                const response = await fetch(serverUrl, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                        'Authorization': `Basic $${btoa(`$${username}:$${password}`)}`,
                    },
                    body: JSON.stringify(data),
                });

                if (response.ok) {
                    const jsonResponse = await response.json();
                    if (jsonResponse && jsonResponse.generated_text) {
                        appendMessage(jsonResponse.generated_text, false);
                    } else {
                        appendMessage('Error: Received an invalid or empty response from the server.', false);
                    }
                } else {
                    const errorText = await response.text();
                    appendMessage(`Error: Server responded with status $${response.status} - $${errorText}`, false);
                }
            } catch (error) {
                appendMessage(`Error: Network error or server is unreachable. Details: $${error.message}`, false);
            }
        }

        sendButton.addEventListener('click', sendMessage);
        messageInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') sendMessage();
        });
    </script>
</body>
</html>
