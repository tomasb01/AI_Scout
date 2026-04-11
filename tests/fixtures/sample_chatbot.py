"""Sample chatbot backend for testing code analysis."""

import anthropic
from flask import Flask, request, jsonify

app = Flask(__name__)
client = anthropic.Anthropic()

SYSTEM_PROMPT = "You are Fleurdin, a helpful florist assistant specialized in essential oils and aromatherapy."


@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.json["message"]
    session_id = request.json.get("session_id", "default")

    history = load_history(session_id)

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        system=SYSTEM_PROMPT,
        messages=history + [{"role": "user", "content": user_message}],
        max_tokens=1024,
    )

    assistant_text = response.content[0].text
    save_to_db(session_id, user_message, assistant_text)

    return jsonify({"response": assistant_text})


def load_history(session_id):
    import sqlite3
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ?", (session_id,))
    return [{"role": r, "content": c} for r, c in cursor.fetchall()]


def save_to_db(session_id, user_msg, assistant_msg):
    import sqlite3
    conn = sqlite3.connect("chat.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                   (session_id, "user", user_msg))
    cursor.execute("INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                   (session_id, "assistant", assistant_msg))
    conn.commit()
