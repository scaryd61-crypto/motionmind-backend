import os
import requests
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)

allowed_origins = os.environ.get(
    "ALLOWED_ORIGINS",
    "https://scaryd61-crypto.github.io"
).split(",")

CORS(app, origins=[origin.strip() for origin in allowed_origins])

REPLICATE_API_URL = "https://api.replicate.com/v1"
MODEL_OWNER = "minimax"
MODEL_NAME = "video-01"


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "success": True,
        "message": "MotionMind AI Backend"
    })


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "status": "ok"
    })


@app.route("/generate", methods=["POST"])
def generate():
    token = os.environ.get("REPLICATE_API_TOKEN")

    if not token:
        return jsonify({
            "success": False,
            "error": "Backend API token is not configured."
        }), 500

    data = request.get_json(silent=True) or {}
    prompt = data.get("prompt", "")

    if not isinstance(prompt, str) or not prompt.strip():
        return jsonify({
            "success": False,
            "error": "Please provide a prompt."
        }), 400

    if len(prompt) > 10000:
        return jsonify({
            "success": False,
            "error": "Prompt is too long."
        }), 400

    try:
        response = requests.post(
            f"{REPLICATE_API_URL}/models/"
            f"{MODEL_OWNER}/{MODEL_NAME}/predictions",
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            },
            json={"input": {"prompt": prompt.strip()}},
            timeout=30
        )

        result = response.json()

        if not response.ok:
            return jsonify({
                "success": False,
                "error": result.get("detail", "Video request failed.")
            }), response.status_code

        return jsonify({
            "success": True,
            "id": result.get("id"),
            "status": result.get("status"),
            "urls": result.get("urls", {})
        }), 202

    except requests.RequestException:
        return jsonify({
            "success": False,
            "error": "Could not connect to the video service."
        }), 502


@app.route("/prediction/<prediction_id>", methods=["GET"])
def prediction(prediction_id):
    token = os.environ.get("REPLICATE_API_TOKEN")

    if not token:
        return jsonify({
            "success": False,
            "error": "Backend API token is not configured."
        }), 500

    try:
        response = requests.get(
            f"{REPLICATE_API_URL}/predictions/{prediction_id}",
            headers={"Authorization": f"Bearer {token}"},
            timeout=30
        )

        result = response.json()

        if not response.ok:
            return jsonify({
                "success": False,
                "error": "Could not retrieve prediction status."
            }), response.status_code

        return jsonify({
            "success": True,
            "id": result.get("id"),
            "status": result.get("status"),
            "output": result.get("output"),
            "error": result.get("error")
        })

    except requests.RequestException:
        return jsonify({
            "success": False,
            "error": "Could not connect to the video service."
        }), 502


AI_PROMPTS = {
    "thumbnail": (
        "You help YouTube creators. Give 5 short thumbnail headline options "
        "(max 40 characters each), one sub-text line for each, and 2 colour "
        "pairs with a short reason. Plain text only, numbered list."
    ),
    "analyze": (
        "You review a YouTube title, description and opening words. Give: "
        "1) hook strength, 2) title clarity, 3) keyword use, 4) three specific "
        "improvements. Be honest and short. Plain text only."
    ),
    "ideas": (
        "You help YouTube creators. Give 5 video ideas for the topic, each with "
        "a title, a one-line hook, difficulty (Easy, Medium or Hard) and one "
        "production tip. Plain text only."
    ),
}


@app.route("/ai", methods=["POST"])
def ai():
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        return jsonify({
            "success": False,
            "error": "AI is not configured on the server."
        }), 500

    data = request.get_json(silent=True) or {}
    task = data.get("task", "")
    text = data.get("text", "")

    if task not in AI_PROMPTS:
        return jsonify({
            "success": False,
            "error": "Unknown task."
        }), 400

    if not isinstance(text, str) or not text.strip():
        return jsonify({
            "success": False,
            "error": "Please type some text first."
        }), 400

    if len(text) > 4000:
        return jsonify({
            "success": False,
            "error": "Text is too long (max 4000 characters)."
        }), 400

    model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    try:
        response = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
            params={"key": api_key},
            headers={"Content-Type": "application/json"},
            json={
                "contents": [{
                    "parts": [{
                        "text": AI_PROMPTS[task] + "\n\nUser input:\n" + text.strip()
                    }]
                }]
            },
            timeout=30
        )

        if not response.ok:
            return jsonify({
                "success": False,
                "error": "The AI service could not answer right now. Try again later."
            }), 502

        result = response.json()
        reply = result["candidates"][0]["content"]["parts"][0]["text"]

        return jsonify({
            "success": True,
            "text": reply
        })

    except (requests.RequestException, KeyError, IndexError, ValueError):
        return jsonify({
            "success": False,
            "error": "The AI service could not answer right now. Try again later."
        }), 502


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
