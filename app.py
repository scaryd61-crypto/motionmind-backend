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


if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "5000")),
        debug=False
    )
