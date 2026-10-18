from flask import Flask, request, jsonify
import os
import sys

# Fix path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from predict_engine import load_model, predict_app, analyze_url

app = Flask(__name__)

# Fix emoji encoding issue (Windows)
sys.stdout.reconfigure(encoding='utf-8')

# ─────────────────────────────────────────────
# LOAD MODEL ON STARTUP
# ─────────────────────────────────────────────
print("Working Dir:", os.getcwd())
print("Files here:", os.listdir())

print("Loading ML Model...")
model, features = load_model()
print("ML Model loaded successfully!")


# ─────────────────────────────────────────────
# HOME ROUTE
# ─────────────────────────────────────────────
@app.route('/')
def home():
    return jsonify({
        "message": "Suraksha Backend Running",
        "endpoints": [
            "/scan/app (POST)",
            "/scan/url (POST)",
            "/health (GET)"
        ]
    })


# ─────────────────────────────────────────────
# HEALTH CHECK (IMPORTANT FOR TESTING)
# ─────────────────────────────────────────────
@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": True
    })


# ─────────────────────────────────────────────
# APP SCAN
# ─────────────────────────────────────────────
@app.route('/scan/app', methods=['POST'])
def scan_app():
    try:
        app_data = request.get_json()

        if not app_data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Normalize Android JSON keys
        normalized = {
            "package_name": app_data.get("package_name") or app_data.get("packageName", "unknown"),
            "app_name": app_data.get("app_name") or app_data.get("appName", "Unknown App"),
            "permissions": app_data.get("permissions", []),
            "api_calls": app_data.get("api_calls") or app_data.get("apiCalls", []),
            "intents": app_data.get("intents", [])
        }

        result = predict_app(model, features, normalized)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# URL SCAN
# ─────────────────────────────────────────────
@app.route('/scan/url', methods=['POST'])
def scan_url():
    try:
        data = request.get_json()

        if not data or 'url' not in data:
            return jsonify({"error": "Missing 'url' field"}), 400

        result = analyze_url(data['url'])
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────
# RUN SERVER (ONLY ONCE)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)