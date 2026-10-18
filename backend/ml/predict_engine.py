import pickle
import json
import re
import numpy as np
import os
from datetime import datetime
from typing import Dict, List

# ─────────────────────────────────────────────
# SYSTEM PACKAGE DETECTION
# ─────────────────────────────────────────────
SYSTEM_PACKAGE_PREFIXES = (
    "com.android.",
    "android.",
    "com.google.android.",
)

KNOWN_SAFE_PACKAGES = {
    "com.google.android.gms",
    "com.android.settings",
    "com.android.systemui",
    "com.android.phone",
    "com.android.contacts",
}

def is_system_package(package_name: str) -> bool:
    if package_name in KNOWN_SAFE_PACKAGES:
        return True
    return any(package_name.startswith(p) for p in SYSTEM_PACKAGE_PREFIXES)


# ─────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────
def load_model():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    model_path = os.path.join(base_dir, "malware_detector.pkl")
    features_path = os.path.join(base_dir, "selected_features.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(features_path, "r") as f:
        features = json.load(f)["features"]

    return model, features


# ─────────────────────────────────────────────
# FEATURE VECTOR
# ─────────────────────────────────────────────
def build_feature_vector(app_data, selected_features):

    permissions = set(p.split('.')[-1] for p in app_data.get("permissions", []))
    api_calls = set(app_data.get("api_calls", []))
    intents = set(app_data.get("intents", []))

    vector = []

    for feat in selected_features:
        if feat.startswith("PERM_"):
            vector.append(1 if feat[5:] in permissions else 0)

        elif feat.startswith("API_"):
            raw = feat[4:]
            dotted = raw.replace("_", ".")
            vector.append(1 if (raw in api_calls or dotted in api_calls) else 0)

        elif feat.startswith("INTENT_"):
            vector.append(1 if feat[7:] in intents else 0)

        else:
            vector.append(0)

    return np.array(vector).reshape(1, -1)


# ─────────────────────────────────────────────
# DANGER RULES
# ─────────────────────────────────────────────
HIGH_DANGER_COMBOS = [
    {"READ_SMS", "SEND_SMS"},
    {"CAMERA", "RECORD_AUDIO", "ACCESS_FINE_LOCATION"},
]

DANGEROUS_PERMS = {"SEND_SMS", "READ_SMS", "BIND_DEVICE_ADMIN"}
DANGEROUS_APIS = {"sendTextMessage", "getDeviceId", "exec"}


def check_danger_combos(permissions_set):
    return any(combo.issubset(permissions_set) for combo in HIGH_DANGER_COMBOS)


# ─────────────────────────────────────────────
# RISK ENGINE
# ─────────────────────────────────────────────
def get_risk_level(prob, is_system):

    if is_system:
        return "LOW"

    if prob < 0.55:
        return "LOW"
    elif prob < 0.80:
        return "MEDIUM"
    return "HIGH"


def get_risk_score(prob, is_system):
    score = int(prob * 100)
    if is_system:
        return min(score, 30)
    return score


# ─────────────────────────────────────────────
# MAIN PREDICTION
# ─────────────────────────────────────────────
def predict_app(model, selected_features, app_data):

    package = app_data.get("package_name", "unknown")
    is_system = is_system_package(package)

    X = build_feature_vector(app_data, selected_features)

    prob = float(model.predict_proba(X)[0][1])
    pred = model.predict(X)[0]

    permissions = set(p.split('.')[-1] for p in app_data.get("permissions", []))
    api_calls = set(app_data.get("api_calls", []))

    perm_count = len(permissions)

    risk_level = get_risk_level(prob, is_system)
    risk_score = get_risk_score(prob, is_system)

    # ─────────────────────────────
    # 🚨 HARD MALWARE OVERRIDE (FINAL FIX)
    # ─────────────────────────────
    if any(p in DANGEROUS_PERMS for p in permissions):
        risk_level = "HIGH"
        risk_score = max(risk_score, 85)

    if check_danger_combos(permissions):
        risk_level = "HIGH"
        risk_score = max(risk_score, 80)

    if any(api in api_calls for api in DANGEROUS_APIS):
        risk_level = "HIGH"
        risk_score = max(risk_score, 80)

    # ─────────────────────────────
    # SAFETY ADJUSTMENTS
    # ─────────────────────────────
    if not is_system:
        if perm_count <= 2 and prob < 0.75:
            risk_level = "LOW"
            risk_score = min(risk_score, 15)

    return {
        "package_name": package,
        "app_name": app_data.get("app_name", "Unknown"),
        "scan_timestamp": datetime.now().isoformat(),

        "classification": "MALWARE" if risk_level == "HIGH" else "BENIGN",
        "malware_probability": round(prob, 4),

        "risk_level": risk_level,
        "risk_score": risk_score,

        "is_system_app": is_system,
        "permissions_count": perm_count,
        "api_calls_count": len(api_calls),
        "intents_count": len(app_data.get("intents", [])),

        "flags": list(permissions.intersection(DANGEROUS_PERMS))
    }


# ─────────────────────────────────────────────
# URL SCANNER
# ─────────────────────────────────────────────
def analyze_url(url):

    score = 0
    signals = []

    patterns = [
        r'(\d{1,3}\.){3}\d{1,3}',
        r'@',
        r'login|verify|bank|secure|account',
        r'\.tk$|\.ml$|\.ga$',
        r'bit\.ly|tinyurl'
    ]

    for p in patterns:
        if re.search(p, url.lower()):
            score += 15
            signals.append("Suspicious pattern")

    if not url.startswith("https://"):
        score += 10
        signals.append("No HTTPS")

    score = min(score, 100)

    if score >= 65:
        level = "HIGH"
    elif score >= 35:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "url": url,
        "risk_level": level,
        "risk_score": score,
        "signals": signals,
        "is_safe": score < 35
    }