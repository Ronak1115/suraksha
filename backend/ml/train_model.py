"""
============================================================
  Android Malware Detection - ML Model Training
  Dataset: Drebin-style feature set (Permissions + API + Intents)
  Algorithm: Random Forest Classifier
  Author: AI Security System
============================================================
"""

import numpy as np
import pandas as pd
import pickle
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, roc_auc_score
)
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
#  STEP 1: Define Feature Space (Drebin-style)
# ─────────────────────────────────────────────

DANGEROUS_PERMISSIONS = [
    "READ_CONTACTS", "WRITE_CONTACTS", "READ_CALL_LOG", "WRITE_CALL_LOG",
    "READ_SMS", "SEND_SMS", "RECEIVE_SMS", "READ_PHONE_STATE",
    "CALL_PHONE", "PROCESS_OUTGOING_CALLS", "CAMERA",
    "RECORD_AUDIO", "ACCESS_FINE_LOCATION", "ACCESS_COARSE_LOCATION",
    "READ_EXTERNAL_STORAGE", "WRITE_EXTERNAL_STORAGE",
    "GET_ACCOUNTS", "USE_CREDENTIALS", "MANAGE_ACCOUNTS",
    "INSTALL_PACKAGES", "DELETE_PACKAGES", "MOUNT_UNMOUNT_FILESYSTEMS",
    "CHANGE_WIFI_STATE", "INTERNET", "RECEIVE_BOOT_COMPLETED",
    "VIBRATE", "WAKE_LOCK", "FOREGROUND_SERVICE",
    "SYSTEM_ALERT_WINDOW", "BIND_DEVICE_ADMIN",
    "READ_LOGS", "DUMP", "BATTERY_STATS",
    "ACCESS_NETWORK_STATE", "CHANGE_NETWORK_STATE",
    "BLUETOOTH", "BLUETOOTH_ADMIN", "NFC",
    "BIND_ACCESSIBILITY_SERVICE", "BIND_INPUT_METHOD",
    "DISABLE_KEYGUARD", "EXPAND_STATUS_BAR",
]

SUSPICIOUS_API_CALLS = [
    "getDeviceId", "getSubscriberId", "getSimSerialNumber",
    "getLine1Number", "getVoiceMailNumber",
    "sendTextMessage", "sendMultipartTextMessage",
    "getLastKnownLocation", "requestLocationUpdates",
    "getCellLocation", "getNeighboringCellInfo",
    "execShellCommand", "Runtime.exec", "ProcessBuilder",
    "DexClassLoader", "loadClass", "invoke",
    "getContentResolver", "query", "insert",
    "getSystemService", "PackageManager",
    "TelephonyManager", "SmsManager",
    "HttpURLConnection", "URL.openConnection",
    "Socket", "ServerSocket",
    "cipher", "encrypt", "decrypt",
    "ContentResolver", "AccountManager",
    "setWifiEnabled", "startActivity",
    "sendBroadcast", "startService",
    "abortBroadcast", "setComponentEnabledSetting",
    "getInstalledPackages", "getRunningAppProcesses",
    "killBackgroundProcesses", "forceStopPackage",
]

SUSPICIOUS_INTENTS = [
    "BOOT_COMPLETED", "ACTION_PACKAGE_ADDED",
    "ACTION_PACKAGE_REMOVED", "ACTION_PACKAGE_REPLACED",
    "SMS_RECEIVED", "NEW_OUTGOING_CALL",
    "ACTION_SEND", "ACTION_VIEW",
    "CALL_BUTTON", "ACTION_CALL",
    "ACTION_SCREEN_ON", "ACTION_SCREEN_OFF",
    "ACTION_POWER_CONNECTED", "ACTION_POWER_DISCONNECTED",
    "ACTION_BATTERY_LOW", "PHONE_STATE",
    "DATA_SMS_RECEIVED", "WAP_PUSH_RECEIVED",
    "QUICKBOOT_POWERON", "LOCKED_BOOT_COMPLETED",
    "ACTION_MEDIA_MOUNTED",
]

ALL_FEATURES = (
    [f"PERM_{p}" for p in DANGEROUS_PERMISSIONS] +
    [f"API_{a.replace('.', '_')}" for a in SUSPICIOUS_API_CALLS] +
    [f"INTENT_{i}" for i in SUSPICIOUS_INTENTS]
)

print(f"[INFO] Total features defined: {len(ALL_FEATURES)}")
print(f"       - Permissions : {len(DANGEROUS_PERMISSIONS)}")
print(f"       - API Calls   : {len(SUSPICIOUS_API_CALLS)}")
print(f"       - Intents     : {len(SUSPICIOUS_INTENTS)}")


# ─────────────────────────────────────────────
#  STEP 2: Synthetic Dataset Generation
#  (Replace with real Drebin CSV if available)
# ─────────────────────────────────────────────

def generate_synthetic_drebin_dataset(n_samples=5000, random_state=42):
    """
    Generates a synthetic Drebin-style dataset.
    Malware apps tend to request more dangerous permissions and
    use more suspicious API calls than benign apps.
    
    To use the REAL Drebin dataset:
      1. Download from https://drebin.mlsec.org/
      2. Use drebin_feature_extractor.py to parse APKs
      3. Replace this function with pd.read_csv('drebin_features.csv')
    """
    np.random.seed(random_state)
    n_malware = n_samples // 2
    n_benign  = n_samples - n_malware

    def gen_samples(n, malware=True):
        data = []
        for _ in range(n):
            row = {}
            # Permissions
            for perm in DANGEROUS_PERMISSIONS:
                if malware:
                    row[f"PERM_{perm}"] = int(np.random.random() < 0.70)
                else:
                    row[f"PERM_{perm}"] = int(np.random.random() < 0.20)
            # API Calls
            for api in SUSPICIOUS_API_CALLS:
                key = f"API_{api.replace('.', '_')}"
                if malware:
                    row[key] = int(np.random.random() < 0.65)
                else:
                    row[key] = int(np.random.random() < 0.15)
            # Intents
            for intent in SUSPICIOUS_INTENTS:
                if malware:
                    row[f"INTENT_{intent}"] = int(np.random.random() < 0.60)
                else:
                    row[f"INTENT_{intent}"] = int(np.random.random() < 0.18)
            row['label'] = 1 if malware else 0
            data.append(row)
        return data

    malware_data = gen_samples(n_malware, malware=True)
    benign_data  = gen_samples(n_benign,  malware=False)
    df = pd.DataFrame(malware_data + benign_data)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    print(f"\n[DATASET] Generated {len(df)} samples")
    print(f"          Malware : {df['label'].sum()} ({df['label'].mean()*100:.1f}%)")
    print(f"          Benign  : {(df['label']==0).sum()} ({(1-df['label'].mean())*100:.1f}%)")
    return df


# ─────────────────────────────────────────────
#  STEP 3: Preprocessing & Feature Selection
# ─────────────────────────────────────────────

def preprocess_and_select_features(df, top_k=80):
    print("\n[PREPROCESSING] Starting feature engineering...")
    X = df.drop('label', axis=1)
    y = df['label']

    # Mutual Information based feature selection
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=X.columns).sort_values(ascending=False)
    top_features = mi_series.head(top_k).index.tolist()

    print(f"[FEATURE SELECTION] Top {top_k} features selected via Mutual Information")
    print(f"  Top 10: {top_features[:10]}")
    return X[top_features], y, top_features


# ─────────────────────────────────────────────
#  STEP 4: Train Random Forest Model
# ─────────────────────────────────────────────

def train_random_forest(X_train, y_train):
    print("\n[TRAINING] Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print("[TRAINING] Model training complete ✓")
    return model


# ─────────────────────────────────────────────
#  STEP 5: Evaluate Model
# ─────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, X_train, y_train):
    y_pred     = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_pred_proba)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1', n_jobs=-1)

    results = {
        "accuracy":          round(acc, 4),
        "precision":         round(prec, 4),
        "recall":            round(rec, 4),
        "f1_score":          round(f1, 4),
        "roc_auc":           round(auc, 4),
        "cv_f1_mean":        round(cv_scores.mean(), 4),
        "cv_f1_std":         round(cv_scores.std(), 4),
    }

    print("\n" + "="*55)
    print("  MODEL EVALUATION RESULTS")
    print("="*55)
    print(f"  Accuracy  : {acc*100:.2f}%")
    print(f"  Precision : {prec*100:.2f}%")
    print(f"  Recall    : {rec*100:.2f}%")
    print(f"  F1-Score  : {f1*100:.2f}%")
    print(f"  ROC-AUC   : {auc*100:.2f}%")
    print(f"  CV F1     : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
    print("="*55)
    print("\n[CLASSIFICATION REPORT]")
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malware']))

    cm = confusion_matrix(y_test, y_pred)
    print("[CONFUSION MATRIX]")
    print(f"  TN={cm[0,0]}  FP={cm[0,1]}")
    print(f"  FN={cm[1,0]}  TP={cm[1,1]}")

    return results


# ─────────────────────────────────────────────
#  STEP 6: Save Model & Metadata
# ─────────────────────────────────────────────

def save_model(model, selected_features, results, output_dir="./"):
    model_path    = f"{output_dir}malware_detector.pkl"
    features_path = f"{output_dir}selected_features.json"
    results_path  = f"{output_dir}model_results.json"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(features_path, 'w') as f:
        json.dump({"features": selected_features, "count": len(selected_features)}, f, indent=2)

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[SAVED] Model        → {model_path}")
    print(f"[SAVED] Features     → {features_path}")
    print(f"[SAVED] Results      → {results_path}")
    return model_path


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("="*55)
    print("  ANDROID MALWARE DETECTION - MODEL TRAINING")
    print("="*55)

    # 1. Generate / Load dataset
    df = generate_synthetic_drebin_dataset(n_samples=6000)

    # 2. Preprocess
    X, y, selected_features = preprocess_and_select_features(df, top_k=80)

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(f"\n[SPLIT] Train: {len(X_train)} | Test: {len(X_test)}")

    # 4. Train
    model = train_random_forest(X_train, y_train)

    # 5. Evaluate
    results = evaluate_model(model, X_test, y_test, X_train, y_train)

    # 6. Save
    save_model(model, selected_features, results, output_dir="./")

    print("\n[✓] Training pipeline complete!")
