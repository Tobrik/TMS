"""
DDXPlus sklearn training script.
Trains a RandomForestClassifier on DDXPlus patient data.

Outputs (same folder):
  - model.pkl          — trained sklearn Pipeline (encoder + classifier)
  - feature_names.json — ordered list of feature tokens used by the model
  - label_classes.json — ordered list of pathology labels
"""

import json
import ast
import csv
import sys
import os
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ─── Paths ───
BASE = Path(__file__).parent
EVIDENCES_JSON  = BASE / "release_evidences.json"
TRAIN_CSV       = BASE / "release_train_patients"
VALIDATE_CSV    = BASE / "release_validate_patients"
MODEL_OUT       = BASE / "model.pkl"
FEATURES_OUT    = BASE / "feature_names.json"
LABELS_OUT      = BASE / "label_classes.json"

# ─── Step 1: Build full feature vocabulary from evidences JSON ───
print("Building feature vocabulary from evidences JSON...")
with open(EVIDENCES_JSON, encoding="utf-8") as f:
    ev_meta = json.load(f)

feature_set = set()
for ev_id, meta in ev_meta.items():
    dtype = meta.get("data_type", "B")
    if dtype == "B":
        # Binary: feature = ev_id itself
        feature_set.add(ev_id)
    else:
        # Categorical (C) or Multi-choice (M): one feature per possible value
        for val in meta.get("possible-values", []):
            feature_set.add(f"{ev_id}_@_{val}")

# Also add AGE and SEX
feature_set.add("__AGE__")
feature_set.add("__SEX_M__")

# Sort for deterministic order
FEATURE_NAMES = sorted(feature_set)
feat_idx = {f: i for i, f in enumerate(FEATURE_NAMES)}
N_FEATURES = len(FEATURE_NAMES)
print(f"  Total features: {N_FEATURES}")


# ─── Step 2: Parse & encode rows ───
def encode_row(row: dict) -> np.ndarray:
    vec = np.zeros(N_FEATURES, dtype=np.float32)

    # AGE (normalized to 0-1, max age 100)
    try:
        age = float(row["AGE"]) / 100.0
    except (ValueError, KeyError):
        age = 0.0
    if "__AGE__" in feat_idx:
        vec[feat_idx["__AGE__"]] = age

    # SEX
    if row.get("SEX", "").upper() == "M":
        if "__SEX_M__" in feat_idx:
            vec[feat_idx["__SEX_M__"]] = 1.0

    # EVIDENCES — list of strings like "E_48" or "E_54_@_V_161"
    try:
        evidences = ast.literal_eval(row["EVIDENCES"])
    except Exception:
        evidences = []

    for ev in evidences:
        if ev in feat_idx:
            vec[feat_idx[ev]] = 1.0

    return vec


def load_dataset(csv_path: Path, max_rows: int = None):
    X_rows = []
    y_rows = []
    print(f"  Loading {csv_path.name}...")
    with open(csv_path, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if max_rows and i >= max_rows:
                break
            X_rows.append(encode_row(row))
            y_rows.append(row["PATHOLOGY"])
            if (i + 1) % 100_000 == 0:
                print(f"    Processed {i+1:,} rows...")
    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows)
    print(f"  Loaded {len(y):,} samples, {X.shape[1]} features")
    return X, y


# ─── Step 3: Load data ───
print("\nLoading training data...")
X_train, y_train = load_dataset(TRAIN_CSV)

print("\nLoading validation data...")
X_val, y_val = load_dataset(VALIDATE_CSV)

# ─── Step 4: Train ───
print("\nTraining RandomForestClassifier...")
clf = RandomForestClassifier(
    n_estimators=50,
    max_depth=25,       # limit depth to reduce model size significantly
    min_samples_leaf=4,
    n_jobs=-1,          # use all CPU cores
    random_state=42,
    verbose=1,
)
clf.fit(X_train, y_train)

# ─── Step 5: Save ───
print("Saving model and metadata...")
joblib.dump(clf, MODEL_OUT, compress=3)
print(f"  model.pkl → {MODEL_OUT}")

with open(FEATURES_OUT, "w", encoding="utf-8") as f:
    json.dump(FEATURE_NAMES, f, ensure_ascii=False, indent=2)
print(f"  feature_names.json → {FEATURES_OUT}")

with open(LABELS_OUT, "w", encoding="utf-8") as f:
    json.dump(clf.classes_.tolist(), f, ensure_ascii=False, indent=2)
print(f"  label_classes.json → {LABELS_OUT}")

# ─── Step 6: Evaluate ───
print("\nEvaluating on validation set...")
y_pred = clf.predict(X_val)
acc = accuracy_score(y_val, y_pred)
print(f"\n  Accuracy: {acc:.4f} ({acc*100:.2f}%)")
print("\nPer-class report:")
print(classification_report(y_val, y_pred, zero_division=0))

print("\nDone! Model trained and saved.")
