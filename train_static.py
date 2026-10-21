import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

from preprocess import NUM_HAND_VALUES, normalize_landmarks

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

DATA_FILE = "data/landmarks.csv"
MODEL_FILE = "model_static.pkl"
CONFUSION_FILE = "confusion_matrix_static.png"

TEST_SIZE = 0.2
RANDOM_STATE = 42
MIN_SAMPLES_PER_CLASS = 100

# --------------------------------------------------
# LOAD DATA (handles both CSV formats)
#   64 columns = 63 hand values + label                (very first collector)
#   69 columns = 63 hand + 5 face values + label       (face-position collector)
# --------------------------------------------------

if not os.path.exists(DATA_FILE):
    raise SystemExit(f"{DATA_FILE} not found.")

df = pd.read_csv(DATA_FILE).dropna()

if df.shape[1] == 64:
    uses_face = False
    print("Format: hand only (64 columns)")
elif df.shape[1] == 69:
    uses_face = True
    print("Format: hand + face position (69 columns)")
else:
    raise SystemExit(
        f"Unexpected number of columns: {df.shape[1]} (expected 64 or 69)."
    )

print(f"Loaded {len(df)} samples\n")
counts = df["label"].value_counts()
print(counts.to_string())

for label, n in counts.items():
    if n < MIN_SAMPLES_PER_CLASS:
        print(f"WARNING: {label} has only {n} samples - collect more.")

# --------------------------------------------------
# PREPROCESS
# --------------------------------------------------

values = df.drop(columns=["label"]).values.astype(np.float32)

X = np.array([
    normalize_landmarks(row[:NUM_HAND_VALUES],
                        row[NUM_HAND_VALUES:] if uses_face else None)
    for row in values
])
y = df["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y,
)

# --------------------------------------------------
# TRAIN + EVALUATE
# --------------------------------------------------


def make_model():
    return RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


model = make_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}\n")
print(classification_report(y_test, y_pred, zero_division=0))

labels = sorted(model.classes_)
cm = confusion_matrix(y_test, y_pred, labels=labels)
ConfusionMatrixDisplay(cm, display_labels=labels).plot(cmap="Blues")
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(CONFUSION_FILE)
print(f"Saved {CONFUSION_FILE}")

# --------------------------------------------------
# FINAL MODEL: retrain on all samples, then save
# --------------------------------------------------

final_model = make_model()
final_model.fit(X, y)

joblib.dump(
    {"model": final_model, "labels": sorted(set(y)), "uses_face": uses_face},
    MODEL_FILE,
)
print(f"Saved {MODEL_FILE}")