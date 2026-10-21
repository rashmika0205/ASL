import glob
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from preprocess import FRAME_RAW_VALUES, SEQ_LEN, sequence_features
from signs import SIGNS

# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

SEQ_DIR = "data/sequences"
MODEL_FILE = "model.pkl"
CONFUSION_FILE = "confusion_matrix.png"

RANDOM_STATE = 42
MIN_CLIPS_PER_SIGN = 40

# --------------------------------------------------
# LOAD CLIPS
# --------------------------------------------------

X, y, groups = [], [], []

for sign in SIGNS:
    files = sorted(glob.glob(os.path.join(SEQ_DIR, sign, "*.npy")))
    kept = 0
    for path in files:
        clip = np.load(path)
        if clip.shape != (SEQ_LEN, FRAME_RAW_VALUES):
            print(f"Skipping {path}: unexpected shape {clip.shape}")
            continue
        X.append(sequence_features(clip))
        y.append(sign)
        groups.append(os.path.basename(path).split("_")[0])   # recording session
        kept += 1
    print(f"{sign:10s} {kept} clips")
    if 0 < kept < MIN_CLIPS_PER_SIGN:
        print(f"  WARNING: {sign} has fewer than {MIN_CLIPS_PER_SIGN} clips.")
    if kept == 0:
        print(f"  WARNING: no clips for {sign} - it will never be predicted.")

if not X:
    raise SystemExit(f"No clips found in {SEQ_DIR}. Run collect_sequences.py first.")

X = np.array(X)
y = np.array(y)
groups = np.array(groups)
sessions = sorted(set(groups))

print(f"\nTotal clips: {len(X)}   Features per clip: {X.shape[1]}")
print(f"Recording sessions: {len(sessions)}")

# --------------------------------------------------
# SPLIT
# --------------------------------------------------

if len(sessions) >= 3:
    # whole sessions go to either train or test -> honest accuracy
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25,
                                 random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    print("Split by recording session (honest test).")
else:
    train_idx, test_idx = train_test_split(
        np.arange(len(X)), test_size=0.2, random_state=RANDOM_STATE,
        stratify=y,
    )
    print("NOTE: fewer than 3 sessions, so test clips come from the same "
          "sessions as training clips and accuracy will look better than it is. "
          "Record in 3+ separate sessions for a fairer number.")

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

missing = set(y) - set(y_train)
if missing:
    print(f"WARNING: no training clips for {sorted(missing)} in this split.")

# --------------------------------------------------
# TRAIN + EVALUATE
# --------------------------------------------------


def make_model():
    return RandomForestClassifier(
        n_estimators=400,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


model = make_model()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nTest accuracy: {accuracy_score(y_test, y_pred):.3f}\n")

present = sorted(set(y_test) | set(y_pred))
print(classification_report(y_test, y_pred, labels=present, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=present)

pairs = [(cm[i, j], present[i], present[j])
         for i in range(len(present)) for j in range(len(present))
         if i != j and cm[i, j] > 0]
pairs.sort(reverse=True)
if pairs:
    print("Most confused (true -> predicted):")
    for count, true, pred in pairs[:5]:
        print(f"  {true} -> {pred}: {count}")

fig, ax = plt.subplots(figsize=(12, 10))
ConfusionMatrixDisplay(cm, display_labels=present).plot(
    ax=ax, cmap="Blues", xticks_rotation=90, colorbar=False)
plt.title("Confusion matrix (test set)")
plt.tight_layout()
plt.savefig(CONFUSION_FILE)
print(f"\nSaved {CONFUSION_FILE}")

# --------------------------------------------------
# FINAL MODEL: retrain on ALL clips, then save
# --------------------------------------------------

final_model = make_model()
final_model.fit(X, y)

joblib.dump(
    {"model": final_model, "labels": sorted(set(y)), "seq_len": SEQ_LEN},
    MODEL_FILE,
)
print(f"Saved {MODEL_FILE} (trained on all {len(X)} clips)")