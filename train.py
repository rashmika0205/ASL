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
SAME_SESSION_TEST_FRACTION = 0.25   # used only for signs recorded in one session

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

print(f"\nTotal clips: {len(X)}   Features per clip: {X.shape[1]}")
print(f"Recording sessions: {len(set(groups))}")

# --------------------------------------------------
# SPLIT (every sign is always present in training)
#   sign recorded in 2+ sessions -> its NEWEST session is the test set
#                                   ("cross-session": honest accuracy)
#   sign recorded in 1 session   -> its LAST 25% of clips are the test set
#                                   ("same-session": optimistic accuracy)
# --------------------------------------------------

train_idx, test_idx, test_kind = [], [], []
single_session = []

print("\nSign        sessions  train  test  test type")
for sign in sorted(set(y)):
    idx = np.where(y == sign)[0]            # already in time order (files sorted by name)
    sess = sorted(set(groups[idx]))

    if len(sess) >= 2:
        held = sess[-1]
        test = idx[groups[idx] == held]
        train = idx[groups[idx] != held]
        kind = "cross-session"
    else:
        single_session.append(sign)
        cut = max(1, int(round(len(idx) * (1 - SAME_SESSION_TEST_FRACTION))))
        train, test = idx[:cut], idx[cut:]
        kind = "same-session"

    train_idx += list(train)
    test_idx += list(test)
    test_kind += [kind] * len(test)
    print(f"{sign:11s} {len(sess):8d} {len(train):6d} {len(test):5d}  {kind}")

test_kind = np.array(test_kind)
train_idx, test_idx = np.array(train_idx), np.array(test_idx)

X_train, X_test = X[train_idx], X[test_idx]
y_train, y_test = y[train_idx], y[test_idx]

if single_session:
    print(f"\nNOTE: {len(single_session)} signs were recorded in only ONE session "
          f"({', '.join(single_session)}).")
    print("      Their test score is optimistic and the model has not seen them in "
          "another setting.")
    print("      Record them again in a separate run of collect_sequences.py "
          "(different time, light or position).")

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

print(f"\nOverall test accuracy: {accuracy_score(y_test, y_pred):.3f}")
for kind in ("cross-session", "same-session"):
    mask = test_kind == kind
    if mask.any():
        acc = accuracy_score(y_test[mask], y_pred[mask])
        print(f"  {kind:14s}: {acc:.3f}  ({mask.sum()} clips)"
              + ("   <- the honest number" if kind == "cross-session" else ""))
print()

present = sorted(set(y_test) | set(y_pred))
print(classification_report(y_test, y_pred, labels=present, zero_division=0))

cm = confusion_matrix(y_test, y_pred, labels=present)

pairs = [(cm[i, j], present[i], present[j])
         for i in range(len(present)) for j in range(len(present))
         if i != j and cm[i, j] > 0]
pairs.sort(reverse=True)
if pairs:
    print("Most confused (true -> predicted):")
    for count, true, pred in pairs[:8]:
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