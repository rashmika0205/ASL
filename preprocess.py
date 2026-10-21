import numpy as np

NUM_HAND_VALUES = 63    # 21 landmarks x (x, y, z)
NUM_FACE_VALUES = 5     # face_cx, face_cy, face_w, face_h, face_found
CLIP = 5.0

SEQ_LEN = 30                                  # frames per clip
FRAME_RAW_VALUES = 2 * NUM_HAND_VALUES + NUM_FACE_VALUES   # 131


# --------------------------------------------------
# SINGLE HAND (unchanged)
# --------------------------------------------------

def normalize_landmarks(row, face=None):
    """
    68 features for ONE hand:
      63 hand-shape values (wrist subtracted, scaled by hand size)
      5 values for hand position relative to the face + face_found flag

    An all-zero hand row (no hand) returns all zeros.
    """
    pts = np.asarray(row, dtype=np.float32).reshape(21, 3)
    face_feats = np.zeros(NUM_FACE_VALUES, dtype=np.float32)

    if not pts.any():
        return np.concatenate([pts.flatten(), face_feats])

    wrist_xy = pts[0, :2].copy()
    centre_xy = pts[:, :2].mean(axis=0)

    if face is not None:
        face = np.asarray(face, dtype=np.float32)
        if face[4] > 0 and face[2] > 0 and face[3] > 0:
            fx, fy, fw, fh = face[:4]
            face_feats[0] = (centre_xy[0] - fx) / fw
            face_feats[1] = (centre_xy[1] - fy) / fh
            face_feats[2] = (wrist_xy[0] - fx) / fw
            face_feats[3] = (wrist_xy[1] - fy) / fh
            face_feats[:4] = np.clip(face_feats[:4], -CLIP, CLIP)
            face_feats[4] = 1.0

    pts = pts - pts[0]
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 0:
        pts = pts / scale

    return np.concatenate([pts.flatten(), face_feats])


# --------------------------------------------------
# SEQUENCES (new)
# --------------------------------------------------

def extract_frame_raw(hand_landmarks_list, face):
    """
    One camera frame -> 131 raw values:
        [hand slot A: 63][hand slot B: 63][face: 5]

    Hands are ordered left-to-right on screen. A single hand always goes in
    slot A, slot B stays zero.
    """
    raw = np.zeros(FRAME_RAW_VALUES, dtype=np.float32)

    hands = sorted(hand_landmarks_list[:2], key=lambda hand: hand[0].x)
    for slot, hand in enumerate(hands):
        values = [v for lm in hand for v in (lm.x, lm.y, lm.z)]
        start = slot * NUM_HAND_VALUES
        raw[start:start + NUM_HAND_VALUES] = values

    raw[2 * NUM_HAND_VALUES:] = face
    return raw


def frame_features(raw):
    """131 raw values -> 136 features (68 per hand slot)."""
    face = raw[2 * NUM_HAND_VALUES:]
    slot_a = normalize_landmarks(raw[:NUM_HAND_VALUES], face)
    slot_b = normalize_landmarks(raw[NUM_HAND_VALUES:2 * NUM_HAND_VALUES], face)
    return np.concatenate([slot_a, slot_b])


def resample_by_time(times, frames, n=SEQ_LEN):
    """
    Pick `n` frames evenly spaced in TIME (nearest recorded frame), so a clip
    looks the same whether the camera ran at 15 or 30 fps.
    """
    times = np.asarray(times, dtype=np.float64)
    frames = np.asarray(frames, dtype=np.float32)
    targets = np.linspace(times[0], times[-1], n)
    idx = np.abs(times[None, :] - targets[:, None]).argmin(axis=1)
    return frames[idx]


def sequence_features(seq_raw):
    """
    (30, 131) raw clip -> one feature vector (952 values) describing:
      - average hand shape/position in 4 time segments
      - how much everything varies
      - how fast it moves
      - net change from start to end
    """
    feats = np.array([frame_features(f) for f in seq_raw])        # (30, 136)

    segment_means = [s.mean(axis=0) for s in np.array_split(feats, 4)]
    spread = feats.std(axis=0)
    speed = np.abs(np.diff(feats, axis=0)).mean(axis=0)
    net_change = feats[-5:].mean(axis=0) - feats[:5].mean(axis=0)

    return np.concatenate(segment_means + [spread, speed, net_change]).astype(np.float32)