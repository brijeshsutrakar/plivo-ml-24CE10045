import argparse
import csv
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit
from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    # Context thoda bada karte hain taaki robust pattern mile
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(7, dtype=np.float32)
        
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    
    if len(e) < 5:
        return np.zeros(7, dtype=np.float32)
        
    # 1. Energy Features
    final_e = e[-3:].mean()
    mean_e = e.mean()
    energy_drop = mean_e - final_e
    
    # New: Energy Ratio (Relative scaling)
    energy_ratio = final_e / (mean_e + 1e-5)
    
    # 2. Pitch Dynamics
    final_pitch = voiced[-3:].mean() if len(voiced) >= 3 else 0.0
    pitch_slope = voiced[-1] - voiced[-4] if len(voiced) >= 4 else 0.0
    
    # New: Pitch Variance (Monotonicity check)
    pitch_std = voiced.std() if len(voiced) > 2 else 0.0
    
    # 3. Voiced ratio
    voiced_ratio = len(voiced) / len(f0) if len(f0) > 0 else 0.0

    return np.array([
        final_e,
        energy_drop,
        energy_ratio,
        final_pitch,
        pitch_slope,
        pitch_std,
        voiced_ratio
    ], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    cache = {}
    X, y, groups, keys = [], [], [], []
    for r in rows:
        path = os.path.join(args.data_dir, r["audio_file"])
        if path not in cache:
            cache[path] = load_wav(path)
        x, sr = cache[path]
        X.append(extract_features(x, sr, float(r["pause_start"])))
        y.append(1 if r["label"] == "eot" else 0)
        groups.append(r["turn_id"])
        keys.append((r["turn_id"], r["pause_index"]))
    X, y = np.array(X), np.array(y)

    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=0).split(X, y, groups))
    
    # Thodi si tree depth badha rahe hain naye features ko accomodate karne ke liye
    clf = RandomForestClassifier(n_estimators=200, max_depth=7, min_samples_leaf=3, random_state=0, class_weight="balanced")
    clf.fit(X[tr], y[tr])
    print(f"held-out turn accuracy: {clf.score(X[te], y[te]):.3f}")

    clf.fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"wrote {len(keys)} predictions -> {args.out}")

if __name__ == "__main__":
    main()