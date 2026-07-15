import argparse
import csv
import os
import pickle
import numpy as np
from features import load_wav, speech_before, frame_energy_db, f0_contour

def extract_features(x, sr, pause_start):
    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(7, dtype=np.float32)
    e = frame_energy_db(seg, sr)
    f0 = f0_contour(seg, sr)
    voiced = f0[f0 > 0]
    if len(e) < 5:
        return np.zeros(7, dtype=np.float32)
    final_e = e[-3:].mean()
    mean_e = e.mean()
    energy_drop = mean_e - final_e
    energy_ratio = final_e / (mean_e + 1e-5)
    final_pitch = voiced[-3:].mean() if len(voiced) >= 3 else 0.0
    pitch_slope = voiced[-1] - voiced[-4] if len(voiced) >= 4 else 0.0
    pitch_std = voiced.std() if len(voiced) > 2 else 0.0
    voiced_ratio = len(voiced) / len(f0) if len(f0) > 0 else 0.0
    return np.array([final_e, energy_drop, energy_ratio, final_pitch, pitch_slope, pitch_std, voiced_ratio], dtype=np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    if not os.path.exists("model.pkl"):
        print("Error: model.pkl not found!")
        return

    with open("model.pkl", "rb") as f:
        clf = pickle.load(f)

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    X, keys = [], []
    for r in rows:
        x, sr = load_wav(os.path.join(args.data_dir, r["audio_file"]))
        X.append(extract_features(x, sr, float(r["pause_start"])))
        keys.append((r["turn_id"], r["pause_index"]))

    p = clf.predict_proba(np.array(X))[:, 1]
    with open(args.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["turn_id", "pause_index", "p_eot"])
        for (tid, pi), pi_p in zip(keys, p):
            w.writerow([tid, pi, f"{pi_p:.4f}"])
    print(f"Wrote predictions to {args.out}")

if __name__ == "__main__":
    main()