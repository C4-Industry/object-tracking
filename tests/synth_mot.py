"""Sekuens MOT17 palsu (format file sama persis) untuk menguji loader, runner dan evaluasi tanpa data asli."""

from pathlib import Path

import numpy as np


def make_fake_mot17(root, name="MOT17-02", detector="SDP", length=150, fps=30, n_ped=10, seed=0):
    rng = np.random.default_rng(seed)
    d = Path(root) / "train" / f"{name}-{detector}"
    (d / "gt").mkdir(parents=True, exist_ok=True)
    (d / "det").mkdir(parents=True, exist_ok=True)
    (d / "seqinfo.ini").write_text(
        f"[Sequence]\nname={name}-{detector}\nimDir=img1\nframeRate={fps}\nseqLength={length}\n"
        "imWidth=1920\nimHeight=1080\nimExt=.jpg\n")
    gt, det = [], []
    pos = rng.uniform([100, 300], [1700, 700], (n_ped, 2))
    vel = rng.normal(0, 2.5, (n_ped, 2))
    life = [(int(rng.integers(1, length // 3)), int(rng.integers(length // 2, length + 1))) for _ in range(n_ped)]
    static = [(1500.0, 500.0), (300.0, 650.0)]                    # dua "orang diam" (kelas 7, tidak dinilai)
    for f in range(1, length + 1):
        pos += vel
        for i in range(n_ped):
            if life[i][0] <= f <= life[i][1]:
                x, y = pos[i]
                gt.append([f, i + 1, x, y, 50, 120, 1, 1, 1.0])
                if rng.random() < 0.9:
                    sc = rng.uniform(0.2, 1.0)
                    det.append([f, -1, x + rng.normal(0, 3), y + rng.normal(0, 3), 50 + rng.normal(0, 2), 120 + rng.normal(0, 3), sc])
        for k, (x, y) in enumerate(static):
            gt.append([f, 100 + k, x, y, 50, 120, 0, 7, 1.0])
            det.append([f, -1, x + rng.normal(0, 2), y + rng.normal(0, 2), 50, 120, rng.uniform(0.6, 0.9)])
        for _ in range(rng.poisson(0.5)):
            det.append([f, -1, *rng.uniform([0, 0], [1800, 900]), 50, 120, rng.uniform(0.05, 0.6)])
    fmt = ["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%d", "%d", "%.2f"]
    np.savetxt(d / "gt" / "gt.txt", np.array(gt), delimiter=",", fmt=fmt)
    np.savetxt(d / "det" / "det.txt", np.array(det), delimiter=",", fmt=["%d", "%d", "%.2f", "%.2f", "%.2f", "%.2f", "%.3f"])
    return Path(root)
