"""Ringkasan per kamera dari hasil cari_klip_kandidat.py: ukuran orang dan titik deteksi yang diam.

Dipakai untuk memilih kamera untuk dua uji:
  - Resolusi tinggi: kamera yang orangnya kecil di input detektor. Detektor 960 mengecilkan frame 2688 px
    jadi 960 px (x0,36) dan frame 4K jadi x0,25. Orang setinggi 120 px di video 4K hanya 30 px di input detektor.
  - Mask area: kamera dengan box "orang" yang diam di tempat yang sama hampir sepanjang video
    (sering benda: handle hand pallet, tiang, manekin, poster, pantulan). Ini perlu dicek mata, karena orang yang
    duduk lama juga terlihat diam. Potongan gambar setiap titik disimpan untuk dicek.

Contoh:
  python scripts/analisis_kamera.py --scan outputs/kandidat_gt outputs/kandidat_gt_lain \\
      --videos "~/Dataset_Momen_PPE_Anonim" --out outputs/analisis_kamera
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import numpy as np

INPUT = 960            # sisi input detektor (letterbox persegi)
SMALL_IN = 48          # orang di bawah ini (px di input detektor) mulai sulit untuk detektor 960
STATIC_IOU, STATIC_SHARE, STATIC_MIN_S = 0.7, 0.4, 60


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = iw * ih
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + bb[None] - inter, 1e-6)


def static_spots(per_sec: list[np.ndarray]) -> list[dict]:
    """Box yang muncul di tempat yang sama (IoU >= 0,7) di >= 40% detik video."""
    n = len(per_sec)
    allb = [b for bs in per_sec for b in bs]
    if n < STATIC_MIN_S or not allb:
        return []
    allb = np.array(allb, float)
    secs = np.concatenate([[i] * len(bs) for i, bs in enumerate(per_sec)])
    m = iou(allb, allb) >= STATIC_IOU
    spots, used = [], np.zeros(len(allb), bool)
    order = np.argsort(-m.sum(1))
    for i in order:
        if used[i]:
            continue
        members = np.where(m[i] & ~used)[0]
        share = len(set(secs[members])) / n
        used[members] = True
        if share >= STATIC_SHARE:
            spots.append({"box": np.median(allb[members], 0).round().astype(int).tolist(), "persen_detik": round(100 * share)})
    return spots


def crop(video: Path, t: int, box, out: Path, pad: int = 120) -> None:
    x1, y1, x2, y2 = box
    w, h = x2 - x1 + 2 * pad, y2 - y1 + 2 * pad
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(t), "-i", str(video), "-frames:v", "1", "-vf",
                    f"crop={w}:{h}:{max(0, x1 - pad)}:{max(0, y1 - pad)},drawbox=x={min(pad, x1)}:y={min(pad, y1)}:"
                    f"w={x2 - x1}:h={y2 - y1}:color=red:t=4", str(out)], check=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", nargs="+", required=True, help="folder output cari_klip_kandidat.py")
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    vdir, out = Path(args.videos).expanduser(), Path(args.out).expanduser()
    (out / "titik_diam").mkdir(parents=True, exist_ok=True)

    res = {}
    for d in args.scan:
        for r in csv.DictReader(open(Path(d) / "ringkasan.csv")):
            w, h = (int(v) for v in r["resolusi"].split("x"))
            res[r["video"]] = (w, h)
    cams = defaultdict(lambda: {"h_in": [], "h_src": [], "orang": 0, "ragu": 0, "detik": 0, "video": [], "titik": []})
    for d in args.scan:
        for f in sorted((Path(d) / "per_detik").glob("*.csv")):
            video = f.stem + ".mp4"
            if video not in res:
                continue
            W, H = res[video]
            s = INPUT / max(W, H)
            rows = list(csv.DictReader(open(f)))
            per_sec = [np.array(json.loads(r["boxes"]), float).reshape(-1, 4) for r in rows]
            c = cams[video[:3]]
            c["res"] = f"{W}x{H}"
            c["skala"] = round(s, 3)
            c["video"].append(video)
            c["detik"] += len(rows)
            c["orang"] += sum(int(r["orang"]) for r in rows)
            c["ragu"] += sum(int(r["ragu"]) for r in rows)
            for b in per_sec:
                hs = b[:, 3] - b[:, 1]
                c["h_src"] += hs.tolist()
                c["h_in"] += (hs * s).tolist()
            for k, sp in enumerate(static_spots(per_sec)):
                name = f"{f.stem}_titik{k + 1}.jpg"
                t = next(i for i, b in enumerate(per_sec) if len(b) and iou(np.array([sp["box"]], float), b).max() >= STATIC_IOU)
                crop(vdir / video, t, sp["box"], out / "titik_diam" / name)
                c["titik"].append({"video": video, **sp, "gambar": f"titik_diam/{name}"})

    rows = []
    for cam in sorted(cams):
        c = cams[cam]
        h_in = np.array(c["h_in"]) if c["h_in"] else np.array([np.nan])
        rows.append({"kamera": cam, "resolusi": c["res"], "skala_ke_input": c["skala"], "video_dipindai": len(c["video"]),
                     "detik": c["detik"], "box_orang": len(c["h_in"]),
                     "tinggi_median_px_input": round(float(np.nanmedian(h_in)), 1),
                     "tinggi_p10_px_input": round(float(np.nanpercentile(h_in, 10)), 1),
                     f"persen_di_bawah_{SMALL_IN}px_input": round(100 * float(np.mean(h_in < SMALL_IN)), 1),
                     "ragu_per_orang": round(c["ragu"] / max(c["orang"], 1), 2),
                     "titik_diam": len(c["titik"]),
                     "titik_diam_detail": "; ".join(f"{t['video']} {t['box']} {t['persen_detik']}%" for t in c["titik"])})
    with (out / "per_kamera.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} kamera -> {out / 'per_kamera.csv'}")


if __name__ == "__main__":
    main()
