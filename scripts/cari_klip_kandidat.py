"""Cari potongan video yang ramai dan banyak orang tertutup, sebagai kandidat klip ground truth baru.

Setiap video dibaca 1 frame per detik, lalu dideteksi dengan D-FINE. Per detik dihitung:
  orang        jumlah orang dengan skor >= 0,5
  ragu         jumlah orang dengan skor 0,2 sampai 0,5 (sering orang yang tertutup sebagian)
  dempet       pasangan orang yakin yang box-nya saling menutupi (>= 30% luas box yang lebih kecil, sama dengan
               aturan "ramai" di evaluate.py temanmu), setelah box dobel dibuang
  kecil        orang yang tingginya < 60 px di resolusi asli (diabaikan saat penilaian)
Lalu dicari jendela 60 detik dengan skor tertinggi per video:
  skor = rata-rata orang + 2 x rata-rata dempet + 0,5 x rata-rata ragu

Hasil: CSV per detik (bisa dilanjutkan kalau terputus) dan CSV ringkasan per video, diurutkan dari skor tertinggi.

Contoh:
  python scripts/cari_klip_kandidat.py --videos "~/Dataset_Momen_PPE_Anonim" --log momen_log.csv \
      --min-persons 4 --out outputs/kandidat_gt
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from object_tracking.detector import OnnxDFineDetector  # noqa: E402

PERSON_HI, PERSON_LO, CROWD_IOA, SMALL_PX = 0.5, 0.2, 0.3, 60


def probe(path: Path) -> dict:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height,r_frame_rate:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    num, den = s["r_frame_rate"].split("/")
    return {"width": int(s["width"]), "height": int(s["height"]), "fps": float(num) / float(den),
            "duration": float(d["format"]["duration"])}


def frames_1fps(path: Path, width: int, height: int, scale_w: int = 1920, start: int = 0):
    """Satu frame per detik lewat ffmpeg (mulai detik `start`), diperkecil ke lebar scale_w
    (detektor 960 tidak kehilangan apa pun)."""
    sw = min(scale_w, width)
    sh = int(round(height * sw / width / 2)) * 2
    cmd = ["ffmpeg", "-v", "error", "-ss", str(start), "-i", str(path), "-vf", f"fps=1,scale={sw}:{sh}",
           "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n = sw * sh * 3
    k = start
    try:
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield k, np.frombuffer(buf, np.uint8).reshape(sh, sw, 3), width / sw
            k += 1
    finally:
        p.kill()


def pair_overlap(boxes: np.ndarray) -> int:
    n = len(boxes)
    if n < 2:
        return 0
    x1, y1, x2, y2 = boxes.T
    area = np.maximum((x2 - x1) * (y2 - y1), 1e-6)
    iw = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
    ih = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
    ioa = iw * ih / np.minimum(area[:, None], area[None])
    return int((np.triu(ioa, 1) >= CROWD_IOA).sum())


def nms_keep(b: np.ndarray, thr: float = 0.6) -> np.ndarray:
    """Buang box dobel (D-FINE kadang memberi dua box untuk satu orang). Box sudah urut skor tidak tersedia,
    jadi yang lebih besar dipertahankan."""
    if len(b) < 2:
        return b
    order = np.argsort(-(b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))
    keep = []
    for i in order:
        if all(_iou(b[i], b[j]) < thr for j in keep):
            keep.append(i)
    return b[keep]


def _iou(a, b) -> float:
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    return inter / max((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter, 1e-6)


def best_window(rows: list[dict], win: int) -> dict:
    if not rows:
        return {}
    for r in rows:   # dempet dihitung ulang dari orang yakin (skor >= 0,5) setelah box dobel dibuang
        r["dempet"] = pair_overlap(nms_keep(np.array(json.loads(r["boxes"]), float).reshape(-1, 4)))
    a = {k: np.array([r[k] for r in rows], float) for k in ("orang", "ragu", "dempet", "kecil")}
    score = a["orang"] + 2 * a["dempet"] + 0.5 * a["ragu"]
    w = min(win, len(score))
    cs = np.concatenate([[0], np.cumsum(score)])
    means = (cs[w:] - cs[:-w]) / w
    s = int(np.argmax(means))
    sl = slice(s, s + w)
    return {"mulai_s": s, "selesai_s": s + w, "skor": round(float(means[s]), 2),
            "rata_orang": round(float(a["orang"][sl].mean()), 2), "maks_orang": int(a["orang"][sl].max()),
            "rata_dempet": round(float(a["dempet"][sl].mean()), 2), "rata_ragu": round(float(a["ragu"][sl].mean()), 2),
            "persen_kecil": round(float(100 * a["kecil"][sl].sum() / max(1, a["orang"][sl].sum() + a["ragu"][sl].sum())), 1)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", required=True, help="folder video momen")
    ap.add_argument("--log", default=None, help="momen_log.csv (kolom clip_file, persons_max) untuk memilih video")
    ap.add_argument("--min-persons", type=int, default=4, help="hanya video dengan persons_max >= ini (butuh --log)")
    ap.add_argument("--clips", nargs="*", help="nama file video tertentu (menimpa --log)")
    ap.add_argument("--skip", nargs="*", default=[], help="video yang dilewati (misalnya sumber klip GT yang sudah ada)")
    ap.add_argument("--model", default=str(ROOT / "models" / "dfine-n-stg2-960.onnx"))
    ap.add_argument("--window", type=int, default=60, help="panjang potongan kandidat (detik)")
    ap.add_argument("--out", default=str(ROOT / "outputs" / "kandidat_gt"))
    ap.add_argument("--budget-sec", type=float, default=0, help="berhenti setelah sekian detik (0 = tanpa batas). "
                    "Jalankan lagi untuk melanjutkan; hasil sebagian disimpan di per_detik/*.part.csv")
    args = ap.parse_args()

    vdir = Path(args.videos).expanduser()
    if args.clips:
        names = args.clips
    elif args.log:
        with open(Path(args.log).expanduser() if Path(args.log).expanduser().exists() else vdir / args.log) as f:
            names = [r["clip_file"] for r in csv.DictReader(f) if int(r["persons_max"] or 0) >= args.min_persons]
    else:
        names = sorted(p.name for p in vdir.glob("*.mp4"))
    names = [n for n in names if n not in set(args.skip)]
    out = Path(args.out).expanduser()
    (out / "per_detik").mkdir(parents=True, exist_ok=True)
    det = OnnxDFineDetector(args.model)
    print(f"{len(names)} video, model {Path(args.model).name}", flush=True)

    summary = []
    t_start = time.time()
    for i, name in enumerate(names, 1):
        path = vdir / name
        per_sec = out / "per_detik" / (Path(name).stem + ".csv")
        meta = probe(path)
        if not per_sec.exists():
            part = per_sec.with_suffix(".part.csv")
            fields = ["detik", "orang", "ragu", "dempet", "kecil", "boxes"]
            done = sum(1 for _ in part.open()) - 1 if part.exists() else 0
            t0 = time.time()
            with part.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                if done <= 0:
                    w.writeheader()
                    done = 0
                finished = True
                for k, img, s in frames_1fps(path, meta["width"], meta["height"], start=done):
                    persons, _, _ = det.detect(img, conf_thresh=PERSON_LO)
                    b = np.array([[p.x1, p.y1, p.x2, p.y2] for p in persons], float).reshape(-1, 4) * s
                    c = np.array([p.conf for p in persons], float)
                    hi = b[c >= PERSON_HI]
                    w.writerow({"detik": k, "orang": int((c >= PERSON_HI).sum()), "ragu": int((c < PERSON_HI).sum()),
                                "dempet": pair_overlap(b), "kecil": int(((b[:, 3] - b[:, 1]) < SMALL_PX).sum()),
                                "boxes": json.dumps(np.round(hi).astype(int).tolist())})
                    f.flush()
                    if args.budget_sec and time.time() - t_start > args.budget_sec:
                        finished = False
                        break
            if not finished:
                print(f"[{i}/{len(names)}] {name}: berhenti di detik {k} (batas waktu), jalankan lagi untuk lanjut",
                      flush=True)
                sys.exit(3)
            part.rename(per_sec)
            print(f"[{i}/{len(names)}] {name}: selesai, {time.time() - t0:.0f} s", flush=True)
        with per_sec.open() as f:
            rows = [{k: (int(v) if k != "boxes" else v) for k, v in r.items()} for r in csv.DictReader(f)]
        bw = best_window(rows, args.window)
        summary.append({"video": name, "kamera": name[:3], "resolusi": f"{meta['width']}x{meta['height']}",
                        "durasi_s": round(meta["duration"], 1), **bw})

    summary.sort(key=lambda r: -r.get("skor", 0))
    with (out / "ringkasan.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(f"selesai: {out / 'ringkasan.csv'}")


if __name__ == "__main__":
    main()
