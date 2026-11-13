"""Benchmark tracker di MOT17 (split train, yang punya label ID).

Hanya butuh file label MOT17 (gt.txt, det.txt, seqinfo.ini), tidak butuh gambar, karena tracker dinilai
memakai deteksi publik dari MOT17 (FRCNN dan SDP). Jadi semua tracker mendapat deteksi yang persis sama,
dan yang dibandingkan murni kualitas tracking-nya.

Lisensi data MOT17: CC BY-NC-SA 3.0 (non-komersial). Dipakai hanya untuk mengukur tracker, bukan untuk
melatih model yang dikirim ke klien.
"""

from __future__ import annotations

import configparser
import time
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment

from . import metrics as mm
from .zoo import TrackerRunner, TrackerSpec

LABELS_URL = "https://motchallenge.net/data/MOT17Labels.zip"
TRAIN_SEQS = ["MOT17-02", "MOT17-04", "MOT17-05", "MOT17-09", "MOT17-10", "MOT17-11", "MOT17-13"]
STATIC_SEQS = {"MOT17-02", "MOT17-04", "MOT17-09"}     # kamera diam, paling mirip CCTV
DISTRACTOR_CLASSES = (2, 7, 8, 12)                      # orang di kendaraan, orang diam, distraktor, pantulan


# ---------------- data ----------------

def download_labels(root: str | Path) -> Path:
    """Unduh dan ekstrak MOT17Labels.zip (label dan deteksi publik, tanpa gambar) ke `root`."""
    root = Path(root)
    if (root / "train" / "MOT17-02-SDP" / "gt" / "gt.txt").exists():
        return root
    root.mkdir(parents=True, exist_ok=True)
    zpath = root / "MOT17Labels.zip"
    if not zpath.exists():
        print(f"mengunduh {LABELS_URL} ...")
        try:
            urllib.request.urlretrieve(LABELS_URL, zpath)
        except Exception as e:
            if zpath.exists():
                zpath.unlink()
            raise RuntimeError(
                f"Gagal mengunduh ({e}). Unduh manual {LABELS_URL} dari browser, "
                f"lalu taruh file zip-nya di {zpath} dan jalankan sel ini lagi."
            ) from e
    with zipfile.ZipFile(zpath) as z:
        z.extractall(root)
    if not (root / "train").exists():                     # beberapa versi zip punya folder induk MOT17/
        inner = next((p for p in root.iterdir() if (p / "train").exists()), None)
        if inner is None:
            raise RuntimeError(f"Isi zip tidak dikenal, folder 'train' tidak ditemukan di {root}")
        return inner
    return root


@dataclass
class Seq:
    name: str            # MOT17-02
    detector: str        # FRCNN, SDP, DPM
    fps: float
    length: int
    width: int
    height: int
    dets: dict           # frame -> Nx5 [x1, y1, x2, y2, skor]
    gt: np.ndarray       # baris gt.txt: frame, id, x, y, w, h, mark, class, visibility

    @property
    def static(self) -> bool:
        return self.name in STATIC_SEQS


def load_seq(root: str | Path, name: str, detector: str = "SDP") -> Seq:
    d = Path(root) / "train" / f"{name}-{detector}"
    ini = configparser.ConfigParser()
    ini.read(d / "seqinfo.ini")
    s = ini["Sequence"]
    raw = np.loadtxt(d / "det" / "det.txt", delimiter=",", ndmin=2)
    dets = {}
    for f in np.unique(raw[:, 0]).astype(int):
        r = raw[raw[:, 0] == f]
        dets[f] = np.stack([r[:, 2], r[:, 3], r[:, 2] + r[:, 4], r[:, 3] + r[:, 5], r[:, 6]], 1).astype(np.float32)
    gt = np.loadtxt(d / "gt" / "gt.txt", delimiter=",", ndmin=2)
    return Seq(name, detector, float(s["frameRate"]), int(s["seqLength"]), int(s["imWidth"]), int(s["imHeight"]),
               dets, gt)


def load_all(root: str | Path, detectors=("FRCNN", "SDP"), seqs=TRAIN_SEQS) -> list[Seq]:
    return [load_seq(root, n, d) for d in detectors for n in seqs]


def det_score_summary(seqs: list[Seq]) -> pd.DataFrame:
    rows = []
    for det in sorted({s.detector for s in seqs}):
        sc = np.concatenate([v[:, 4] for s in seqs if s.detector == det for v in s.dets.values()])
        rows.append({"detektor": det, "jumlah box": len(sc), "min": sc.min(), "p10": np.percentile(sc, 10),
                     "median": np.median(sc), "p90": np.percentile(sc, 90), "max": sc.max()})
    return pd.DataFrame(rows).set_index("detektor")


# ---------------- menjalankan tracker ----------------

def frame_step(seq: Seq, target_fps: float | None) -> int:
    """Lompatan frame untuk meniru FPS proses yang lebih rendah. None = FPS asli."""
    return 1 if not target_fps else max(1, int(round(seq.fps / target_fps)))


def run_tracker(spec: TrackerSpec, seq: Seq, target_fps: float | None = None) -> tuple[dict, dict]:
    """Jalankan satu tracker di satu sekuens. Hasil: {frame: (ids, boxes xyxy)} dan info waktu."""
    step = frame_step(seq, target_fps)
    real_fps = seq.fps / step
    run = TrackerRunner(spec, real_fps)
    empty = np.zeros((0, 5), np.float32)
    out = {}
    t0 = time.perf_counter()
    for f in range(1, seq.length + 1, step):
        t = (f - 1) / seq.fps
        dets = seq.dets.get(f, empty) if run.uses_detection() else None
        tracks = run.step(t, dets)
        out[f] = (np.array([tr.id for tr in tracks], dtype=np.int64),
                  np.array([tr.box for tr in tracks], dtype=np.float64).reshape(-1, 4))
    ms = (time.perf_counter() - t0) * 1000 / max(1, len(out))
    return out, {"fps": real_fps, "frames": len(out), "ms_per_frame": ms, "ids_created": run.ids_created}


def _xyxy_to_xywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)


def _ioa(hb_xywh: np.ndarray, regions_xyxy: np.ndarray) -> np.ndarray:
    """Porsi luas box (xywh) yang ada di dalam tiap area (xyxy)."""
    x1, y1 = hb_xywh[:, 0:1], hb_xywh[:, 1:2]
    x2, y2 = x1 + hb_xywh[:, 2:3], y1 + hb_xywh[:, 3:4]
    iw = np.clip(np.minimum(x2, regions_xyxy[None, :, 2]) - np.maximum(x1, regions_xyxy[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(y2, regions_xyxy[None, :, 3]) - np.maximum(y1, regions_xyxy[None, :, 1]), 0, None)
    area = np.maximum(hb_xywh[:, 2:3] * hb_xywh[:, 3:4], 1e-6)
    return iw * ih / area


def to_eval_frames(seq: Seq, hyp: dict) -> tuple[list, list]:
    """Siapkan gt dan hasil tracker per frame, dengan aturan resmi MOTChallenge:
    - gt yang dinilai: kelas pejalan kaki (1) dengan tanda 'dinilai' (kolom 7 = 1)
    - box tracker yang cocok (IoU >= 0,5) dengan orang diam, orang di kendaraan, pantulan, atau distraktor
      dibuang, jadi tidak dihitung salah maupun benar.
    - (klip CCTV) box tracker yang separuh lebih luasnya ada di area yang diabaikan (seq.ignore_regions) dibuang,
      dan hanya frame di seq.eval_frames yang dinilai kalau label tidak ada di setiap frame."""
    gt_by_f = {int(f): seq.gt[seq.gt[:, 0] == f] for f in np.unique(seq.gt[:, 0])}
    empty_gt = np.zeros((0, seq.gt.shape[1]))
    eval_frames = getattr(seq, "eval_frames", None)       # klip yang hanya dilabeli di sebagian frame
    regions = getattr(seq, "ignore_regions", None) or []  # area yang tidak dinilai: [(f_dari, f_sampai, xyxy)]
    gts, hyps = [], []
    for f in sorted(hyp):
        if eval_frames is not None and f not in eval_frames:
            continue
        g = gt_by_f.get(f, empty_gt)
        ids, boxes = hyp[f]
        hb = _xyxy_to_xywh(boxes)
        act = [r[2] for r in regions if r[0] <= f <= r[1]]
        if act and len(hb):
            inside = _ioa(hb, np.array(act, np.float64)).max(1) >= 0.5
            ids, hb = ids[~inside], hb[~inside]
        if len(g) and len(hb):
            s = mm.box_iou_xywh(g[:, 2:6], hb)
            s[s < 0.5 - mm.EPS] = 0
            rows, cols = linear_sum_assignment(-s)
            ok = s[rows, cols] > 0 + mm.EPS
            rows, cols = rows[ok], cols[ok]
            drop = cols[np.isin(g[rows, 7].astype(int), DISTRACTOR_CLASSES)]
            keep = np.setdiff1d(np.arange(len(hb)), drop)
            ids, hb = ids[keep], hb[keep]
        gk = g[(g[:, 6] != 0) & (g[:, 7] == 1)] if len(g) else g
        gts.append((gk[:, 1].astype(np.int64), gk[:, 2:6]))
        hyps.append((ids, hb))
    return gts, hyps


def evaluate(seq: Seq, hyp: dict) -> dict:
    g, h = to_eval_frames(seq, hyp)
    return mm.evaluate_sequence(g, h)


def run_benchmark(specs: dict[str, TrackerSpec], seqs: list[Seq], fps_list=(None,), progress=None) -> list[dict]:
    """Semua kombinasi tracker x sekuens x FPS. Hasil: list baris berisi angka mentah per sekuens.
    fps_list: None = FPS asli sekuens."""
    rows = []
    jobs = [(n, s, f) for n in specs for f in fps_list for s in seqs]
    it = progress(jobs) if progress else jobs
    for name, seq, fps in it:
        hyp, info = run_tracker(specs[name], seq, fps)
        rows.append({"tracker": name, "seq": seq.name, "detektor": seq.detector, "kamera": "diam" if seq.static else "bergerak",
                     "target_fps": fps or "asli", **info, "raw": evaluate(seq, hyp)})
    return rows


def summary(rows: list[dict], by=("tracker",), sort="HOTA") -> pd.DataFrame:
    """Gabungkan baris hasil run_benchmark per kelompok (mis. tracker, atau tracker + kamera)."""
    df = pd.DataFrame(rows)
    out = []
    for key, grp in df.groupby(list(by), sort=False):
        key = key if isinstance(key, tuple) else (key,)
        m = mm.summarize(mm.combine(list(grp["raw"])))
        out.append({**dict(zip(by, key)), **m, "ms/frame": float(np.average(grp["ms_per_frame"], weights=grp["frames"]))})
    res = pd.DataFrame(out).set_index(list(by))
    return res.sort_values(sort, ascending=False) if sort else res
