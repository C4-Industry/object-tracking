"""Evaluasi tracking tanpa label ID manual.

Cara kerja:
  1. Detektor dijalankan SEKALI per klip pada FPS referensi yang cukup tinggi (default 12,5 FPS),
     hasilnya disimpan (cache). Semua varian setting berikutnya memakai cache ini, jadi cepat.
  2. Referensi: ByteTrack Ultralytics (implementasi independen) pada FPS referensi, diberi semua orang
     dengan skor >= 0,1. Di FPS setinggi ini gerak antar frame kecil, jadi ID-nya relatif stabil.
  3. Tiap varian (FPS proses, deteksi tiap N frame, jenis tracker) dibandingkan dengan referensi di
     waktu yang sama: IDF1, ID switch, cakupan, dan jumlah ID yang dibuat.

Keterbatasan: referensi bukan label manusia. Angka di sini mengukur seberapa dekat varian murah
ke tracking "mahal", bukan kebenaran mutlak. Untuk paper tetap perlu label ID manual (CVAT).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .boxes import TrackerBoxes
from .lite import iou_matrix, nms
from .video_io import decode_cost, iter_frames  # noqa: F401  (diekspor ulang supaya bisa diimpor dari sini)


# ---------------- cache deteksi ----------------

def build_det_cache(detector, video: str | Path, ref_fps: float = 12.5, max_sec: float | None = 60.0,
                    conf: float = 0.05, progress: bool = True) -> dict:
    """Jalankan detektor pada FPS referensi. Hasil: {'t': [...], 'dets': [array Nx6 (cls, score, x1, y1, x2, y2)], ...}."""
    ts, dets, det_ms, dec_ms = [], [], [], []
    it = iter_frames(video, ref_fps, max_sec)
    if progress:
        try:
            from tqdm.auto import tqdm
            it = tqdm(it, desc=Path(video).name, leave=False, total=int((max_sec or 60) * ref_fps))
        except ImportError:
            pass
    for t, frame, ms in it:
        p, h, v, timing = detector.detect_timed(frame, conf_thresh=conf)
        rows = [(b.cls_id, b.conf, b.x1, b.y1, b.x2, b.y2) for b in p + h + v]
        ts.append(t)
        dets.append(np.array(rows, dtype=np.float32).reshape(-1, 6))
        det_ms.append(timing["det_pre_ms"] + timing["det_infer_ms"] + timing["det_post_ms"])
        dec_ms.append(ms)
    return {"video": str(video), "ref_fps": ref_fps, "t": np.array(ts), "dets": dets,
            "det_ms": np.array(det_ms), "decode_ms": np.array(dec_ms)}


def save_cache(cache: dict, path: str | Path) -> None:
    np.savez_compressed(path, t=cache["t"], det_ms=cache["det_ms"], decode_ms=cache["decode_ms"],
                        meta=np.array([cache["video"], str(cache["ref_fps"])]),
                        **{f"d{i}": d for i, d in enumerate(cache["dets"])})


def load_cache(path: str | Path) -> dict:
    z = np.load(path, allow_pickle=False)
    n = len(z["t"])
    return {"video": str(z["meta"][0]), "ref_fps": float(z["meta"][1]), "t": z["t"],
            "dets": [z[f"d{i}"] for i in range(n)], "det_ms": z["det_ms"], "decode_ms": z["decode_ms"]}


def _persons(d: np.ndarray, min_score: float, nms_iou: float) -> np.ndarray:
    p = d[(d[:, 0] == 0) & (d[:, 1] >= min_score)]
    if nms_iou > 0 and len(p) > 1:
        p = p[nms(p[:, 2:6], p[:, 1], nms_iou)]
    return p


def _subsample(cache: dict, fps: float) -> list[int]:
    step = max(1, int(round(cache["ref_fps"] / fps)))
    return list(range(0, len(cache["t"]), step))


# ---------------- menjalankan tracker dari cache ----------------

def run_reference(cache: dict, person_low: float = 0.10, high: float = 0.50, buffer_sec: float = 3.0,
                  nms_iou: float = 0.6) -> list:
    """ByteTrack Ultralytics di semua frame cache (FPS referensi). Output: list (t, [(id, box)])."""
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker


    fps = cache["ref_fps"]
    args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=high, track_low_thresh=person_low,
                           new_track_thresh=high + 0.05, track_buffer=max(1, int(round(buffer_sec * fps))), match_thresh=0.8, fuse_score=True)
    trk = BYTETracker(args)  # track_buffer dalam frame (tanpa frame_rate supaya sama di semua versi Ultralytics)
    out = []
    for t, d in zip(cache["t"], cache["dets"]):
        p = _persons(d, person_low, nms_iou)
        if len(p):
            xywh = np.stack([(p[:, 2] + p[:, 4]) / 2, (p[:, 3] + p[:, 5]) / 2, p[:, 4] - p[:, 2], p[:, 5] - p[:, 3]], 1)
            res = trk.update(TrackerBoxes(xywh, p[:, 1], np.zeros(len(p))))
        else:
            res = trk.update(TrackerBoxes())
        out.append((float(t), [(int(r[4]), r[:4].astype(np.float32)) for r in res]))
    return out


def run_current_pipeline(cache: dict, fps: float, person_thr: float = 0.50, buffer_sec: float = 3.0,
                         nms_iou: float = 0.0) -> list:
    """Meniru tracker pipeline sekarang: ByteTrack Ultralytics, hanya orang skor >= 0,5, deteksi di setiap frame."""
    from types import SimpleNamespace

    from ultralytics.trackers.byte_tracker import BYTETracker


    idx = _subsample(cache, fps)
    args = SimpleNamespace(tracker_type="bytetrack", track_high_thresh=person_thr, track_low_thresh=0.10,
                           new_track_thresh=person_thr, track_buffer=max(1, int(round(buffer_sec * fps))),
                           match_thresh=0.8, fuse_score=True)
    trk = BYTETracker(args)  # frame_rate 30: max_time_lost = track_buffer frame, sama seperti engine.py
    out = []
    for i in idx:
        d = cache["dets"][i]
        p = _persons(d, person_thr, nms_iou)
        if len(p):
            xywh = np.stack([(p[:, 2] + p[:, 4]) / 2, (p[:, 3] + p[:, 5]) / 2, p[:, 4] - p[:, 2], p[:, 5] - p[:, 3]], 1)
            res = trk.update(TrackerBoxes(xywh, p[:, 1], np.zeros(len(p))))
        else:
            res = trk.update(TrackerBoxes())
        out.append((float(cache["t"][i]), [(int(r[4]), r[:4].astype(np.float32)) for r in res]))
    return out


# ---------------- metrik ----------------

def compare(ref: list, hyp: list, iou_thr: float = 0.5) -> dict:
    """IDF1, ID switch, cakupan dan presisi box, dibandingkan dengan referensi pada waktu yang sama."""
    ref_by_t = {round(t, 3): r for t, r in ref}
    pair_counts: dict[tuple[int, int], int] = {}
    n_ref = n_hyp = n_match = 0
    last_hyp_for_ref: dict[int, int] = {}
    idsw = 0
    hyp_ids_per_ref: dict[int, set] = {}
    hyp_ids = set()
    for t, H in hyp:
        R = ref_by_t.get(round(t, 3))
        if R is None:
            continue
        n_ref += len(R)
        n_hyp += len(H)
        hyp_ids.update(h[0] for h in H)
        if not R or not H:
            continue
        iou = iou_matrix(np.stack([r[1] for r in R]), np.stack([h[1] for h in H]))
        ri, hi = linear_sum_assignment(1 - iou)
        for i, j in zip(ri, hi):
            if iou[i, j] < iou_thr:
                continue
            rid, hid = R[i][0], H[j][0]
            n_match += 1
            pair_counts[(rid, hid)] = pair_counts.get((rid, hid), 0) + 1
            hyp_ids_per_ref.setdefault(rid, set()).add(hid)
            if rid in last_hyp_for_ref and last_hyp_for_ref[rid] != hid:
                idsw += 1
            last_hyp_for_ref[rid] = hid
    # IDF1: pasangan ID global terbaik
    idtp = 0
    if pair_counts:
        rids = sorted({k[0] for k in pair_counts})
        hids = sorted({k[1] for k in pair_counts})
        M = np.zeros((len(rids), len(hids)))
        for (r, h), c in pair_counts.items():
            M[rids.index(r), hids.index(h)] = c
        a, b = linear_sum_assignment(-M)
        idtp = int(M[a, b].sum())
    idf1 = 2 * idtp / max(n_ref + n_hyp, 1)
    ref_ids = {r[0] for _, R in ref for r in R}
    return {"IDF1": idf1, "id_switches": idsw, "coverage": n_match / max(n_ref, 1),
            "precision": n_match / max(n_hyp, 1), "ids_hyp": len(hyp_ids), "ids_ref": len(ref_ids),
            "frag_per_person": float(np.mean([len(v) for v in hyp_ids_per_ref.values()])) if hyp_ids_per_ref else 0.0,
            "ref_boxes": n_ref, "hyp_boxes": n_hyp}


def capacity(det_ms: float, fps: float, detect_every: int, decode_ms_per_video_sec: float = 0.0,
             cls_ms_per_sec: float = 0.0, track_ms: float = 0.2, budget_ms_per_sec: float = 1000.0) -> dict:
    """Perkiraan kamera per satu jalur proses (1 worker). Beban per kamera per detik = deteksi + tracker + classifier + decode."""
    det_per_sec = fps / max(detect_every, 1)
    per_cam = det_ms * det_per_sec + track_ms * fps + cls_ms_per_sec + decode_ms_per_video_sec
    return {"ms_per_camera_second": per_cam, "cameras_per_worker": budget_ms_per_sec / max(per_cam, 1e-6),
            "det_per_sec": det_per_sec}
