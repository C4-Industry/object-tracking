"""Adegan sintetis untuk mencoba tracker tanpa video dan tanpa detektor.

Beberapa orang berjalan di area kerja (tampak dari CCTV), lengkap dengan hal yang menyulitkan tracker:
- orang berpapasan,
- orang tertutup (forklift atau rak) selama beberapa detik, deteksinya hilang,
- deteksi ragu-ragu (skor rendah) saat orang setengah tertutup,
- box dobel berskor rendah dan deteksi palsu sesekali,
- posisi box sedikit bergetar.

Label ID yang benar (ground truth) ikut dibuat, jadi bisa dinilai dengan `object_tracking.metrics`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Person:
    pid: int
    start: tuple[float, float]       # posisi kaki (x, y) saat muncul
    velocity: tuple[float, float]    # piksel per detik
    t_in: float                      # detik muncul
    t_out: float                     # detik keluar
    height: float = 170.0            # tinggi box (piksel)
    occluded: list = field(default_factory=list)   # [(t_mulai, t_selesai)] tertutup penuh

    def box(self, t: float) -> np.ndarray:
        x = self.start[0] + self.velocity[0] * (t - self.t_in)
        y = self.start[1] + self.velocity[1] * (t - self.t_in)
        h = self.height * (0.75 + 0.25 * y / 720)          # makin bawah makin dekat kamera, makin besar
        w = 0.38 * h
        return np.array([x - w / 2, y - h, x + w / 2, y], dtype=np.float32)

    def state(self, t: float) -> str:
        if not (self.t_in <= t <= self.t_out):
            return "absent"
        for a, b in self.occluded:
            if a <= t < b:
                return "hidden"
            if a - 0.5 <= t < a or b <= t < b + 0.5:
                return "partial"
        return "visible"


def default_people() -> list[Person]:
    return [
        Person(1, (150, 520), (55, 0), 0.0, 20.0, occluded=[(6.0, 8.5)]),          # tertutup forklift 2,5 detik
        Person(2, (1150, 640), (-50, -4), 0.5, 20.0),                              # berpapasan dengan orang 1, lebih dekat kamera
        Person(3, (600, 330), (10, 3), 2.0, 20.0, height=130),                     # orang jauh, gerak pelan
        Person(4, (250, 690), (35, -6), 9.0, 20.0, occluded=[(13.0, 14.0)]),       # masuk di tengah video, tertutup 1 detik
        Person(5, (1050, 360), (0, 0), 0.0, 20.0, height=140),                     # berdiri diam di dekat rak
    ]


def make_scene(duration: float = 20.0, fps: float = 12.5, people: list[Person] | None = None, seed: int = 0,
               jitter: float = 4.0, miss_rate: float = 0.05, false_rate: float = 0.15) -> dict:
    """Hasil: {'t': [...], 'dets': [Nx5 xyxy+skor], 'gt': [(ids, boxes xyxy)], 'size': (W, H), 'fps'}."""
    rng = np.random.default_rng(seed)
    people = people or default_people()
    ts, dets, gts = [], [], []
    for k in range(int(duration * fps) + 1):
        t = k / fps
        rows, g_ids, g_boxes = [], [], []
        for p in people:
            st = p.state(t)
            if st == "absent":
                continue
            b = p.box(t)
            g_ids.append(p.pid)
            g_boxes.append(b)
            if st == "hidden" or rng.random() < miss_rate:
                continue
            score = rng.uniform(0.2, 0.45) if st == "partial" else rng.uniform(0.6, 0.95)
            rows.append([*(b + rng.normal(0, jitter, 4)), score])
            if rng.random() < 0.08:                            # box dobel berskor rendah di orang yang sama
                rows.append([*(b + rng.normal(0, 8, 4)), rng.uniform(0.15, 0.35)])
        if rng.random() < false_rate:                          # deteksi palsu (bayangan, manekin, dsb.)
            x, y = rng.uniform(100, 1180), rng.uniform(250, 700)
            rows.append([x - 25, y - 130, x + 25, y, rng.uniform(0.15, 0.6)])
        ts.append(t)
        dets.append(np.array(rows, dtype=np.float32).reshape(-1, 5))
        gts.append((np.array(g_ids, dtype=np.int64), np.array(g_boxes, dtype=np.float32).reshape(-1, 4)))
    return {"t": np.array(ts), "dets": dets, "gt": gts, "size": (1280, 720), "fps": fps, "people": people}


def to_cache(scene: dict) -> dict:
    """Format cache deteksi (reference_eval/zoo.run_spec): baris (cls, skor, x1, y1, x2, y2)."""
    rows = [np.concatenate([np.zeros((len(d), 1), np.float32), d[:, 4:5], d[:, :4]], axis=1) for d in scene["dets"]]
    return {"video": "sintetis", "ref_fps": scene["fps"], "t": scene["t"], "dets": rows,
            "det_ms": np.zeros(len(rows)), "decode_ms": np.zeros(len(rows))}


def evaluate(scene: dict, hyp: list) -> dict:
    """Nilai hasil tracker terhadap label sintetis. hyp: list (t, [(id, box xyxy)]) seperti zoo.run_spec."""
    from . import metrics as mm

    gt_by_t = {round(float(t), 3): g for t, g in zip(scene["t"], scene["gt"])}
    G, H = [], []
    for t, tracks in hyp:
        ids, boxes = gt_by_t[round(float(t), 3)]
        G.append((ids, _xywh(boxes)))
        H.append((np.array([i for i, _ in tracks], np.int64),
                  _xywh(np.array([b for _, b in tracks], np.float32).reshape(-1, 4))))
    return mm.summarize(mm.evaluate_sequence(G, H))


def _xywh(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    return np.stack([b[:, 0], b[:, 1], b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)


def id_timeline(scene: dict, hyp: list, iou_thr: float = 0.5) -> list[tuple[float, int, int]]:
    """Untuk tiap waktu dan tiap orang sebenarnya: ID tracker yang memegangnya (-1 = tidak terlacak)."""
    from scipy.optimize import linear_sum_assignment

    from .lite import iou_matrix

    gt_by_t = {round(float(t), 3): g for t, g in zip(scene["t"], scene["gt"])}
    rows = []
    for t, tracks in hyp:
        ids, boxes = gt_by_t[round(float(t), 3)]
        got = {int(g): -1 for g in ids}
        if len(ids) and tracks:
            iou = iou_matrix(boxes, np.array([b for _, b in tracks], np.float32).reshape(-1, 4))
            r, c = linear_sum_assignment(-iou)
            for i, j in zip(r, c):
                if iou[i, j] >= iou_thr:
                    got[int(ids[i])] = int(tracks[j][0])
        rows += [(float(t), g, tid) for g, tid in got.items()]
    return rows
