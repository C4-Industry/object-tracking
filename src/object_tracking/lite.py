"""Tracker orang ringan untuk FPS rendah (2 sampai 5 FPS per kamera).

Bedanya dengan ByteTrack Ultralytics yang dipakai pipeline sekarang:
  - Kalman memakai waktu nyata (detik), bukan "per frame". Kecepatan dalam piksel/detik,
    jadi model gerak tetap benar di 2 FPS maupun 25 FPS, dan saat frame dilewati.
  - Semua batas waktu dalam detik (berapa lama orang hilang tetap diingat, dsb.).
  - Deteksi skor rendah benar-benar dipakai di tahap kedua (bukan dibuang sebelum tracker).
  - Opsi buffered IoU (C-BIoU, WACV 2023): box diperbesar sementara saat matching, supaya orang
    yang bergeser jauh di antara dua frame (FPS rendah) tetap tersambung.
  - Opsi NSA: box ber-skor rendah (sering setengah tertutup) dipercaya lebih sedikit oleh Kalman.
  - predict(t): posisi tiap orang bisa ditebak di frame yang tidak dideteksi.
  - Opsi dari OC-SORT (CVPR 2023), semua default mati supaya hasil lama tetap sama:
      ORU: saat track yang sempat hilang ketemu lagi, kecepatan dihitung ulang dari observasi
           terakhir ke observasi baru (bukan dari tebakan Kalman yang sudah melenceng).
      OCR: tahap matching tambahan memakai box terakhir yang benar-benar terlihat.
  - Opsi DIoU: biaya matching ikut menghitung jarak pusat, jadi box yang sudah tidak bertumpuk
    (gerak jauh di FPS rendah) masih bisa disambung ke orang terdekat.
  - Opsi gerbang rasio tinggi: orang dekat (box besar) tidak boleh mengambil ID orang jauh (box kecil).

Tidak ada Re-ID: seragam pekerja mirip, dan brief melarang identifikasi orang.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment


def iou_matrix(a: np.ndarray, b: np.ndarray, buffer: float = 0.0) -> np.ndarray:
    """IoU antar box xyxy. buffer > 0: tiap box diperbesar buffer x lebar/tinggi di tiap sisi dulu."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    a = _expand(a, buffer)
    b = _expand(b, buffer)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return (inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)).astype(np.float32)


def diou_matrix(a: np.ndarray, b: np.ndarray, buffer: float = 0.0) -> np.ndarray:
    """DIoU = IoU - (jarak pusat / diagonal kotak gabungan)^2. Nilainya -1..1."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    iou = iou_matrix(a, b, buffer)
    a = _expand(a, buffer)
    b = _expand(b, buffer)
    acx, acy = (a[:, 0] + a[:, 2]) / 2, (a[:, 1] + a[:, 3]) / 2
    bcx, bcy = (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2
    d2 = (acx[:, None] - bcx[None, :]) ** 2 + (acy[:, None] - bcy[None, :]) ** 2
    ex1 = np.minimum(a[:, None, 0], b[None, :, 0])
    ey1 = np.minimum(a[:, None, 1], b[None, :, 1])
    ex2 = np.maximum(a[:, None, 2], b[None, :, 2])
    ey2 = np.maximum(a[:, None, 3], b[None, :, 3])
    c2 = np.maximum((ex2 - ex1) ** 2 + (ey2 - ey1) ** 2, 1e-9)
    return (iou - d2 / c2).astype(np.float32)


def nms(boxes: np.ndarray, scores: np.ndarray, iou_thr: float = 0.6) -> np.ndarray:
    """Indeks box yang dipertahankan. D-FINE (DETR) kadang mengeluarkan box orang dobel berskor rendah
    di atas orang yang sama; box dobel ini membuat tracker salah sambung di area ramai."""
    if len(boxes) == 0:
        return np.zeros(0, dtype=int)
    order = np.argsort(-scores)
    keep = []
    iou = iou_matrix(boxes, boxes)
    removed = np.zeros(len(boxes), dtype=bool)
    for i in order:
        if removed[i]:
            continue
        keep.append(i)
        removed |= iou[i] >= iou_thr
    return np.array(sorted(keep), dtype=int)


def _expand(boxes: np.ndarray, b: float) -> np.ndarray:
    if b <= 0:
        return boxes
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    return np.stack([boxes[:, 0] - b * w, boxes[:, 1] - b * h, boxes[:, 2] + b * w, boxes[:, 3] + b * h], axis=1)


def _xyxy_to_z(b) -> np.ndarray:
    x1, y1, x2, y2 = b
    return np.array([(x1 + x2) / 2, (y1 + y2) / 2, max(1.0, x2 - x1), max(1.0, y2 - y1)], dtype=np.float64)


@dataclass
class TrackerConfig:
    high_thresh: float = 0.50       # deteksi >= ini ikut tahap pertama
    low_thresh: float = 0.10        # deteksi di antara low dan high ikut tahap kedua (ByteTrack)
    new_thresh: float = 0.55        # skor minimal untuk membuat track baru
    min_iou: float = 0.30           # IoU minimal tahap pertama (setelah box diperbesar buffer)
    min_iou_low: float = 0.40       # IoU minimal tahap kedua (deteksi skor rendah)
    buffers: tuple = (0.1, 0.4)     # C-BIoU bertingkat. (0.0,) = IoU biasa
    max_lost_sec: float = 3.0       # orang yang hilang tetap diingat selama ini
    confirm_hits: int = 2           # track baru dianggap pasti setelah terlihat sekian kali
    coast_sec: float = 1.0          # di frame tanpa deteksi, track ditampilkan sampai sekian detik sejak terakhir terlihat
    lost_velocity_decay_sec: float = 1.0  # setelah hilang selama ini, kecepatan dianggap nol (cegah box "terbang")
    nsa: bool = True                # Kalman lebih tidak percaya box skor rendah
    std_pos: float = 0.05           # noise posisi relatif tinggi box
    std_vel: float = 0.6            # noise kecepatan relatif tinggi box, per detik
    std_meas: float = 0.05
    cost: str = "iou"               # "iou" atau "diou" (ikut jarak pusat; nilai bisa negatif, gerbang tetap min_iou)
    oru: bool = False               # OC-SORT ORU: kecepatan dihitung ulang saat track hilang ketemu lagi
    ocr: bool = False               # OC-SORT OCR: matching tambahan dari box terakhir yang terlihat
    ocr_min_iou: float = 0.30
    max_h_ratio: float = 0.0        # > 0: tolak pasangan kalau tinggi box berbeda lebih dari rasio ini (mis. 1.6)


@dataclass
class Track:
    id: int
    x: np.ndarray                   # [cx, cy, w, h, vcx, vcy, vw, vh] (kecepatan per detik)
    P: np.ndarray
    t: float                        # waktu state terakhir
    score: float
    hits: int = 1
    confirmed: bool = False
    last_det_t: float = 0.0
    first_t: float = 0.0
    det_idx: int = -1               # indeks deteksi yang cocok di update terakhir (-1 = tidak ada)
    last_obs: np.ndarray | None = None  # box xyxy terakhir yang benar-benar terlihat (untuk ORU/OCR)
    missed: int = 0                 # berapa update deteksi berturut-turut track ini tidak ketemu
    data: dict = field(default_factory=dict)  # tempat menyimpan status helm/vest dll.

    @property
    def box(self) -> np.ndarray:
        cx, cy, w, h = self.x[:4]
        return np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dtype=np.float32)


class LiteTracker:
    def __init__(self, cfg: TrackerConfig | None = None):
        self.cfg = cfg or TrackerConfig()
        self.tracks: list[Track] = []
        self._next_id = 1
        self.t = None
        self._prev_update_t = None

    # ---------- Kalman ----------
    def _predict_track(self, tr: Track, t: float) -> None:
        dt = max(0.0, t - tr.t)
        if dt == 0:
            return
        c = self.cfg
        lost_for = t - tr.last_det_t
        if lost_for > c.lost_velocity_decay_sec:
            tr.x[4:] = 0.0
        F = np.eye(8)
        F[:4, 4:] = np.eye(4) * dt
        h = max(tr.x[3], 1.0)
        q_pos = (c.std_pos * h) ** 2 * dt
        q_vel = (c.std_vel * h) ** 2 * dt
        Q = np.diag([q_pos, q_pos, q_pos, q_pos, q_vel, q_vel, q_vel * 0.25, q_vel * 0.25])
        tr.x = F @ tr.x
        tr.x[2:4] = np.maximum(tr.x[2:4], 2.0)
        tr.P = F @ tr.P @ F.T + Q
        tr.t = t

    def _update_track(self, tr: Track, box, score: float, t: float) -> None:
        c = self.cfg
        z = _xyxy_to_z(box)
        h = max(z[3], 1.0)
        r = (c.std_meas * h) ** 2
        if c.nsa:
            r *= max(0.1, 1.0 - float(score)) * 2.0
        R = np.eye(4) * r
        if c.oru and tr.missed > 0 and tr.last_obs is not None:
            # ORU: track sempat hilang. Kalman sudah berjalan sendiri dengan kecepatan lama, jadi
            # kecepatan dihitung ulang dari observasi terakhir ke observasi baru (gerak rata-rata selama hilang).
            dt = max(t - tr.last_det_t, 1e-3)
            z0 = _xyxy_to_z(tr.last_obs)
            v = (z - z0) / dt
            v[2:] *= 0.5
            tr.x = np.concatenate([z, v])
            tr.P = np.diag([(c.std_meas * h) ** 2] * 4 + [(c.std_vel * h) ** 2] * 4)
            self._finish_update(tr, box, score, t)
            return
        H = np.zeros((4, 8))
        H[:, :4] = np.eye(4)
        S = H @ tr.P @ H.T + R
        K = tr.P @ H.T @ np.linalg.inv(S)
        tr.x = tr.x + K @ (z - H @ tr.x)
        tr.P = (np.eye(8) - K @ H) @ tr.P
        self._finish_update(tr, box, score, t)

    def _finish_update(self, tr: Track, box, score: float, t: float) -> None:
        c = self.cfg
        tr.last_obs = np.asarray(box, dtype=np.float32).copy()
        tr.missed = 0
        tr.score = float(score)
        tr.hits += 1
        tr.last_det_t = t
        if tr.hits >= c.confirm_hits:
            tr.confirmed = True

    def _new_track(self, box, score: float, t: float, first_frame: bool) -> Track:
        c = self.cfg
        z = _xyxy_to_z(box)
        h = z[3]
        x = np.concatenate([z, np.zeros(4)])
        P = np.diag([(2 * c.std_pos * h) ** 2] * 4 + [(3 * c.std_vel * h) ** 2] * 4)
        tr = Track(id=self._next_id, x=x, P=P, t=t, score=float(score), last_det_t=t, first_t=t,
                   confirmed=first_frame or c.confirm_hits <= 1, last_obs=np.asarray(box, dtype=np.float32).copy())
        self._next_id += 1
        return tr

    # ---------- API ----------
    def predict(self, t: float) -> list[Track]:
        """Frame tanpa deteksi: geser semua track ke waktu t. Mengembalikan track yang layak ditampilkan."""
        for tr in self.tracks:
            self._predict_track(tr, t)
            tr.det_idx = -1
        self.t = t
        return [tr for tr in self.tracks if tr.confirmed and (t - tr.last_det_t) <= self.cfg.coast_sec]

    def update(self, boxes: np.ndarray, scores: np.ndarray, t: float) -> list[Track]:
        """Frame dengan deteksi. boxes: Nx4 xyxy, scores: N. Mengembalikan track yang cocok di frame ini."""
        c = self.cfg
        first_frame = self.t is None
        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        for tr in self.tracks:
            self._predict_track(tr, t)
            tr.det_idx = -1
        self.t = t

        high = np.where(scores >= c.high_thresh)[0]
        low = np.where((scores >= c.low_thresh) & (scores < c.high_thresh))[0]
        confirmed = [tr for tr in self.tracks if tr.confirmed]
        tentative = [tr for tr in self.tracks if not tr.confirmed]

        # Tahap 1: track pasti (aktif + hilang) vs deteksi skor tinggi, bertingkat per buffer
        un_tr, un_det = confirmed, list(high)
        for b in c.buffers:
            m, un_tr, un_det = self._match(un_tr, un_det, boxes, scores, c.min_iou, b, fuse=True)
            for tr, d in m:
                self._update_track(tr, boxes[d], scores[d], t)
                tr.det_idx = int(d)

        # Tahap 2: track yang masih aktif (cocok di update sebelumnya) vs deteksi skor rendah
        prev = self._prev_update_t
        recent = [tr for tr in un_tr if prev is not None and tr.last_det_t >= prev - 1e-6]
        m, _, _ = self._match(recent, list(low), boxes, scores, c.min_iou_low, c.buffers[-1], fuse=False)
        for tr, d in m:
            self._update_track(tr, boxes[d], scores[d], t)
            tr.det_idx = int(d)
        un_tr = [tr for tr in un_tr if tr.det_idx < 0]

        # Tahap 2b (OCR): sisa track pasti vs sisa deteksi skor tinggi, memakai box terakhir yang terlihat
        if c.ocr and un_tr and un_det:
            m, un_tr, un_det = self._match(un_tr, un_det, boxes, scores, c.ocr_min_iou, c.buffers[-1],
                                           fuse=True, use_last_obs=True)
            for tr, d in m:
                self._update_track(tr, boxes[d], scores[d], t)
                tr.det_idx = int(d)

        # Tahap 3: track baru (belum pasti) vs sisa deteksi skor tinggi
        m, un_tent, un_det = self._match(tentative, un_det, boxes, scores, 0.3, 0.0, fuse=True)
        for tr, d in m:
            self._update_track(tr, boxes[d], scores[d], t)
            tr.det_idx = int(d)

        for tr in self.tracks:
            if tr.det_idx < 0:
                tr.missed += 1

        # Hapus: track baru yang tidak tersambung, dan track yang sudah hilang terlalu lama
        dead = {id(tr) for tr in un_tent}
        self.tracks = [tr for tr in self.tracks if id(tr) not in dead and (t - tr.last_det_t) <= c.max_lost_sec]

        # Track baru dari deteksi skor tinggi yang tersisa
        for d in un_det:
            if scores[d] >= c.new_thresh:
                tr = self._new_track(boxes[d], scores[d], t, first_frame)
                tr.det_idx = int(d)
                self.tracks.append(tr)

        self._prev_update_t = t
        return [tr for tr in self.tracks if tr.det_idx >= 0 and tr.confirmed]

    def _match(self, tracks, det_idx, boxes, scores, min_iou, buffer, fuse, use_last_obs=False):
        if not tracks or not det_idx:
            return [], list(tracks), list(det_idx)
        c = self.cfg
        tb = np.stack([tr.last_obs if use_last_obs and tr.last_obs is not None else tr.box for tr in tracks])
        db = boxes[det_idx]
        iou = diou_matrix(tb, db, buffer) if c.cost == "diou" else iou_matrix(tb, db, buffer)
        if c.max_h_ratio > 0:
            th = np.maximum(tb[:, 3] - tb[:, 1], 1.0)
            dh = np.maximum(db[:, 3] - db[:, 1], 1.0)
            ratio = np.maximum(th[:, None] / dh[None, :], dh[None, :] / th[:, None])
            iou = np.where(ratio > c.max_h_ratio, -1.0, iou)
        if not fuse:
            sim = iou
        elif c.cost == "diou":
            sim = (iou + 1.0) * scores[det_idx][None, :] - 1.0   # DIoU bisa negatif: geser ke 0..2 dulu
        else:
            sim = iou * scores[det_idx][None, :]
        cost = 1.0 - sim
        r, cidx = linear_sum_assignment(cost)
        matches, used_t, used_d = [], set(), set()
        for i, j in zip(r, cidx):
            if iou[i, j] >= min_iou and (c.cost == "diou" or sim[i, j] > 0):
                matches.append((tracks[i], det_idx[j]))
                used_t.add(i)
                used_d.add(j)
        return (matches,
                [tr for i, tr in enumerate(tracks) if i not in used_t],
                [d for j, d in enumerate(det_idx) if j not in used_d])
