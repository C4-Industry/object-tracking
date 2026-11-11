"""API untuk memasang Lite tracker ke pipeline lain (misalnya pipeline utama hen-detect-ppe).

Hanya butuh numpy dan scipy. Untuk dipasang di repo lain cukup salin dua file: `lite.py` dan `integration.py`.

Dua cara pakai:

1. `PersonTracker`: API umum, satu instance per kamera.

       trk = PersonTracker.tuned()                       # parameter "Lite tuned"
       for t, frame in kamera:                            # t = waktu frame dalam detik
           boxes, scores = detektor(frame)                # semua orang, skor >= 0,05 (jangan disaring 0,5 dulu)
           for p in trk.update(boxes, scores, t):
               p.id, p.box, p.det_idx                     # det_idx = indeks di boxes/scores input

2. `LiteForComplianceTracker`: pengganti langsung tracker Ultralytics di dalam `ComplianceTracker`
   (`src/detect/pipeline/tracking.py` di hen-detect-ppe). Input dan outputnya sama dengan BYTETracker
   Ultralytics, ditambah `timestamp_sec`. Langkahnya ada di docs/INTEGRASI.md.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

from .lite import LiteTracker, Track, TrackerConfig, nms

# Parameter "Lite tuned" (notebook tuning CCTV, 10 klip tuning, dinilai di 5 klip kamera test).
# Sama dengan configs/tracker_tuned.json, ditulis di sini supaya file ini bisa dipakai tanpa folder configs.
LITE_TUNED_PARAMS = {
    "high_thresh": 0.517, "low_thresh": 0.1, "new_thresh": 0.535,
    "min_iou": 0.365, "min_iou_low": 0.386, "buffers": (0.1,),
    "max_lost_sec": 4.159, "lost_velocity_decay_sec": 1.455, "confirm_hits": 2,
    "std_vel": 0.886, "std_pos": 0.099, "nsa": False, "oru": True, "ocr": True,
    "cost": "iou", "max_h_ratio": 0.0,
}
LITE_TUNED_INPUT = {"person_min": 0.05, "nms_iou": 0.7}   # penyaring deteksi orang sebelum masuk tracker
LITE_TUNED_DETECTOR = "dfine-n-1248x704-commercial-v3.onnx"   # ambang skor di atas disetel untuk detektor ini


def load_lite_config(path: str | Path | None = None, name: str = "Lite tuned") -> tuple[TrackerConfig, dict]:
    """Konfigurasi Lite: (TrackerConfig, {"person_min", "nms_iou"}).
    path None = parameter bawaan di file ini. path = file JSON hasil tuning (configs/tracker_tuned.json)."""
    if path is None:
        params, inp = dict(LITE_TUNED_PARAMS), dict(LITE_TUNED_INPUT)
    else:
        spec = json.loads(Path(path).read_text())["specs"][name]["spec"]
        if spec.get("kind", "lite") != "lite":
            raise ValueError(f"'{name}' bukan tracker lite (kind={spec.get('kind')})")
        params = dict(spec["params"])
        inp = {"person_min": float(spec.get("person_min", 0.1)), "nms_iou": float(spec.get("nms_iou", 0.6))}
    known = {f.name for f in fields(TrackerConfig)}
    unknown = set(params) - known
    if unknown:
        raise ValueError(f"parameter tidak dikenal: {sorted(unknown)}")
    if "buffers" in params:
        params["buffers"] = tuple(params["buffers"])
    return TrackerConfig(**params), inp


@dataclass
class TrackOutput:
    """Satu orang yang dilacak di frame ini."""

    id: int
    box: np.ndarray        # xyxy, piksel frame asli
    score: float           # skor deteksi terakhir
    det_idx: int           # indeks deteksi di input update() yang cocok, -1 kalau posisi tebakan
    predicted: bool        # True = tidak ada deteksi di frame ini, posisi ditebak Kalman
    age_sec: float         # sudah berapa lama orang ini dilacak
    track: Track           # objek track asli. track.data bisa dipakai menyimpan status helm/vest

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return tuple(float(v) for v in self.box)


class PersonTracker:
    """Lite tracker untuk SATU kamera. Buat satu instance per kamera.

    - Waktu dalam detik (`timestamp_sec`), boleh dari frame_idx / fps atau jam kamera. Harus naik.
    - Berikan SEMUA deteksi orang (skor >= 0,05). Penyaringan skor, NMS box dobel, dan pembagian deteksi
      yakin vs ragu dilakukan di dalam.
    - Output hanya orang yang sudah pasti (terlihat minimal 2 kali) dan cocok dengan deteksi di frame ini.
    """

    def __init__(self, config: TrackerConfig | None = None, person_min: float = 0.10, nms_iou: float = 0.6):
        self.cfg = config or TrackerConfig()
        self.person_min = float(person_min)
        self.nms_iou = float(nms_iou)
        self.reset()

    @classmethod
    def tuned(cls, path: str | Path | None = None, name: str = "Lite tuned") -> "PersonTracker":
        cfg, inp = load_lite_config(path, name)
        return cls(cfg, **inp)

    def reset(self) -> None:
        self.lite = LiteTracker(self.cfg)
        self.shown_ids: set[int] = set()
        self._last_t: float | None = None

    @property
    def ids_shown(self) -> int:
        """Jumlah ID berbeda yang pernah dikeluarkan (bukan termasuk kandidat yang tidak pernah pasti)."""
        return len(self.shown_ids)

    def _check_time(self, t: float) -> float:
        t = float(t)
        if self._last_t is not None and t < self._last_t:
            raise ValueError(f"timestamp mundur ({t} < {self._last_t}). Pakai satu PersonTracker per kamera.")
        self._last_t = t
        return t

    def _out(self, tr: Track, t: float, det_idx: int, predicted: bool) -> TrackOutput:
        self.shown_ids.add(tr.id)
        return TrackOutput(tr.id, tr.box, float(tr.score), det_idx, predicted, float(t - tr.first_t), tr)

    def update(self, boxes, scores, timestamp_sec: float) -> list[TrackOutput]:
        """Frame dengan deteksi. boxes: Nx4 xyxy, scores: N (deteksi orang saja)."""
        t = self._check_time(timestamp_sec)
        b = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        s = np.asarray(scores, dtype=np.float32).reshape(-1)
        if len(b) != len(s):
            raise ValueError(f"jumlah box ({len(b)}) dan skor ({len(s)}) berbeda")
        keep = np.where(s >= self.person_min)[0]
        if self.nms_iou > 0 and len(keep) > 1:
            keep = keep[nms(b[keep], s[keep], self.nms_iou)]
        tracks = self.lite.update(b[keep], s[keep], t)
        return [self._out(tr, t, int(keep[tr.det_idx]), False) for tr in tracks]

    def predict(self, timestamp_sec: float) -> list[TrackOutput]:
        """Frame tanpa deteksi (detektor dilewati): posisi ditebak, sampai coast_sec sejak terakhir terlihat."""
        t = self._check_time(timestamp_sec)
        return [self._out(tr, t, -1, True) for tr in self.lite.predict(t)]


class CameraTrackers:
    """Kumpulan PersonTracker, satu per kamera, dibuat otomatis saat kamera pertama kali muncul."""

    def __init__(self, factory=None):
        self.factory = factory or PersonTracker.tuned
        self.by_camera: dict = {}

    def __getitem__(self, camera_id) -> PersonTracker:
        if camera_id not in self.by_camera:
            self.by_camera[camera_id] = self.factory()
        return self.by_camera[camera_id]

    def update(self, camera_id, boxes, scores, timestamp_sec: float) -> list[TrackOutput]:
        return self[camera_id].update(boxes, scores, timestamp_sec)

    def reset(self, camera_id=None) -> None:
        for k in ([camera_id] if camera_id is not None else list(self.by_camera)):
            if k in self.by_camera:
                self.by_camera[k].reset()


class LiteForComplianceTracker:
    """Pengganti tracker Ultralytics (BYTETracker dan sejenisnya) di dalam ComplianceTracker.

    Input sama: objek dengan atribut `xyxy` (atau `xywh`) dan `conf` (TrackerBoxes).
    Output sama: array (N, 8) [x1, y1, x2, y2, track_id, score, cls, idx], idx = indeks deteksi input.
    Tambahan: `timestamp_sec`. Kalau tidak diberikan, waktu dihitung dari nomor frame / fps (hanya benar
    kalau setiap frame diproses tanpa ada yang dilewati).
    """

    def __init__(self, fps: float = 25.0, config: TrackerConfig | None = None, person_min: float | None = None,
                 nms_iou: float | None = None, config_path: str | Path | None = None):
        cfg, inp = load_lite_config(config_path)
        self.fps = float(fps)
        self.tracker = PersonTracker(config or cfg,
                                     person_min=inp["person_min"] if person_min is None else person_min,
                                     nms_iou=inp["nms_iou"] if nms_iou is None else nms_iou)
        self.frame_idx = 0
        self._warned = False

    def reset(self) -> None:
        self.tracker.reset()
        self.frame_idx = 0

    def update(self, dets, img=None, timestamp_sec: float | None = None) -> np.ndarray:
        self.frame_idx += 1
        if timestamp_sec is None:
            if not self._warned:
                warnings.warn("LiteForComplianceTracker: timestamp_sec tidak diberikan, memakai frame_idx / fps. "
                              "Berikan timestamp_sec supaya benar saat ada frame yang dilewati.", stacklevel=2)
                self._warned = True
            timestamp_sec = self.frame_idx / self.fps
        if dets is None or len(dets) == 0:
            boxes, scores = np.zeros((0, 4), np.float32), np.zeros(0, np.float32)
        else:
            boxes, scores = np.asarray(dets.xyxy, np.float32), np.asarray(dets.conf, np.float32)
        out = self.tracker.update(boxes, scores, timestamp_sec)
        if not out:
            return np.zeros((0, 8), dtype=np.float32)
        return np.array([[*o.box, o.id, o.score, 0.0, o.det_idx] for o in out], dtype=np.float32)
