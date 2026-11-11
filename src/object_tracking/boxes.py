"""Struktur box kecil yang dipakai bersama. Hanya butuh numpy."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BBox:
    """Satu deteksi dalam koordinat piksel frame asli. cls_id: 0 orang, 1 helm, 2 vest."""

    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls_id: int

    @property
    def w(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def h(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def cx(self) -> float:
        return (self.x1 + self.x2) / 2.0

    @property
    def cy(self) -> float:
        return (self.y1 + self.y2) / 2.0


class TrackerBoxes:
    """Adaptor input untuk tracker Ultralytics: atribut `xywh`, `xyxy`, `conf`, `cls`.
    Sama dengan TrackerBoxes di pipeline utama, disalin supaya repo ini tidak bergantung ke sana."""

    def __init__(self, xywh=None, conf=None, cls=None):
        self.xywh = np.asarray(xywh if xywh is not None else np.empty((0, 4)), dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(conf if conf is not None else np.empty(0), dtype=np.float32).reshape(-1)
        self.cls = np.asarray(cls if cls is not None else np.zeros(len(self.conf)), dtype=np.float32).reshape(-1)

    @property
    def xyxy(self) -> np.ndarray:
        if len(self.xywh) == 0:
            return np.empty((0, 4), dtype=np.float32)
        cx, cy, w, h = self.xywh.T
        return np.stack([cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0], axis=-1)

    def __getitem__(self, idx):
        return TrackerBoxes(self.xywh[idx], self.conf[idx], self.cls[idx])

    def __len__(self):
        return len(self.conf)
