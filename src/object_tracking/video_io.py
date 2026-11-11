"""Baca video dan tulis video: dipakai evaluasi, perbandingan tracker, dan aplikasi Gradio."""

from __future__ import annotations

import time
from pathlib import Path

import cv2

VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm"}


def iter_frames(path: str | Path, fps: float, max_sec: float | None = None, start_sec: float = 0.0):
    """Ambil frame pada `fps` (kira-kira). Frame yang dilewati hanya di-grab (tetap di-decode oleh codec,
    tapi tidak dikonversi). Yield (t_detik, frame_bgr, decode_ms)."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(src_fps / fps)))
    if start_sec > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_sec * src_fps))
    k = int(start_sec * src_fps)
    try:
        while True:
            t0 = time.perf_counter()
            if not cap.grab():
                break
            if k % step == 0:
                ok, frame = cap.retrieve()
                ms = (time.perf_counter() - t0) * 1000
                t = k / src_fps
                if max_sec is not None and t - start_sec > max_sec:
                    break
                if ok:
                    yield t, frame, ms
            k += 1
    finally:
        cap.release()


def decode_cost(path: str | Path, seconds: float = 10.0) -> dict:
    """Biaya decode per detik video untuk satu kamera (semua frame harus di-decode, walau hanya sedikit yang dipakai)."""
    cap = cv2.VideoCapture(str(path))
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n, t0 = 0, time.perf_counter()
    while n < int(seconds * src_fps) and cap.grab():
        n += 1
    el = time.perf_counter() - t0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    return {"src_fps": src_fps, "frames": n, "ms_per_frame": el * 1000 / max(n, 1),
            "ms_per_video_second": el * 1000 / max(n / src_fps, 1e-9), "width": width, "height": height}


def video_meta(path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    m = {"fps": float(fps), "frames": n, "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
         "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), "duration": n / fps if n > 0 else 0.0}
    cap.release()
    return m


def list_videos(dirs) -> list[Path]:
    out = []
    for d in dirs:
        d = Path(d)
        if d.exists():
            out += sorted(p for p in d.iterdir() if p.suffix.lower() in VIDEO_EXT)
    return out


def open_writer(path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    """H.264 (avc1) kalau tersedia, supaya video bisa diputar di VS Code/browser. Kalau tidak, mp4v."""
    for cc in ("avc1", "mp4v"):
        w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*cc), fps, size)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError(f"tidak bisa menulis video: {path}")
