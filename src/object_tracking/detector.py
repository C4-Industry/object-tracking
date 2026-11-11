"""Detektor D-FINE berbasis ONNX Runtime, dengan pilihan runtime CPU atau CoreML (Apple).

Logika pre/post-processing sama dengan detektor D-FINE di pipeline utama (hen-detect-ppe)
(letterbox, un-letterbox ke koordinat asli), tapi provider bisa dipilih
dan waktu tiap sub-tahap dicatat terpisah.

Ukuran input dibaca dari ONNX, jadi model persegi (960x960) dan persegi panjang
(544x960, 704x1248) sama-sama didukung. Cara resize bisa dipilih:
  - "area"   : cv2.INTER_AREA. Hasilnya mendekati resize PIL yang dipakai saat training.
               Di test commercial_combined_v3, AP50 helm naik ~3 poin dibanding "linear".
  - "linear" : cv2.INTER_LINEAR (cara lama).
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

from .boxes import BBox

COREML_UNITS = ("ALL", "CPUAndGPU", "CPUAndNeuralEngine", "CPUOnly")
RESIZE_INTERP = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR}


def letterbox(
    rgb: np.ndarray, target_h: int, target_w: int | None = None, resize: str = "linear"
) -> tuple[np.ndarray, float, int, int]:
    """Skala proporsional, pad hitam di tengah. target_w None = persegi (target_h x target_h)."""
    target_w = target_h if target_w is None else target_w
    h, w = rgb.shape[:2]
    scale = min(target_w / float(w), target_h / float(h))
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = RESIZE_INTERP.get(resize, cv2.INTER_LINEAR)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=interp)
    pad_left, pad_top = (target_w - new_w) // 2, (target_h - new_h) // 2
    canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
    return canvas, scale, pad_left, pad_top


def model_input_hw(model_path: str | Path) -> tuple[int, int] | None:
    """Tinggi dan lebar input dari ONNX, atau None kalau dimensinya dinamis."""
    s = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    shape = s.get_inputs()[0].shape
    if len(shape) == 4 and isinstance(shape[2], int) and isinstance(shape[3], int):
        return int(shape[2]), int(shape[3])
    return None


def build_session(
    model_path: str | Path,
    runtime: str = "cpu",
    coreml_compute_units: str = "ALL",
    cpu_threads: int = 0,
) -> ort.InferenceSession:
    """Buat sesi ONNX Runtime. Dimensi dinamis dikunci ke 1 supaya CoreML bisa dipakai."""
    model_path = str(model_path)
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if cpu_threads and cpu_threads > 0:
        so.intra_op_num_threads = int(cpu_threads)

    # Kunci semua dimensi dinamis (batch dll) ke 1
    probe = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    for node in list(probe.get_inputs()) + list(probe.get_outputs()):
        for d in node.shape:
            if isinstance(d, str):
                try:
                    so.add_free_dimension_override_by_name(d, 1)
                except Exception:
                    pass
    del probe

    providers: list = ["CPUExecutionProvider"]
    if runtime == "coreml":
        if "CoreMLExecutionProvider" not in ort.get_available_providers():
            raise RuntimeError(
                "CoreML tidak tersedia di onnxruntime ini. CoreML hanya ada di macOS. Pilih runtime 'cpu'."
            )
        units = coreml_compute_units if coreml_compute_units in COREML_UNITS else "ALL"
        providers = [
            ("CoreMLExecutionProvider", {"ModelFormat": "MLProgram", "MLComputeUnits": units}),
            "CPUExecutionProvider",
        ]
    return ort.InferenceSession(model_path, sess_options=so, providers=providers)


class OnnxDFineDetector:
    def __init__(
        self,
        model_path: str | Path,
        input_size: int | None = None,
        runtime: str = "cpu",
        coreml_compute_units: str = "ALL",
        cpu_threads: int = 0,
        resize: str = "linear",
    ):
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model detektor tidak ditemukan: {self.model_path}")
        hw = model_input_hw(self.model_path)
        if hw is None:  # model dinamis: pakai input_size dari config (persegi)
            side = int(input_size or 960)
            hw = (side, side)
        self.input_h, self.input_w = hw
        self.input_size = max(hw)  # tetap ada untuk kode lama yang membacanya
        self.resize = resize if resize in RESIZE_INTERP else "linear"
        self.runtime = runtime
        self.session = build_session(self.model_path, runtime, coreml_compute_units, cpu_threads)
        self.input_names = [i.name for i in self.session.get_inputs()]
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.active_providers = self.session.get_providers()
        self.last_timing: dict[str, float] = {}

    def detect(self, frame_bgr: np.ndarray, conf_thresh: float) -> tuple[list[BBox], list[BBox], list[BBox]]:
        persons, helmets, vests, timing = self.detect_timed(frame_bgr, conf_thresh)
        self.last_timing = timing
        return persons, helmets, vests

    def detect_timed(self, frame_bgr: np.ndarray, conf_thresh: float):
        """Sama dengan detect, tapi waktu dikembalikan langsung (aman dipakai banyak thread)."""
        t0 = time.perf_counter()
        H, W = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        padded, scale, pad_left, pad_top = letterbox(rgb, self.input_h, self.input_w, self.resize)
        im = np.ascontiguousarray(np.transpose(padded.astype(np.float32) / 255.0, (2, 0, 1))[None, ...])
        sizes = np.array([[self.input_w, self.input_h]], dtype=np.int64)  # urutan [W, H] seperti saat training
        t1 = time.perf_counter()

        outputs = self.session.run(self.output_names, {self.input_names[0]: im, self.input_names[1]: sizes})
        t2 = time.perf_counter()

        out = dict(zip(self.output_names, outputs))
        labels = out.get("labels", outputs[0])[0]
        boxes = out.get("boxes", outputs[1])[0].astype(np.float32).copy()
        scores = out.get("scores", outputs[2])[0]

        boxes[:, [0, 2]] = np.clip((boxes[:, [0, 2]] - pad_left) / scale, 0.0, float(W))
        boxes[:, [1, 3]] = np.clip((boxes[:, [1, 3]] - pad_top) / scale, 0.0, float(H))

        persons: list[BBox] = []
        helmets: list[BBox] = []
        vests: list[BBox] = []
        mask = scores >= conf_thresh
        for b, conf, cls_id in zip(boxes[mask], scores[mask], labels[mask]):
            bb = BBox(x1=float(b[0]), y1=float(b[1]), x2=float(b[2]), y2=float(b[3]), conf=float(conf), cls_id=int(cls_id))
            if cls_id == 0:
                persons.append(bb)
            elif cls_id == 1:
                helmets.append(bb)
            elif cls_id == 2:
                vests.append(bb)
        t3 = time.perf_counter()

        timing = {
            "det_pre_ms": (t1 - t0) * 1000,
            "det_infer_ms": (t2 - t1) * 1000,
            "det_post_ms": (t3 - t2) * 1000,
        }
        return persons, helmets, vests, timing
