"""Satu antarmuka untuk semua tracker orang yang dibandingkan (Lite dan tracker Ultralytics).

Dipakai bersama oleh evaluasi dari cache deteksi (tabel IDF1) dan live view perbandingan, supaya yang
terlihat di video sama persis dengan yang dinilai di tabel.

    spec = PRESETS["Lite tuned"]
    run = TrackerRunner(spec, fps=4)
    for t, persons in ...:            # persons: array Nx5 [x1, y1, x2, y2, skor] semua deteksi orang
        tracks = run.step(t, persons) # list TrackOut(id, box, predicted)

Tracker Ultralytics bekerja per frame (bukan per detik), jadi semua batas waktu di spec ditulis dalam detik
lalu diubah ke jumlah frame sesuai FPS proses. Tracker Ultralytics berbagi satu penghitung ID global,
jadi ID dari tracker itu dipetakan ulang per runner (1, 2, 3, ...).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .boxes import TrackerBoxes
from .lite import LiteTracker, TrackerConfig, nms

ULTRA_KINDS = ("bytetrack", "botsort", "ocsort", "tracktrack", "fasttrack")
# parameter Ultralytics yang satuannya frame (default dibuat untuk 30 FPS), diskalakan ke FPS proses
_FRAME_PARAMS = ("delta_t", "reset_velocity_offset_occ", "reset_pos_offset_occ", "active_occ_to_lost_thresh",
                 "occ_reappear_window", "min_track_len")


@dataclass
class TrackerSpec:
    name: str
    kind: str = "lite"                 # "lite" atau salah satu ULTRA_KINDS
    params: dict = field(default_factory=dict)   # field TrackerConfig (lite) atau isi yaml tracker Ultralytics
    person_min: float = 0.10           # deteksi orang dengan skor di bawah ini dibuang sebelum tracker
    nms_iou: float = 0.60              # box orang dobel dibuang (0 = mati)
    buffer_sec: float = 3.0            # Ultralytics: berapa lama track hilang tetap diingat
    det_every: int = 1                 # lite saja: detektor tiap N frame, frame lain ditebak Kalman

    def with_params(self, name: str | None = None, **kw) -> "TrackerSpec":
        return replace(self, name=name or self.name, params={**self.params, **kw})

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "TrackerSpec":
        return TrackerSpec(**d)


@dataclass
class TrackOut:
    id: int
    box: np.ndarray                    # xyxy, piksel frame asli
    predicted: bool                    # True = posisi tebakan (frame tanpa deteksi)
    score: float = 0.0


def _ultra_args(kind: str, params: dict, fps: float, buffer_sec: float) -> SimpleNamespace:
    from ultralytics.utils import YAML
    from ultralytics.utils import ROOT

    base = YAML.load(ROOT / "cfg" / "trackers" / f"{kind}.yaml")
    base.update({"gmc_method": "none", "with_reid": False})   # CCTV diam, tanpa Re-ID
    for k in _FRAME_PARAMS:
        if k in base:
            base[k] = max(1, int(round(base[k] * fps / 30.0)))
    base.update(params)
    base["track_buffer"] = max(1, int(round(buffer_sec * fps)))
    base["tracker_type"] = kind
    return SimpleNamespace(**base)


def _make_ultra(kind: str, args: SimpleNamespace):
    if kind == "bytetrack":
        from ultralytics.trackers.byte_tracker import BYTETracker as T
    elif kind == "botsort":
        from ultralytics.trackers.bot_sort import BOTSORT as T
    elif kind == "ocsort":
        from ultralytics.trackers.oc_sort import OCSORT as T
    elif kind == "tracktrack":
        from ultralytics.trackers.track_tracker import TRACKTRACK as T
    elif kind == "fasttrack":
        from ultralytics.trackers.fast_tracker import FASTTracker as T
    else:
        raise ValueError(f"tracker tidak dikenal: {kind}")
    return T(args)


class TrackerRunner:
    def __init__(self, spec: TrackerSpec, fps: float):
        self.spec = spec
        self.fps = float(fps)
        self.frame_i = 0
        self._id_map: dict[int, int] = {}
        if spec.kind == "lite":
            self.lite = LiteTracker(TrackerConfig(**spec.params))
            self.ultra = None
        else:
            if spec.det_every != 1:
                raise ValueError("det_every > 1 hanya untuk tracker lite (Ultralytics tidak bisa menebak tanpa deteksi)")
            self.lite = None
            self.ultra = _make_ultra(spec.kind, _ultra_args(spec.kind, spec.params, self.fps, spec.buffer_sec))

    @property
    def ids_created(self) -> int:
        return self.lite._next_id - 1 if self.lite is not None else len(self._id_map)

    def uses_detection(self) -> bool:
        """Apakah frame berikutnya (yang akan diberikan ke step) memakai deteksi."""
        return self.frame_i % max(1, self.spec.det_every) == 0

    def filter(self, persons: np.ndarray) -> np.ndarray:
        p = np.asarray(persons, dtype=np.float32).reshape(-1, 5)
        p = p[p[:, 4] >= self.spec.person_min]
        if self.spec.nms_iou > 0 and len(p) > 1:
            p = p[nms(p[:, :4], p[:, 4], self.spec.nms_iou)]
        return p

    def step(self, t: float, persons: np.ndarray | None, frame: np.ndarray | None = None) -> list[TrackOut]:
        """persons: Nx5 [x1, y1, x2, y2, skor] untuk frame ini. Boleh None kalau frame ini tidak dideteksi
        (hanya untuk lite dengan det_every > 1)."""
        use_det = self.uses_detection()
        self.frame_i += 1
        if self.lite is not None:
            if not use_det or persons is None:
                return [TrackOut(tr.id, tr.box, True, tr.score) for tr in self.lite.predict(t)]
            p = self.filter(persons)
            return [TrackOut(tr.id, tr.box, False, tr.score) for tr in self.lite.update(p[:, :4], p[:, 4], t)]

        p = self.filter(persons if persons is not None else np.zeros((0, 5)))
        if len(p):
            xywh = np.stack([(p[:, 0] + p[:, 2]) / 2, (p[:, 1] + p[:, 3]) / 2, p[:, 2] - p[:, 0], p[:, 3] - p[:, 1]], 1)
            res = self.ultra.update(TrackerBoxes(xywh, p[:, 4], np.zeros(len(p))), frame)
        else:
            res = self.ultra.update(TrackerBoxes(), frame)
        out = []
        for r in res:
            rid = int(r[4])
            lid = self._id_map.setdefault(rid, len(self._id_map) + 1)
            out.append(TrackOut(lid, np.asarray(r[:4], dtype=np.float32), False, float(r[5])))
        return out


def persons_from_rows(d: np.ndarray) -> np.ndarray:
    """Baris cache (cls, skor, x1, y1, x2, y2) -> Nx5 [x1, y1, x2, y2, skor] untuk kelas orang."""
    d = np.asarray(d, dtype=np.float32).reshape(-1, 6)
    p = d[d[:, 0] == 0]
    return np.concatenate([p[:, 2:6], p[:, 1:2]], axis=1)


def run_spec(cache: dict, fps: float, spec: TrackerSpec) -> tuple[list, dict]:
    """Jalankan spec pada cache deteksi di `fps`. Output sama formatnya dengan reference_eval.run_reference."""
    step = max(1, int(round(cache["ref_fps"] / fps)))
    real_fps = cache["ref_fps"] / step
    run = TrackerRunner(spec, real_fps)
    out = []
    for i in range(0, len(cache["t"]), step):
        t = float(cache["t"][i])
        tracks = run.step(t, persons_from_rows(cache["dets"][i]) if run.uses_detection() else None)
        out.append((t, [(tr.id, tr.box) for tr in tracks]))
    return out, {"ids_created": run.ids_created, "fps": real_fps}


# ---------------- preset ----------------

LITE_DEFAULT = TrackerSpec("Lite BIoU", "lite", {})

PRESETS: dict[str, TrackerSpec] = {
    # tracker pipeline sekarang (engine.py): ByteTrack, hanya orang skor >= 0,5, tanpa NMS tambahan
    "ByteTrack (sekarang)": TrackerSpec("ByteTrack (sekarang)", "bytetrack",
                                        {"track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.5},
                                        person_min=0.5, nms_iou=0.0),
    # ByteTrack yang diberi deteksi skor rendah (tahap 2 benar-benar jalan) + NMS orang dobel
    "ByteTrack": TrackerSpec("ByteTrack", "bytetrack",
                             {"track_high_thresh": 0.5, "track_low_thresh": 0.1, "new_track_thresh": 0.55}),
    "OC-SORT": TrackerSpec("OC-SORT", "ocsort", {"track_high_thresh": 0.5, "track_low_thresh": 0.1,
                                                 "new_track_thresh": 0.55, "use_byte": True}),
    "BoT-SORT": TrackerSpec("BoT-SORT", "botsort", {"track_high_thresh": 0.5, "track_low_thresh": 0.1,
                                                    "new_track_thresh": 0.55}),
    "TrackTrack": TrackerSpec("TrackTrack", "tracktrack", {}),
    "FastTrack": TrackerSpec("FastTrack", "fasttrack", {"track_high_thresh": 0.5, "track_low_thresh": 0.1,
                                                        "new_track_thresh": 0.55}),
    "Lite BIoU": LITE_DEFAULT,
}

TUNED_PATH = Path(__file__).resolve().parents[2] / "configs" / "tracker_tuned.json"


def load_tuned(path: str | Path = TUNED_PATH) -> dict[str, TrackerSpec]:
    """Spec hasil tuning notebook 02 (kalau sudah ada). Kunci = nama spec."""
    path = Path(path)
    if not path.exists():
        return {}
    d = json.loads(path.read_text())
    return {k: TrackerSpec.from_dict(v["spec"]) for k, v in d.get("specs", {}).items()}


def save_tuned(specs: dict[str, TrackerSpec], meta: dict, best: str | None = None, info: dict | None = None,
               path: str | Path = TUNED_PATH) -> None:
    """Tulis ulang file tuning: semua spec hasil tuning, nama yang terpilih (`best`), dan info run."""
    d = {"best": best, "info": info or {},
         "specs": {k: {"spec": s.to_dict(), **meta.get(k, {})} for k, s in specs.items()}}
    Path(path).write_text(json.dumps(d, indent=2, ensure_ascii=False, default=float) + "\n")


def best_tuned(path: str | Path = TUNED_PATH) -> str | None:
    path = Path(path)
    return json.loads(path.read_text()).get("best") if path.exists() else None


def all_presets() -> dict[str, TrackerSpec]:
    return {**PRESETS, **load_tuned()}
