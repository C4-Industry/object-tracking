"""Jalankan pipeline hen-detect-ppe dengan Lite tracker di klip ground truth, tanpa mengubah kode pipeline.

Output sama persis dengan ground_truth/run_hen_pipeline.py di hen-detect-ppe (observations.jsonl berformat BE
dan run.json), jadi bisa langsung dinilai dengan evaluate.py temanmu dan notebook 05.

Cara Lite dipasang (sama dengan docs/INTEGRASI.md tahap 1 dan 3, tapi lewat "tambalan" saat runtime):
  - PPEPipeline dibuat seperti biasa (tracker bytetrack), lalu tracker Ultralytics di dalam ComplianceTracker
    diganti LiteForComplianceTracker. Output tracker tetap N x 8 [x1, y1, x2, y2, id, skor, kelas, idx],
    jadi state machine helm/vest dan alert tidak berubah.
  - Waktu sampel (detik) diberikan ke Lite, jadi Lite benar di FPS berapa pun.
  - --person-to-tracker 0.1 (tahap 3): orang dengan skor 0,1 ke atas ikut ke tracker. Default: sama dengan
    --person-conf (0,5), yaitu perilaku pipeline sekarang.

Frame yang diproses sama dengan run_hen_pipeline.py: sampel k = frame floor((k + 0.5) * 15 / fps) (sampler ffmpeg,
sama dengan BE). Hasil deteksi disimpan di --det-cache, jadi menjalankan tracker lain di klip yang sama tidak
mengulang detektor.

Contoh (dari folder object-tracking):
  python scripts/run_pipeline_lite.py --run hen_20261112_lite --tracker lite \\
      --hen-repo ../hen-detect-ppe --gt-root "~/PPE - PT SRP/ground truth"
  python scripts/run_pipeline_lite.py --run hen_20261112_lite-skorrendah --tracker lite --person-to-tracker 0.1 ...
  python scripts/run_pipeline_lite.py --run hen_20261112_bytetrack --tracker bytetrack ...   # pembanding
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from object_tracking.integration import LiteForComplianceTracker, load_lite_config  # noqa: E402


def sha16(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sample_frame(k: int, sample_fps: float, fps: float, sampler: str) -> int:
    """Sama dengan evaluate.sample_frame di hen-detect-ppe."""
    if sampler == "ffmpeg":
        return int(math.floor(round((k + 0.5) * fps / sample_fps, 6)))
    return int(math.floor(round(k / sample_fps * fps, 6) + 0.5))


def evidence(visible: bool, seen: bool) -> str:
    if not visible:
        return "not_visible"
    return "seen" if seen else "missing"


def xyxy(b) -> list[float] | None:
    return None if b is None else [float(b.x1), float(b.y1), float(b.x2), float(b.y2)]


class TimedLite(LiteForComplianceTracker):
    """LiteForComplianceTracker yang mengambil waktu dari atribut `now` (diisi sebelum process_frame), karena
    ComplianceTracker memanggil tracker.update(boxes) tanpa waktu. Indeks deteksi terakhir disimpan di `last_idx`."""

    now: float = 0.0

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.last_idx: dict[int, int] = {}

    def update(self, dets, img=None, timestamp_sec=None):
        out = super().update(dets, img, timestamp_sec=self.now if timestamp_sec is None else timestamp_sec)
        self.last_idx = {int(r[4]): int(r[7]) for r in out}
        return out


class CachedDetector:
    """Bungkus detektor pipeline: deteksi (semua kelas, skor >= 0,05) disimpan per frame video."""

    FLOOR = 0.05

    def __init__(self, inner, cache_file: Path | None, BBox):
        self.inner, self.cache_file, self.BBox = inner, cache_file, BBox
        self.frame = -1
        self.data: dict[int, np.ndarray] = {}
        if cache_file is not None and cache_file.exists():
            z = np.load(cache_file)
            self.data = {int(k[1:]): z[k] for k in z.files}
        self.dirty = False

    def detect(self, frame, imgsz, conf_thresh):
        rows = self.data.get(self.frame)
        if rows is None:
            p, h, v = self.inner.detect(frame, imgsz=imgsz, conf_thresh=min(conf_thresh, self.FLOOR))
            rows = np.array([[b.x1, b.y1, b.x2, b.y2, b.conf, c] for c, bs in enumerate((p, h, v)) for b in bs],
                            np.float32).reshape(-1, 6)
            self.data[self.frame] = rows
            self.dirty = True
        out = ([], [], [])
        for x1, y1, x2, y2, conf, c in rows:
            if conf >= conf_thresh:
                out[int(c)].append(self.BBox(x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                                             conf=float(conf), cls_id=int(c)))
        return out

    def save(self):
        if self.cache_file is not None and self.dirty:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.cache_file, **{f"f{k}": v for k, v in self.data.items()})
            self.dirty = False


def resolve_video(meta: dict, hen: Path, gt_root: Path) -> Path:
    for p in (hen / meta["video"], gt_root / Path(meta["video"]).name):
        if p.exists():
            return p
    raise FileNotFoundError(f"video {meta['video']} tidak ada di {hen} maupun {gt_root}")


def lite_config(name: str) -> tuple:
    """bawaan = Lite tanpa tuning, tuned = LITE_TUNED_DETECTOR, atau nama preset lain di LITE_PRESETS."""
    return load_lite_config(detector=None if name == "tuned" else name)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="nama run, format <pipeline>_<YYYYMMDD>_<perubahan>")
    ap.add_argument("--hen-repo", default=str(ROOT.parent / "hen-detect-ppe"))
    ap.add_argument("--gt-root", default=None, help="folder ground truth (berisi frozen/, runs/, video). "
                                                     "Default: <hen-repo>/dataset/ground_truth")
    ap.add_argument("--clips", nargs="*", help="default: semua klip di frozen/ kecuali versi _blur dan _reenc")
    ap.add_argument("--detector", default="models/dfine_n_stg2/model.onnx", help="relatif terhadap --hen-repo")
    ap.add_argument("--tracker", default="lite", help="lite, atau tracker bawaan pipeline (bytetrack, ocsort, ...)")
    ap.add_argument("--lite-config", default="bawaan", help="bawaan | tuned | nama preset detektor di LITE_PRESETS")
    ap.add_argument("--person-conf", type=float, default=0.50)
    ap.add_argument("--person-to-tracker", type=float, default=None,
                    help="skor orang minimal yang ikut ke tracker (default = --person-conf)")
    ap.add_argument("--helmet-conf", type=float, default=0.30)
    ap.add_argument("--vest-conf", type=float, default=0.25)
    ap.add_argument("--track-lost-s", type=float, default=3.0, help="bytetrack saja (Lite pakai max_lost_sec)")
    ap.add_argument("--fps", type=float, default=3.0)
    ap.add_argument("--sampler", default="ffmpeg", choices=["ffmpeg", "exact"])
    ap.add_argument("--det-cache", default=str(ROOT / "outputs" / "cache" / "pipeline_det"))
    ap.add_argument("--sweep", default=None)
    ap.add_argument("--note", default="")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    hen = Path(args.hen_repo).expanduser().resolve()
    gt_root = Path(args.gt_root).expanduser() if args.gt_root else hen / "dataset" / "ground_truth"
    sys.path.insert(0, str(hen / "src"))
    from detect.pipeline.pipeline import PPEPipeline  # noqa: E402
    from detect.pipeline.types import BBox  # noqa: E402

    clips = args.clips or sorted(p.name for p in (gt_root / "frozen").iterdir()
                                 if (p / "meta.json").exists() and not p.name.endswith(("_blur", "_reenc")))
    run_dir = gt_root / "runs" / args.run
    if (run_dir / "run.json").exists() and not args.overwrite:
        raise SystemExit(f"run '{args.run}' sudah ada ({run_dir}). Pakai nama lain atau --overwrite.")
    run_dir.mkdir(parents=True, exist_ok=True)
    det_path = hen / args.detector
    det_sha = sha16(det_path)
    to_tracker = args.person_conf if args.person_to_tracker is None else args.person_to_tracker
    use_lite = args.tracker == "lite"
    if use_lite:
        cfg, inp = lite_config(args.lite_config)

    timing = {}
    for clip in clips:
        meta = json.loads((gt_root / "frozen" / clip / "meta.json").read_text())
        targets, k = {}, 0
        while (f := sample_frame(k, args.fps, meta["fps"], args.sampler)) < meta["n_frames"]:
            targets[f] = k / args.fps
            k += 1
        pipe = PPEPipeline(
            detector_path=str(det_path), pose_backend="heuristic", tracker_type="bytetrack" if use_lite else args.tracker,
            filter_by_pose=False, person_conf_thresh=to_tracker, helmet_conf_thresh=args.helmet_conf,
            vest_conf_thresh=args.vest_conf, track_buffer=max(1, round(args.track_lost_s * args.fps)),
            fps=args.fps, device="cpu",
        )
        if use_lite:
            lite = TimedLite(fps=args.fps, config=cfg, person_min=min(inp["person_min"], to_tracker),
                             nms_iou=inp["nms_iou"])
            pipe.tracker.tracker = lite
            pipe.tracker.byte_tracker = lite
            idx_of = lite.last_idx
        else:
            idx_of = {}
            inner = pipe.tracker.tracker
            original = inner.update

            def update(*a, _orig=original, **kw):
                out = _orig(*a, **kw)
                idx_of.clear()
                idx_of.update({int(r[4]): int(r[7]) for r in out})
                return out

            inner.update = update
        cache = CachedDetector(pipe.detector, Path(args.det_cache) / f"{clip}_{det_sha}.npz", BBox)
        pipe.detector = cache

        import cv2
        cap = cv2.VideoCapture(str(resolve_video(meta, hen, gt_root)))
        out = run_dir / clip
        out.mkdir(parents=True, exist_ok=True)
        n_obs, frame, infer_s, track_s = 0, 0, 0.0, 0.0
        last = max(targets)
        with (out / "observations.jsonl").open("w") as fh:
            while frame <= last:
                if frame not in targets:
                    if not cap.grab():      # frame yang tidak diproses: cukup dilewati
                        break
                    frame += 1
                    continue
                ok, img = cap.read()
                if not ok:
                    break
                if frame in targets:
                    t_s = targets[frame]
                    cache.frame = frame
                    if use_lite:
                        lite.now = t_s
                    t0 = time.perf_counter()
                    res = pipe.process_frame(img, timestamp_sec=t_s)
                    infer_s += time.perf_counter() - t0
                    track_s += res.latency_ms.get("tracking_ms", 0.0) / 1000
                    for tp in res.tracked_persons:
                        i = idx_of.get(int(tp.track_id), -1)
                        a = res.associated_persons[i] if 0 <= i < len(res.associated_persons) else None
                        fh.write(json.dumps({
                            "camera_id": clip, "mono_ts": t_s, "track_id": int(tp.track_id),
                            "bbox": [float(tp.box.x1), float(tp.box.y1), float(tp.box.x2), float(tp.box.y2)],
                            "person_conf": float(tp.box.conf),
                            "helmet": evidence(tp.head_visible, tp.helmet_seen),
                            "vest": evidence(tp.torso_visible, tp.vest_seen),
                            "helmet_conf": a.helmet_box.conf if a and a.helmet_box else None,
                            "vest_conf": a.vest_box.conf if a and a.vest_box else None,
                            "helmet_bbox": xyxy(a.helmet_box) if a else None,
                            "vest_bbox": xyxy(a.vest_box) if a else None,
                            "detection_bbox": xyxy(a.person_box) if a else None,
                        }) + "\n")
                        n_obs += 1
                frame += 1
        cap.release()
        cache.save()
        n = len(targets)
        timing[clip] = {"frames": n, "pipeline_s": round(infer_s, 2), "ms_per_frame": round(1000 * infer_s / max(n, 1), 1),
                        "tracking_ms_per_frame": round(1000 * track_s / max(n, 1), 2)}
        print(f"{clip}: {n_obs} observasi, {n} frame di {args.fps:g} fps, "
              f"tracker {timing[clip]['tracking_ms_per_frame']} ms/frame", flush=True)

    def git(cwd, *a):
        r = subprocess.run(["git", *a], cwd=cwd, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""

    hen_commit = git(hen, "rev-parse", "--short", "HEAD") or "?"
    ot_commit = git(ROOT, "rev-parse", "--short", "HEAD") or "?"
    tracker_cfg = None
    if use_lite:
        from dataclasses import asdict
        tracker_cfg = {"preset": args.lite_config, **asdict(cfg), "person_min": min(inp["person_min"], to_tracker),
                       "nms_iou": inp["nms_iou"]}
    (run_dir / "run.json").write_text(json.dumps({
        "run": args.run, "pipeline": "hen-detect-ppe" + (" + Lite (object-tracking)" if use_lite else ""),
        "git_commit": hen_commit, "object_tracking_commit": ot_commit if use_lite else None,
        "note": args.note, "weights": {"path": args.detector, "sha256": det_sha, "bytes": det_path.stat().st_size},
        "created": datetime.datetime.now().isoformat(timespec="seconds"), "clips": clips, "sweep": args.sweep,
        "timing": timing, "machine": {"host": "local", "device": "cpu", "provider": "onnxruntime CPU"},
        "config": {"detector": args.detector, "pose_backend": "heuristic", "filter_by_pose": False,
                   "tracker": args.tracker, "tracker_config": tracker_cfg, "person_to_tracker": to_tracker,
                   "track_lost_s": args.track_lost_s, "nms": tracker_cfg["nms_iou"] if tracker_cfg else None,
                   "thresholds": {"person": args.person_conf, "helmet": args.helmet_conf, "vest": args.vest_conf},
                   "sample_fps": args.fps, "sampler": args.sampler, "evidence": "raw per frame"},
    }, indent=1) + "\n")
    print("run.json ->", run_dir / "run.json")


if __name__ == "__main__":
    main()
