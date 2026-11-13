"""Klip CCTV palsu dengan label ground truth format CVAT (for video 1.1 XML dan MOT 1.1), untuk tes dan uji notebook 05.
Video: orang = kotak merah di latar abu-abu, sehingga detektor palsu (FakeDetector) bisa menemukannya."""

import types
from pathlib import Path

import cv2
import numpy as np

from object_tracking import synthetic as syn


class FakeDetector:
    """Detektor palsu: kotak merah di frame = orang. Skor acak 0,3 sampai 0,95."""

    active_providers = ["fake"]

    def detect_timed(self, frame, conf_thresh=0.05):
        m = (frame[:, :, 2] > 200) & (frame[:, :, 1] < 80)
        n, _, st, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8))
        rng = np.random.default_rng(int(st[1:, 0].sum()) if n > 1 else 0)
        persons = [types.SimpleNamespace(x1=float(x), y1=float(y), x2=float(x + w), y2=float(y + h),
                                         conf=float(rng.uniform(0.3, 0.95)), cls_id=0)
                   for x, y, w, h, a in st[1:] if a > 150]
        return persons, [], [], {"det_pre_ms": 0.0, "det_infer_ms": 0.0, "det_post_ms": 0.0}


def make_cvat_clip(folder, name="kamera01_klip01", duration=12.0, fps=25, size=(1280, 720), seed=0,
                   keyframe_every=5, fmt="xml"):
    """Tulis video + label. fmt: "xml" (CVAT for video 1.1) atau "mot" (gt/gt.txt)."""
    d = Path(folder) / name
    d.mkdir(parents=True, exist_ok=True)
    people = syn.default_people()
    n = int(duration * fps)
    w = cv2.VideoWriter(str(d / f"{name}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    for f in range(n):
        t = f / fps
        img = np.full((size[1], size[0], 3), 90, np.uint8)
        for p in people:
            if p.state(t) in ("visible", "partial"):
                x1, y1, x2, y2 = p.box(t).astype(int)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), -1)
        w.write(img)
    w.release()
    if fmt == "mot":
        (d / "gt").mkdir(exist_ok=True)
        rows = []
        for f in range(n):
            t = f / fps
            for p in people:
                st = p.state(t)
                if st == "absent":
                    continue
                x1, y1, x2, y2 = p.box(t)
                rows.append(f"{f + 1},{p.pid},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f},1,1,{0.2 if st == 'hidden' else 1.0}")
        (d / "gt" / "gt.txt").write_text("\n".join(rows) + "\n")
        return d
    tracks = []
    for p in people:
        boxes = []
        f_in, f_out = int(np.ceil(p.t_in * fps)), min(n - 1, int(p.t_out * fps))
        for f in range(f_in, f_out + 1):
            t = f / fps
            key = (f - f_in) % keyframe_every == 0 or f == f_out
            if not key:
                continue                              # frame di antara keyframe: diisi interpolasi oleh loader
            x1, y1, x2, y2 = p.box(t)
            occ = int(p.state(t) == "hidden")
            boxes.append(f'    <box frame="{f}" keyframe="1" outside="0" occluded="{occ}" xtl="{x1:.2f}" ytl="{y1:.2f}" '
                         f'xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"></box>')
        if f_out < n - 1:
            boxes.append(f'    <box frame="{f_out + 1}" keyframe="1" outside="1" occluded="0" xtl="{x1:.2f}" '
                         f'ytl="{y1:.2f}" xbr="{x2:.2f}" ybr="{y2:.2f}" z_order="0"></box>')
        tracks.append(f'  <track id="{p.pid - 1}" label="person" source="manual">\n' + "\n".join(boxes) + "\n  </track>")
    tracks.append('  <track id="99" label="forklift" source="manual">\n'
                  '    <box frame="0" keyframe="1" outside="0" occluded="0" xtl="10" ytl="10" xbr="200" ybr="150" z_order="0"></box>\n  </track>')
    xml = f"""<?xml version="1.0" encoding="utf-8"?>
<annotations>
  <version>1.1</version>
  <meta>
    <task>
      <id>1</id><name>{name}</name><size>{n}</size><mode>interpolation</mode>
      <start_frame>0</start_frame><stop_frame>{n - 1}</stop_frame><frame_filter></frame_filter>
      <labels><label><name>person</name></label><label><name>forklift</name></label></labels>
      <original_size><width>{size[0]}</width><height>{size[1]}</height></original_size>
    </task>
  </meta>
{chr(10).join(tracks)}
</annotations>
"""
    (d / "annotations.xml").write_text(xml)
    return d


def make_frozen_clip(root, name="v99_klip_0000-0012s", duration=12.0, fps=15, size=(1280, 720), every=1,
                     ignore=None, run_shift=2):
    """Klip format 'frozen' PT SRP (frozen/<klip>/gt.jsonl + meta.json, video <klip>_15fps.mp4) dan satu output
    pipeline palsu di runs/palsu/<klip>/observations.jsonl yang diambil 3 FPS dengan pergeseran run_shift frame."""
    import json

    root = Path(root)
    d = root / "frozen" / name
    d.mkdir(parents=True, exist_ok=True)
    people = syn.default_people()
    n = int(duration * fps)
    w = cv2.VideoWriter(str(root / f"{name}_15fps.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    rows = []
    for f in range(n):
        t = f / fps
        img = np.full((size[1], size[0], 3), 90, np.uint8)
        for p in people:
            st = p.state(t)
            if st == "absent":
                continue
            b = p.box(t)
            if st in ("visible", "partial"):
                cv2.rectangle(img, tuple(b[:2].astype(int)), tuple(b[2:].astype(int)), (0, 0, 255), -1)
            if f % every == 0:
                rows.append({"frame": f, "t_s": round(t, 4), "track_id": 100 + p.pid, "bbox": [round(float(v), 1) for v in b],
                             "helmet": "worn", "vest": "worn", "occluded": st != "visible", "small": False})
        w.write(img)
    w.release()
    (d / "gt.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    meta = {"clip": name, "video": f"dataset/ground_truth/{name}_15fps.mp4", "width": size[0], "height": size[1],
            "fps": fps, "n_frames": n, "eval_step": 5, "eval_fps": 3.0, "gt_step": 1 if every == 1 else None,
            "shift_frames": 0, "min_h_px": 60, "ignore_regions": ignore or [], "n_tracks": len(people)}
    (d / "meta.json").write_text(json.dumps(meta))
    rd = root / "runs" / "palsu" / name
    rd.mkdir(parents=True, exist_ok=True)
    obs = []
    for k in range(int(duration * 3)):
        f = 5 * k + run_shift                             # sampel ffmpeg jatuh 2 frame setelah 5k
        t = f / fps
        for p in people:
            if p.state(t) in ("visible", "partial"):
                obs.append({"camera_id": name, "mono_ts": k / 3, "track_id": p.pid, "bbox": [float(v) for v in p.box(t)]})
    (rd / "observations.jsonl").write_text("\n".join(json.dumps(o) for o in obs) + "\n")
    (root / "runs" / "palsu" / "run.json").write_text(json.dumps(
        {"run": "palsu", "pipeline": "tes", "clips": [name], "weights": {"sha256": "abc"},
         "config": {"tracker": "bytetrack", "sample_fps": 3.0, "thresholds": {"person": 0.5}}}))
    return root
