"""Siapkan klip ground truth baru: potong video jadi 15 fps dan buat pre-label (box + ID) untuk CVAT.

Untuk setiap kandidat (video, detik mulai, detik selesai):
  1. <out>/<momen>_<SSSS>-<SSSS>s_15fps.mp4 (+ .meta.txt), format dan nama sama dengan klip GT yang sudah ada.
  2. <out>/prelabel/<klip>/observations.jsonl: deteksi D-FINE + Lite tracker di 3 fps, format sama dengan
     replay BE, jadi bisa langsung diubah ke CVAT dengan obs_to_cvat.py di hen-detect-ppe:
       python ground_truth/obs_to_cvat.py <observations.jsonl> <klip>.xml --width W --height H

Pre-label memakai frame 5k (sampel k pada 3 fps = detik k/3 tepat), sesuai pemetaan obs_to_cvat.py
(mono_ts k/3 -> frame 5k). Pre-label lama dari replay BE (ffmpeg) jatuh di frame 5k + 2, sehingga box CVAT
terlambat 2 frame dan klip harus di-freeze dengan --shift-frames 2. Dengan pre-label ini pergeseran itu tidak ada.

Lite memakai orang dengan skor 0,1 ke atas, jadi orang yang tertutup sebagian tetap tersambung dan ID lebih
jarang terputus. Pengecekan dan koreksi di CVAT tetap wajib (pre-label bukan ground truth).

Contoh:
  python scripts/siapkan_klip_gt.py --videos "~/Dataset_Momen_PPE_Anonim" --out "~/PPE - PT SRP/ground truth/kandidat_baru" \\
      --clip v16_momen_010.mp4:255:315 --clip v10_momen_001.mp4:83:143
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from object_tracking.detector import OnnxDFineDetector  # noqa: E402
from object_tracking.integration import PersonTracker  # noqa: E402
from object_tracking.lite import TrackerConfig  # noqa: E402

GT_FPS, PRE_FPS = 15, 3


def cut(src: Path, start: int, end: int, out: Path, preset: str) -> None:
    tmp = out.with_suffix(".tmp.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-i", str(src), "-t", str(end - start),
                    "-vf", f"fps={GT_FPS}", "-an", "-c:v", "libx264", "-preset", preset, "-crf", "18",
                    "-pix_fmt", "yuv420p", str(tmp)], check=True)
    tmp.rename(out)


def frames(path: Path, every: int):
    """Frame ke-0, every, 2*every, ... dari video 15 fps (dibaca berurutan, tanpa seek)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    f = 0
    while True:
        if f % every:
            if not cap.grab():
                break
        else:
            ok, img = cap.read()
            if not ok:
                break
            yield f, img
        f += 1
    cap.release()


def prelabel(video: Path, out_jsonl: Path, det: OnnxDFineDetector, clip: str) -> dict:
    trk = PersonTracker(TrackerConfig(), person_min=0.10, nms_iou=0.6)
    every = GT_FPS // PRE_FPS
    n_obs, n = 0, 0
    tmp = out_jsonl.with_suffix(".tmp")
    with tmp.open("w") as fh:
        for f, img in frames(video, every):
            k = f // every
            persons, _, _ = det.detect(img, conf_thresh=0.10)
            b = np.array([[p.x1, p.y1, p.x2, p.y2] for p in persons], np.float32).reshape(-1, 4)
            s = np.array([p.conf for p in persons], np.float32)
            for o in trk.update(b, s, k / PRE_FPS):
                fh.write(json.dumps({"camera_id": clip, "mono_ts": k / PRE_FPS, "track_id": int(o.id),
                                     "bbox": [round(float(v), 1) for v in o.box], "person_conf": round(o.score, 3),
                                     "helmet": "unset", "vest": "unset"}) + "\n")
                n_obs += 1
            n += 1
    tmp.rename(out_jsonl)
    return {"frames": n, "observations": n_obs, "ids": trk.ids_shown}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip", action="append", required=True, help="video.mp4:detik_mulai:detik_selesai")
    ap.add_argument("--model", default=str(ROOT / "models" / "dfine-n-stg2-960.onnx"))
    ap.add_argument("--preset", default="veryfast", help="preset x264 (veryfast cukup untuk labeling)")
    args = ap.parse_args()

    vdir, out = Path(args.videos).expanduser(), Path(args.out).expanduser()
    (out / "prelabel").mkdir(parents=True, exist_ok=True)
    det = None
    for spec in args.clip:
        name, s, e = spec.split(":")
        s, e = int(s), int(e)
        clip = f"{Path(name).stem}_{s:04d}-{e:04d}s"
        mp4 = out / f"{clip}_{GT_FPS}fps.mp4"
        if not mp4.exists():
            t0 = time.time()
            cut(vdir / name, s, e, mp4, args.preset)
            (out / f"{clip}_{GT_FPS}fps.meta.txt").write_text(
                f"source={vdir / name}\nstart_s={s}\nend_s={e}\nfps={GT_FPS}\n"
                f"eval_fps={PRE_FPS}   # GT frame 5k <-> sample k (pre-label: sampel k = frame 5k tepat)\n"
                f"created={datetime.datetime.now(datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n")
            print(f"{clip}: video dipotong ({time.time() - t0:.0f} s)", flush=True)
        obs = out / "prelabel" / clip / "observations.jsonl"
        if not obs.exists():
            obs.parent.mkdir(parents=True, exist_ok=True)
            det = det or OnnxDFineDetector(args.model)
            t0 = time.time()
            info = prelabel(mp4, obs, det, clip)
            print(f"{clip}: pre-label {info} ({time.time() - t0:.0f} s)", flush=True)
    print("selesai")


if __name__ == "__main__":
    main()
