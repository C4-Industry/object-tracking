"""Video perbandingan tracker di satu sekuens MOT17: video asli + label manusia + satu panel per tracker.

Semua tracker diberi deteksi publik MOT17 yang sama (seperti di benchmark), jadi beda antar panel murni dari
tracker. Karena MOT17 punya label ID manusia, setiap panel tracker juga menandai ID SWITCH tepat di frame
kejadiannya: orang yang sama (menurut label) tiba-tiba memegang ID tracker yang berbeda. Box-nya diberi tepi
MERAH tebal dan label "SWITCH". Penghitungnya memakai aturan CLEAR MOT (IoU >= 0,5, pasangan lama
diutamakan), jadi total di akhir video sama dengan kolom IDSW di tabel benchmark.

Video mentah diambil dari https://motchallenge.net/sequenceVideos/<seq>-raw.webm (sekitar 5 MB per sekuens,
setengah resolusi). Label dan deteksi tetap dari MOT17Labels.zip. Lisensi MOT17: CC BY-NC-SA 3.0, hanya untuk
evaluasi internal.
"""

from __future__ import annotations

import math
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

from . import mot17
from . import metrics as mm
from .live_compare import _text, id_color, open_writer
from .zoo import TrackerSpec

VIDEO_URL = "https://motchallenge.net/sequenceVideos/{name}-raw.webm"
HEADER_H = 44


def download_video(root: str | Path, name: str) -> Path:
    """Unduh video mentah satu sekuens (mis. MOT17-04) ke root/videos/. Sekali saja."""
    path = Path(root) / "videos" / f"{name}-raw.webm"
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    url = VIDEO_URL.format(name=name)
    print(f"mengunduh {url} ...")
    try:
        urllib.request.urlretrieve(url, path)
    except Exception as e:
        if path.exists():
            path.unlink()
        raise RuntimeError(f"Gagal mengunduh ({e}). Unduh manual {url} lalu taruh di {path}.") from e
    return path


def video_info(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    info = {"width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS), "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}
    cap.release()
    return info


# ---------------- ID switch per frame (CLEAR MOT) ----------------

def idsw_events(gts: list, hyps: list) -> list[set]:
    """Untuk tiap frame: himpunan ID tracker yang baru saja mengalami ID switch.
    Sama persis dengan `mot_metrics.evaluate_sequence` (logika TrackEval): pasangan (gt, tracker) dari timestep
    sebelumnya diutamakan kalau IoU >= 0,5, lalu ID switch terjadi kalau orang (gt) sekarang dipasangkan ke ID
    tracker yang berbeda dari pasangan terakhirnya. Jadi jumlahnya = kolom IDSW."""
    prev: dict[int, int] = {}        # gt id -> hyp id terakhir yang pernah dipasangkan
    prev_step: dict[int, int] = {}   # gt id -> hyp id di timestep sebelumnya
    out = []
    for (g_ids, g_box), (h_ids, h_box) in zip(gts, hyps):
        ev = set()
        if len(g_ids) and len(h_ids):
            s = mm.box_iou_xywh(g_box, h_box)
            cont = np.array([[prev_step.get(int(g)) == int(h) for h in h_ids] for g in g_ids])
            score = s + 1000 * cont
            score[s < 0.5 - mm.EPS] = 0
            r, c = linear_sum_assignment(-score)
            ok = score[r, c] > mm.EPS
            prev_step = {}
            for gi, hi in zip(r[ok], c[ok]):
                g, h = int(g_ids[gi]), int(h_ids[hi])
                if g in prev and prev[g] != h:
                    ev.add(h)
                prev[g] = h
                prev_step[g] = h
        out.append(ev)
    return out


# ---------------- gambar ----------------

def _header(img, title, sub, color=(25, 25, 25)):
    bar = np.full((HEADER_H, img.shape[1], 3), color, np.uint8)
    cv2.putText(bar, title, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(bar, sub, (8, 37), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1, cv2.LINE_AA)
    return np.vstack([bar, img])


def _box(img, b, sx, sy, col, t=1):
    x1, y1, x2, y2 = int(b[0] * sx), int(b[1] * sy), int(b[2] * sx), int(b[3] * sy)
    cv2.rectangle(img, (x1, y1), (x2, y2), col, t)
    return x1, y1, x2, y2


def _info_panel(w, h, lines):
    img = np.full((h, w, 3), 18, np.uint8)
    y = 30
    for s, scale, col in lines:
        cv2.putText(img, s, (14, y), cv2.FONT_HERSHEY_SIMPLEX, scale, col, 1, cv2.LINE_AA)
        y += int(34 * scale) + 8
    return img


def render_comparison(seq: mot17.Seq, video_path: str | Path, specs: dict[str, TrackerSpec], out_path: str | Path,
                      target_fps: float | None = 5, panel_width: int = 640, show_gt: bool = True,
                      max_sec: float | None = None, trail_sec: float = 2.0, progress=None) -> dict:
    """Buat video grid: [video asli] [label manusia] [tracker 1] [tracker 2] ...

    target_fps: FPS proses tracker (None = FPS asli sekuens). Video hasil diputar di FPS ini, jadi durasinya
    sama dengan aslinya dan terlihat seperti CCTV yang diproses di FPS tersebut.
    Hasil: dict berisi path video, path gambar cuplikan, dan tabel angka per tracker."""
    vinfo = video_info(video_path)
    if abs(vinfo["frames"] - seq.length) > 1:
        raise RuntimeError(f"jumlah frame video ({vinfo['frames']}) beda dengan label ({seq.length})")
    step = mot17.frame_step(seq, target_fps)
    proc_fps = seq.fps / step
    last = seq.length if not max_sec else min(seq.length, int(max_sec * seq.fps))

    # 1. jalankan semua tracker sekali (deteksi publik yang sama), nilai, dan cari kejadian ID switch
    runs = {}
    names = list(specs)
    for name in (progress(names) if progress else names):
        hyp, info = mot17.run_tracker(specs[name], seq, target_fps)
        hyp = {f: v for f, v in hyp.items() if f <= last}
        gts, hyps = mot17.to_eval_frames(seq, hyp)
        m = mm.summarize(mm.evaluate_sequence(gts, hyps))
        frames = sorted(hyp)
        ev = idsw_events(gts, hyps)
        kept = {f: set(h[0].tolist()) for f, h in zip(frames, hyps)}
        runs[name] = {"hyp": hyp, "metrics": m, "events": dict(zip(frames, ev)), "kept": kept, "info": info}

    # 2. gambar
    sx = vinfo["width"] / seq.width
    sy = vinfo["height"] / seq.height
    pw = int(panel_width)
    ph = int(round(vinfo["height"] * pw / vinfo["width"]))
    k = pw / vinfo["width"]
    n_panels = 1 + int(show_gt) + len(specs)
    cols = math.ceil(math.sqrt(n_panels))
    rows = math.ceil(n_panels / cols)
    size = (cols * pw, rows * (ph + HEADER_H))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = open_writer(out_path, proc_fps, size)

    gt = seq.gt[(seq.gt[:, 6] != 0) & (seq.gt[:, 7] == 1)]
    gt_by_f = {int(f): gt[gt[:, 0] == f] for f in np.unique(gt[:, 0])}
    state = {n: {"trails": {}, "sw_total": 0, "sw_recent": {}, "ids": set()} for n in specs}
    fps_txt = "asli" if not target_fps else f"{proc_fps:.1f}"
    snap_frame = None
    snap_at = (1 + last) // 2

    cap = cv2.VideoCapture(str(video_path))
    for f in range(1, last + 1):
        ok, frame = cap.read()
        if not ok:
            break
        if (f - 1) % step:
            continue
        t = (f - 1) / seq.fps
        small = cv2.resize(frame, (pw, ph), interpolation=cv2.INTER_AREA)
        panels = [_header(small.copy(), f"Video asli  {seq.name}",
                          f"t = {t:5.1f} s  |  {seq.fps:.0f} FPS asli, diproses {fps_txt} FPS")]
        if show_gt:
            img = small.copy()
            g = gt_by_f.get(f, np.zeros((0, 9)))
            for r in g:
                b = (r[2], r[3], r[2] + r[4], r[3] + r[5])
                x1, y1, _, _ = _box(img, np.array(b) * [sx * k, sy * k, sx * k, sy * k], 1, 1, id_color(int(r[1])))
            panels.append(_header(img, "Label manusia (ground truth)", f"{len(g)} orang di frame ini"))

        for name in specs:
            R, S = runs[name], state[name]
            ids, boxes = R["hyp"].get(f, (np.zeros(0, int), np.zeros((0, 4))))
            ev = R["events"].get(f, set())
            kept = R["kept"].get(f, set())
            S["sw_total"] += len(ev)
            for i in ev:
                S["sw_recent"][i] = t
            S["ids"].update(int(i) for i in ids)
            img = small.copy()
            # jejak gerak (titik kaki)
            for i, b in zip(ids, boxes):
                S["trails"].setdefault(int(i), []).append((t, (b[0] + b[2]) / 2 * sx * k, b[3] * sy * k))
            for i, pts in list(S["trails"].items()):
                pts[:] = [p for p in pts if t - p[0] <= trail_sec]
                if not pts:
                    del S["trails"][i]
                elif len(pts) > 1:
                    cv2.polylines(img, [np.array([(int(x), int(y)) for _, x, y in pts], np.int32)], False,
                                  id_color(i), 1, cv2.LINE_AA)
            for i, b in zip(ids, boxes):
                i = int(i)
                bb = np.array(b) * [sx * k, sy * k, sx * k, sy * k]
                if i not in kept:          # cocok dengan orang yang tidak dinilai (distraktor, orang diam di kendaraan)
                    _box(img, bb, 1, 1, (120, 120, 120))
                    continue
                switched = i in S["sw_recent"] and t - S["sw_recent"][i] <= 1.0
                col = id_color(i)
                x1, y1, x2, y2 = _box(img, bb, 1, 1, (0, 0, 255) if switched else col, 3 if switched else 1)
                if switched:
                    _text(img, f"{i} SWITCH", x1, y1, 0.42, (255, 255, 255), (0, 0, 255))
                else:
                    _text(img, str(i), x1, y1, 0.38, (0, 0, 0), col)
            m = R["metrics"]
            panels.append(_header(img, f"{name}",
                                  f"IDF1 {m['IDF1']:.3f}  HOTA {m['HOTA']:.3f}  |  ID switch {S['sw_total']}"
                                  f"/{int(m['IDSW'])}  |  ID dibuat {len(S['ids'])}"))

        blank = np.full((ph + HEADER_H, pw, 3), 18, np.uint8)
        if len(panels) < rows * cols:
            blank = _info_panel(pw, ph + HEADER_H, [
                ("Cara membaca", 0.6, (255, 255, 255)),
                ("warna + nomor = ID tracker", 0.5, (210, 210, 210)),
                ("garis = jejak 2 detik terakhir", 0.5, (210, 210, 210)),
                ("tepi MERAH 'SWITCH' = ID switch", 0.5, (80, 80, 255)),
                ("  (orang sama, ID tracker berubah)", 0.5, (210, 210, 210)),
                ("abu-abu = box di orang yang tidak dinilai", 0.5, (160, 160, 160)),
                ("ID switch a/b = sejauh ini / total", 0.5, (210, 210, 210)),
                (f"deteksi publik {seq.detector}, sama untuk semua", 0.5, (210, 210, 210)),
            ])
        panels += [blank] * (rows * cols - len(panels))
        grid = np.vstack([np.hstack(panels[r * cols:(r + 1) * cols]) for r in range(rows)])
        writer.write(grid)
        if snap_frame is None and f >= snap_at:
            snap_frame = grid
    cap.release()
    writer.release()

    snap_path = out_path.with_suffix(".jpg")
    if snap_frame is not None:
        cv2.imwrite(str(snap_path), snap_frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    table = [{"tracker": n, "IDF1": runs[n]["metrics"]["IDF1"], "HOTA": runs[n]["metrics"]["HOTA"],
              "MOTA": runs[n]["metrics"]["MOTA"], "IDSW": int(runs[n]["metrics"]["IDSW"]),
              "IDSW di video": state[n]["sw_total"], "ID dibuat": len(state[n]["ids"]),
              "GT_IDs": int(runs[n]["metrics"]["GT_IDs"])} for n in specs]
    return {"video": out_path, "snapshot": snap_path, "table": table, "fps": proc_fps, "size": size}
