"""Ground truth dari CVAT: membaca label ID orang, mengecek label, dan menyiapkan evaluasi tracker.

Format yang didukung:
- **frozen** (`gt.jsonl` + `meta.json`): label PT SRP yang sudah dibekukan dari CVAT. Satu baris per orang per
  frame (`frame`, `track_id`, `bbox` xyxy, `occluded`, `small`, `helmet`, `vest`), plus `ignore_regions`
  (area yang tidak dinilai) dan info video di meta.
- **CVAT for video 1.1** (`annotations.xml`): setiap orang adalah satu `<track>` dengan ID tetap,
  box per frame, tanda `outside` (orang keluar frame) dan `occluded` (tertutup).
- **MOT 1.1** (`gt/gt.txt`): `frame, id, x, y, w, h, not_ignored, class, visibility`, frame mulai dari 1.

Satu klip = satu folder berisi label. Videonya ada di folder yang sama, atau dicari di folder video lain
(berdasarkan nama di meta.json atau nama klip).

Output pipeline lain (`runs/<run>/<klip>/observations.jsonl`) juga bisa dinilai dengan label yang sama.

Hasilnya diubah menjadi `mot17.Seq`, jadi semua alat MOT17 (run_benchmark, summary, video perbandingan) bisa
dipakai langsung. Orang yang tertutup bisa dinilai biasa atau diabaikan (tidak dihitung benar maupun salah),
dengan aturan yang sama seperti distraktor MOT17.
"""

from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import mot17
from .video_io import VIDEO_EXT, iter_frames, video_meta  # noqa: F401

PERSON_LABELS = ("person", "orang", "pekerja", "worker", "pedestrian", "people")
IGNORE_CLASS = 8           # kelas "distraktor" MOT17: box tracker yang cocok dengannya dibuang, tidak dinilai


@dataclass
class GtClip:
    name: str
    source: str                  # "cvat_xml" atau "mot_txt"
    width: int                   # ukuran frame saat dilabeli
    height: int
    n_frames: int                # jumlah frame yang dilabeli (frame 0..n_frames-1)
    boxes: pd.DataFrame          # frame (0-based, frame video), id, x1, y1, x2, y2, occluded, label
    video: Path | None = None
    notes: list = field(default_factory=list)   # peringatan saat membaca label
    fps: float = 0.0
    ignore_regions: list = field(default_factory=list)   # [(frame_dari, frame_sampai, [x1, y1, x2, y2], alasan)]
    labelled_frames: np.ndarray | None = None   # None = setiap frame dilabeli
    meta: dict = field(default_factory=dict)

    @property
    def ids(self) -> np.ndarray:
        return np.sort(self.boxes["id"].unique())


# ---------------- baca label ----------------

def _frame_map(meta) -> tuple[int, int]:
    """(start_frame, step) dari bagian <meta> CVAT. Frame label k = frame video start + k * step."""
    start, step = 0, 1
    if meta is None:
        return start, step
    task = meta.find("task") if meta.find("task") is not None else meta.find("job")
    if task is None:
        return start, step
    sf = task.findtext("start_frame")
    if sf and sf.strip().isdigit():
        start = int(sf)
    ff = (task.findtext("frame_filter") or "").strip()
    if ff.startswith("step="):
        step = int(ff.split("=", 1)[1])
    return start, step


def load_cvat_xml(path: str | Path, person_labels=PERSON_LABELS) -> GtClip:
    path = Path(path)
    root = ET.parse(path).getroot()
    meta = root.find("meta")
    start, step = _frame_map(meta)
    w = h = n = 0
    if meta is not None:
        for tag in ("task", "job"):
            node = meta.find(tag)
            if node is not None:
                n = int(node.findtext("size") or 0)
                osz = node.find("original_size")
                if osz is not None:
                    w, h = int(osz.findtext("width") or 0), int(osz.findtext("height") or 0)
                break
    labels = {l.lower() for l in person_labels}
    rows, notes, skipped = [], [], {}
    for tr in root.iter("track"):
        lab = tr.get("label", "")
        if lab.lower() not in labels:
            skipped[lab] = skipped.get(lab, 0) + 1
            continue
        tid = int(tr.get("id"))
        pts = []
        for b in tr.iter("box"):
            pts.append((int(b.get("frame")), b.get("outside", "0") == "1", b.get("occluded", "0") == "1",
                        float(b.get("xtl")), float(b.get("ytl")), float(b.get("xbr")), float(b.get("ybr"))))
        pts.sort()
        # CVAT for video biasanya sudah menulis setiap frame (termasuk hasil interpolasi). Kalau ada frame yang
        # kosong di antara dua box (export lama), isi dengan interpolasi linear, sama seperti di CVAT.
        for i, (f0, out0, occ0, *b0) in enumerate(pts):
            if out0:
                continue
            rows.append((f0, tid, *b0, occ0, lab))
            if i + 1 < len(pts):
                f1, _out1, _occ1, *b1 = pts[i + 1]
                for f in range(f0 + 1, f1):
                    a = (f - f0) / (f1 - f0)
                    rows.append((f, tid, *[(1 - a) * p + a * q for p, q in zip(b0, b1)], occ0, lab))
    if not rows:
        raise ValueError(f"{path.name}: tidak ada track orang. Label yang ada: {sorted(skipped) or 'tidak ada'}. "
                         f"Atur person_labels kalau nama labelnya berbeda.")
    if skipped:
        notes.append(f"label lain diabaikan: {skipped}")
    shapes = sum(1 for _ in root.iter("image"))
    if shapes:
        notes.append(f"ada {shapes} <image> (anotasi per gambar, tanpa ID). Pakai export 'CVAT for video', bukan 'for images'")
    df = pd.DataFrame(rows, columns=["frame", "id", "x1", "y1", "x2", "y2", "occluded", "label"])
    df["frame"] = start + df["frame"] * step
    if step != 1:
        notes.append(f"label hanya tiap {step} frame video (frame_filter)")
    df = df.drop_duplicates(["frame", "id"]).sort_values(["frame", "id"]).reset_index(drop=True)
    n = n or int(df.frame.max() + 1)
    return GtClip(path.parent.name, "cvat_xml", w, h, start + n * step, df, notes=notes)


def load_mot_txt(path: str | Path, width: int = 0, height: int = 0) -> GtClip:
    """gt.txt format MOT 1.1 dari CVAT: frame (mulai 1), id, x, y, w, h, not_ignored, class, visibility."""
    path = Path(path)
    a = np.loadtxt(path, delimiter=",", ndmin=2)
    vis = a[:, 8] if a.shape[1] > 8 else np.ones(len(a))
    keep = a[:, 6] != 0 if a.shape[1] > 6 else np.ones(len(a), bool)
    df = pd.DataFrame({"frame": a[keep, 0].astype(int) - 1, "id": a[keep, 1].astype(int),
                       "x1": a[keep, 2], "y1": a[keep, 3], "x2": a[keep, 2] + a[keep, 4], "y2": a[keep, 3] + a[keep, 5],
                       "occluded": vis[keep] < 0.5, "label": "person"})
    clip_dir = path.parent.parent if path.parent.name == "gt" else path.parent
    notes = [] if keep.all() else [f"{int((~keep).sum())} box ditandai diabaikan (not_ignored = 0) di file label"]
    return GtClip(clip_dir.name, "mot_txt", width, height, int(df.frame.max() + 1),
                  df.sort_values(["frame", "id"]).reset_index(drop=True), notes=notes)


def load_frozen(clip_dir: str | Path) -> GtClip:
    """Label beku PT SRP: gt.jsonl (satu baris per orang per frame) + meta.json."""
    import json

    d = Path(clip_dir)
    meta = json.loads((d / "meta.json").read_text())
    rows = [json.loads(l) for l in (d / "gt.jsonl").read_text().splitlines() if l.strip()]
    df = pd.DataFrame({
        "frame": [r["frame"] for r in rows], "id": [r["track_id"] for r in rows],
        "x1": [r["bbox"][0] for r in rows], "y1": [r["bbox"][1] for r in rows],
        "x2": [r["bbox"][2] for r in rows], "y2": [r["bbox"][3] for r in rows],
        "occluded": [bool(r.get("occluded", False)) for r in rows], "small": [bool(r.get("small", False)) for r in rows],
        "helmet": [r.get("helmet") for r in rows], "vest": [r.get("vest") for r in rows], "label": "person",
    }).sort_values(["frame", "id"]).reset_index(drop=True)
    fps = float(meta.get("fps") or 0)
    regions = []
    for r in meta.get("ignore_regions") or []:
        f0 = int(np.floor(float(r.get("from_s", 0)) * fps)) if fps else 0
        f1 = int(np.ceil(float(r.get("to_s", 1e9)) * fps)) if fps else 10 ** 9
        regions.append((f0, f1, [float(v) for v in r["bbox"]], r.get("reason", "")))
    notes = []
    labelled = None
    if meta.get("gt_step") in (None, 0):
        labelled = np.sort(df.frame.unique())
        notes.append(f"label hanya di {len(labelled)} frame (tidak setiap frame): hanya frame itu yang dinilai")
    if meta.get("shift_frames"):
        notes.append(f"meta: shift_frames = {meta['shift_frames']} (cek pergeseran label vs video di bagian 1)")
    clip = GtClip(meta.get("clip", d.name), "frozen", int(meta.get("width") or 0), int(meta.get("height") or 0),
                  int(meta.get("n_frames") or df.frame.max() + 1), df, notes=notes, fps=fps,
                  ignore_regions=regions, labelled_frames=labelled, meta=meta)
    return clip


def find_clips(gt_dir: str | Path, video_dirs=(), person_labels=PERSON_LABELS, variants: bool = True) -> list[GtClip]:
    """Semua klip di gt_dir. Satu subfolder = satu klip (gt.jsonl + meta.json, annotations.xml, atau gt/gt.txt).
    variants=False melewati klip turunan yang namanya berakhiran _blur atau _reenc."""
    clips = []
    for d in sorted(p for p in Path(gt_dir).iterdir() if p.is_dir()):
        if not variants and d.name.endswith(("_blur", "_reenc")):
            continue
        xml = next(iter(sorted(d.glob("*.xml"))), None)
        txt = d / "gt" / "gt.txt" if (d / "gt" / "gt.txt").exists() else next(iter(sorted(d.glob("gt*.txt"))), None)
        if (d / "gt.jsonl").exists() and (d / "meta.json").exists():
            clip = load_frozen(d)
        elif xml is not None:
            clip = load_cvat_xml(xml, person_labels)
        elif txt is not None:
            clip = load_mot_txt(txt)
        else:
            continue
        clip.name = d.name
        hint = Path(clip.meta["video"]).name if clip.meta.get("video") else None
        clip.video = _find_video(d, list(video_dirs) + [Path(gt_dir).parent], hint)
        if clip.video is not None and (clip.width == 0 or clip.height == 0):
            m = video_meta(clip.video)
            clip.width, clip.height = m["width"], m["height"]
        clips.append(clip)
    return clips


def _find_video(clip_dir: Path, video_dirs, hint: str | None = None) -> Path | None:
    vids = [p for p in sorted(clip_dir.iterdir()) if p.suffix.lower() in VIDEO_EXT]
    if vids:
        return vids[0]
    names = ([hint] if hint else []) + [f"{clip_dir.name}{ext}" for ext in VIDEO_EXT]
    for vd in video_dirs:
        for n in names:
            p = Path(vd) / n
            if p.exists():
                return p
    return None


# ---------------- cek label ----------------

def check(clip: GtClip) -> dict:
    """Ringkasan dan peringatan untuk satu klip. Jalankan sebelum menilai tracker."""
    b = clip.boxes
    warn = list(clip.notes)
    info = {"klip": clip.name, "format": clip.source, "orang (ID)": int(b.id.nunique()), "box": len(b),
            "frame dilabeli": clip.n_frames if clip.labelled_frames is None else len(clip.labelled_frames),
            "persen tertutup": round(100 * float(b.occluded.mean()), 1),
            "persen kecil": round(100 * float(b["small"].mean()), 1) if "small" in b else 0.0,
            "area diabaikan": len(clip.ignore_regions)}
    if clip.video is None:
        warn.append("video tidak ditemukan")
    else:
        m = video_meta(clip.video)
        info.update({"video": clip.video.name, "FPS video": round(m["fps"], 2), "frame video": m["frames"],
                     "durasi (detik)": round(m["duration"], 1)})
        if abs(m["frames"] - clip.n_frames) > 1:
            warn.append(f"jumlah frame berbeda: label {clip.n_frames}, video {m['frames']}")
        if clip.width and (m["width"], m["height"]) != (clip.width, clip.height):
            warn.append(f"ukuran berbeda: label {clip.width}x{clip.height}, video {m['width']}x{m['height']} "
                        "(box akan diskalakan)")
    hgt = (b.y2 - b.y1)
    tiny = int((hgt < 20).sum())
    if tiny:
        warn.append(f"{tiny} box sangat kecil (tinggi < 20 px)")
    per_id = b.groupby("id").frame.agg(["min", "max", "count"])
    gaps = per_id[(per_id["max"] - per_id["min"] + 1) > per_id["count"] * 1.5]
    if len(gaps):
        warn.append(f"{len(gaps)} ID punya jeda panjang (keluar lalu masuk lagi dengan ID sama, cek apakah benar orang yang sama)")
    dup = b.duplicated(["frame", "id"]).sum()
    if dup:
        warn.append(f"{dup} box dobel (ID sama di frame sama)")
    info["peringatan"] = "; ".join(warn) if warn else "-"
    return info


def overlay(clip: GtClip, frames=None, n: int = 3, width: int = 640):
    """Gambar box label di atas frame video (untuk cek visual). Hasil: gambar RGB gabungan."""
    import cv2

    from .live_compare import id_color

    if clip.video is None:
        raise FileNotFoundError(f"{clip.name}: video tidak ditemukan")
    m = video_meta(clip.video)
    sx = m["width"] / (clip.width or m["width"])
    sy = m["height"] / (clip.height or m["height"])
    if frames is None:
        busy = clip.boxes.groupby("frame").size()
        cand = np.sort(clip.boxes.frame.unique())
        spread = [int(cand[int(i)]) for i in np.linspace(0, len(cand) - 1, n + 2)[1:-1]]
        frames = sorted(set([int(busy.idxmax())] + spread))[:n]
    cap = cv2.VideoCapture(str(clip.video))
    tiles = []
    for f in frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, img = cap.read()
        if not ok:
            continue
        s = width / img.shape[1]
        img = cv2.resize(img, (width, int(img.shape[0] * s)))
        for r in clip.boxes[clip.boxes.frame == f].itertuples():
            p1 = (int(r.x1 * sx * s), int(r.y1 * sy * s))
            p2 = (int(r.x2 * sx * s), int(r.y2 * sy * s))
            col = id_color(int(r.id))
            cv2.rectangle(img, p1, p2, col, 1 if r.occluded else 2, cv2.LINE_AA)
            tag = (" T" if r.occluded else "") + (" K" if getattr(r, "small", False) else "")
            cv2.putText(img, f"{r.id}{tag}", (p1[0], max(12, p1[1] - 3)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        for f0, f1, rb, _ in clip.ignore_regions:
            if f0 <= f <= f1:
                q1 = (int(rb[0] * sx * s), int(rb[1] * sy * s))
                q2 = (int(rb[2] * sx * s), int(rb[3] * sy * s))
                cv2.rectangle(img, q1, q2, (160, 160, 160), 1, cv2.LINE_4)
                cv2.putText(img, "diabaikan", (q1[0] + 3, q1[1] + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        cv2.putText(img, f"frame {f}", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(img)
    cap.release()
    if not tiles:
        raise RuntimeError(f"{clip.name}: frame tidak bisa dibaca")
    return cv2.cvtColor(np.hstack(tiles), cv2.COLOR_BGR2RGB)


# ---------------- deteksi dan Seq ----------------

def detect_clip(detector, clip: GtClip, cache_path: str | Path | None = None, max_sec: float | None = None,
                conf: float = 0.05, progress=None) -> dict:
    """Deteksi orang di SETIAP frame video (supaya semua FPS proses bisa dinilai dari cache yang sama).
    Hasil: {frame (1-based): Nx5 [x1, y1, x2, y2, skor]}, disimpan ke cache_path kalau diberikan."""
    if cache_path is not None and Path(cache_path).exists():
        z = np.load(cache_path)
        return {int(k[1:]): z[k] for k in z.files if k.startswith("f")}
    m = video_meta(clip.video)
    dets, ms = {}, []
    it = iter_frames(clip.video, m["fps"], max_sec)
    if progress is not None:
        it = progress(it, total=int((max_sec or m["duration"]) * m["fps"]))
    for t, frame, _ in it:
        t0 = time.perf_counter()
        persons, _, _, _ = detector.detect_timed(frame, conf_thresh=conf)
        ms.append((time.perf_counter() - t0) * 1000)
        f = int(round(t * m["fps"])) + 1
        dets[f] = np.array([[b.x1, b.y1, b.x2, b.y2, b.conf] for b in persons], np.float32).reshape(-1, 5)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, ms=np.array(ms), **{f"f{k}": v for k, v in dets.items()})
    return dets


def oracle_dets(clip: GtClip, video_size: tuple[int, int] | None = None, score: float = 0.9,
                include_occluded: bool = True) -> dict:
    """"Detektor sempurna": box label dipakai sebagai deteksi. Untuk mengukur kesalahan yang murni dari tracker."""
    sx, sy = _scale(clip, video_size)
    b = clip.boxes if include_occluded else clip.boxes[~clip.boxes.occluded]
    out = {}
    for f, g in b.groupby("frame"):
        arr = np.stack([g.x1 * sx, g.y1 * sy, g.x2 * sx, g.y2 * sy, np.full(len(g), score)], 1)
        out[int(f) + 1] = arr.astype(np.float32)
    return out


def _scale(clip: GtClip, video_size) -> tuple[float, float]:
    if not video_size or not clip.width:
        return 1.0, 1.0
    return video_size[0] / clip.width, video_size[1] / clip.height


class GtSeq(mot17.Seq):
    """Klip CCTV dalam bentuk mot17.Seq. Selalu dianggap kamera diam.
    ignore_regions: [(frame_dari, frame_sampai, [x1, y1, x2, y2])], frame mulai 1. Box tracker yang separuh lebih
    luasnya ada di area ini tidak dinilai. eval_frames: frame (mulai 1) yang dinilai, None = semua."""

    ignore_regions: list = []
    eval_frames: set | None = None

    @property
    def static(self) -> bool:
        return True


def to_seq(clip: GtClip, dets: dict, detector_name: str, occluded: str = "nilai", small: str = "abaikan") -> GtSeq:
    """occluded / small: "nilai" = dinilai seperti orang biasa, "abaikan" = tidak dinilai (box tracker di atasnya
    dibuang, tidak dihitung benar maupun salah, sama seperti distraktor MOT17)."""
    for k, v in (("occluded", occluded), ("small", small)):
        if v not in ("abaikan", "nilai"):
            raise ValueError(f"{k} harus 'abaikan' atau 'nilai'")
    m = video_meta(clip.video)
    sx, sy = _scale(clip, (m["width"], m["height"]))
    b = clip.boxes
    ign = b.occluded.to_numpy() & (occluded == "abaikan")
    if "small" in b:
        ign = ign | (b["small"].to_numpy().astype(bool) & (small == "abaikan"))
    gt = np.stack([b.frame + 1, b.id, b.x1 * sx, b.y1 * sy, (b.x2 - b.x1) * sx, (b.y2 - b.y1) * sy,
                   np.where(ign, 0, 1), np.where(ign, IGNORE_CLASS, 1), np.where(b.occluded, 0.3, 1.0)], 1).astype(float)
    length = min(m["frames"], max(dets)) if dets else m["frames"]
    seq = GtSeq(clip.name, detector_name, float(m["fps"]), int(length), int(m["width"]), int(m["height"]), dets, gt)
    seq.ignore_regions = [(f0 + 1, f1 + 1, [rb[0] * sx, rb[1] * sy, rb[2] * sx, rb[3] * sy]) for f0, f1, rb, _ in clip.ignore_regions]
    seq.eval_frames = None if clip.labelled_frames is None else {int(f) + 1 for f in clip.labelled_frames}
    return seq


# ---------------- output pipeline lain ----------------

def run_info(run_dir: str | Path) -> dict:
    """Ringkasan run.json: pipeline, tracker, FPS, ambang orang, detektor, cara mengambil frame."""
    import json

    p = Path(run_dir) / "run.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    c = d.get("config", {})
    sampler = c.get("sampler") or ("ffmpeg" if str(d.get("pipeline", "")).startswith("srp-ppe-be") else "exact")
    return {"run": Path(run_dir).name, "pipeline": d.get("pipeline", "?"), "tracker": c.get("tracker", "?"),
            "fps": float(c.get("sample_fps") or 0), "person": (c.get("thresholds") or {}).get("person"),
            "person_to_tracker": c.get("person_to_tracker", (c.get("thresholds") or {}).get("person")),
            "detector_sha": (d.get("weights") or {}).get("sha256", ""), "clips": d.get("clips", []),
            "timing": d.get("timing", {}), "note": d.get("note", ""), "sampler": sampler, "sweep": d.get("sweep")}


def sample_frame(k: int, sample_fps: float, fps: float, sampler: str = "exact") -> int:
    """Frame video (mulai 0) yang dilihat pipeline pada sampel ke-k. Sama dengan evaluate.sample_frame di
    hen-detect-ppe: "ffmpeg" (replay BE, `-vf fps=F`) = floor((k + 0,5) x fps / F), yaitu setengah interval
    sesudah mono_ts (3 fps: frame 2, 7, 12, ...). "exact" = round(k / F x fps) (3 fps: frame 0, 5, 10, ...)."""
    import math

    if sampler == "ffmpeg":
        return int(math.floor(round((k + 0.5) * fps / sample_fps, 6)))
    return int(math.floor(round(k / sample_fps * fps, 6) + 0.5))


def load_run(run_dir: str | Path, clip: str, fps: float, shift: int = 0, labelled: set | None = None,
             max_snap: int = 2) -> dict:
    """observations.jsonl sebuah run: {frame (mulai 1): (ids, boxes xyxy)}.

    Waktu sampel (mono_ts) diubah ke frame yang benar-benar dilihat pipeline, sesuai config.sampler di run.json
    (lihat sample_frame), lalu digeser `shift` frame. Kalau klip hanya dilabeli di sebagian frame (`labelled`,
    nomor frame mulai 0), frame digeser ke frame berlabel terdekat (paling jauh max_snap frame)."""
    import json

    info = run_info(run_dir)
    sfps = info["fps"] or 3.0
    rows = []
    for line in (Path(run_dir) / clip / "observations.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            f0 = sample_frame(int(round(float(r["mono_ts"]) * sfps)), sfps, fps, info["sampler"]) + shift
            rows.append((f0, int(r["track_id"]), r["bbox"]))
    if labelled:
        # tiap frame berlabel memakai sampel terdekat saja (FPS tinggi: beberapa sampel dekat satu frame berlabel)
        lab = np.array(sorted(labelled))
        sampled = np.array(sorted({f for f, _, _ in rows}))
        to_lab = {}
        for f in sampled:
            j = int(np.abs(lab - f).argmin())
            d = abs(int(lab[j]) - int(f))
            if d <= max_snap and (int(lab[j]) not in to_lab or d < to_lab[int(lab[j])][1]):
                to_lab[int(lab[j])] = (int(f), d)
        src = {f: lf for lf, (f, _) in to_lab.items()}
        rows = [(src[f], i, b) for f, i, b in rows if f in src]
    out: dict = {}
    for f0, i, b in rows:
        out.setdefault(f0 + 1, ([], []))
        out[f0 + 1][0].append(i)
        out[f0 + 1][1].append(b)
    return {f: (np.array(i, np.int64), np.array(b, np.float64).reshape(-1, 4)) for f, (i, b) in sorted(out.items())}


def estimate_shift(clip: GtClip, hyp: dict, shifts=range(-4, 5)) -> tuple[int, dict]:
    """Cari pergeseran frame yang membuat box hasil (frame mulai 1, shift 0) paling pas dengan label.
    Hasil: (shift terbaik, {shift: rata-rata IoU terbaik per box}). Butuh label di setiap frame."""
    from .lite import iou_matrix

    by_f = {int(f): g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32) for f, g in clip.boxes.groupby("frame")}
    score = {}
    for s in shifts:
        vals = []
        for f, (_, boxes) in hyp.items():
            g = by_f.get(f - 1 + s)
            if g is None or not len(boxes):
                continue
            vals += list(iou_matrix(np.asarray(boxes, np.float32), g).max(1))
        score[s] = float(np.mean(vals)) if vals else float("nan")
    valid = {k: v for k, v in score.items() if v == v}
    best = max(valid, key=valid.get) if valid else 0
    return best, score


def eval_run(seq: GtSeq, run_dir: str | Path, fps_src: float, shift: int = 0, clip: GtClip | None = None) -> dict:
    """Nilai output pipeline lain dengan label seq. Hasil: satu baris seperti mot17.run_benchmark.
    clip: kalau klip hanya dilabeli di sebagian frame, frame pipeline digeser ke frame berlabel terdekat."""
    info = run_info(run_dir)
    labelled = None if clip is None or clip.labelled_frames is None else set(int(f) for f in clip.labelled_frames)
    hyp = load_run(run_dir, seq.name, fps_src, shift, labelled)
    hyp = {f: v for f, v in hyp.items() if f <= seq.length}
    t = info["timing"].get(seq.name, {})
    fps = info["fps"]
    return {"tracker": f"pipeline: {info['run']}", "seq": seq.name, "detektor": f"pipeline {info['detector_sha'][:8]}",
            "kamera": "diam", "target_fps": int(fps) if fps == int(fps) else fps, "fps": fps, "frames": len(hyp),
            "ms_per_frame": float(t.get("ms_per_frame", 0.0)), "ids_created": len({i for v in hyp.values() for i in v[0]}),
            "raw": mot17.evaluate(seq, hyp), "shift": shift, "sampler": info["sampler"],
            "label_digeser": labelled is not None}


# ---------------- pergeseran label vs video ----------------

def label_offset(clip: GtClip, dets: dict, offsets=range(-4, 5), conf: float = 0.5, step: int = 5) -> tuple[int, dict]:
    """Cari pergeseran label terhadap video memakai deteksi di SETIAP frame (dets dari detect_clip, frame mulai 1).
    Label di frame f dibandingkan dengan deteksi di frame f + o. Hasil: (o terbaik, {o: rata-rata IoU terbaik}).
    o = 2 artinya label terlambat 2 frame (box di frame f sebenarnya menggambar frame f + 2). Bisa dipakai juga
    untuk klip yang hanya dilabeli tiap 5 frame, karena deteksinya ada di semua frame."""
    from .lite import iou_matrix

    score = {}
    groups = [(int(f), g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32)) for f, g in clip.boxes.groupby("frame")
              if int(f) % step == int(clip.boxes.frame.min()) % step]
    for o in offsets:
        vals = []
        for f, gb in groups:
            d = dets.get(f + 1 + o)
            if d is None or not len(d) or not len(gb):
                continue
            d = d[d[:, 4] >= conf]
            if len(d):
                vals += list(iou_matrix(gb, d[:, :4].astype(np.float32)).max(1))
        score[o] = float(np.mean(vals)) if vals else float("nan")
    valid = {k: v for k, v in score.items() if v == v}
    return (max(valid, key=valid.get) if valid else 0), score


def shift_labels(clip: GtClip, offset: int) -> GtClip:
    """Salinan klip dengan label digeser `offset` frame (label frame f menjadi frame f + offset)."""
    from dataclasses import replace

    if not offset:
        return clip
    b = clip.boxes.copy()
    b["frame"] = b["frame"] + offset
    b = b[(b.frame >= 0) & (b.frame < clip.n_frames)].reset_index(drop=True)
    lab = None if clip.labelled_frames is None else np.sort(np.unique(b.frame.to_numpy()))
    regions = [(f0 + offset, f1 + offset, box, why) for f0, f1, box, why in clip.ignore_regions]
    return replace(clip, boxes=b, labelled_frames=lab, ignore_regions=regions,
                   notes=clip.notes + [f"label digeser {offset:+d} frame supaya sejajar dengan video"])


# ---------------- tolok ukur detektor ----------------
# Definisi tetap supaya angka detektor lama dan baru bisa dibandingkan. Ubah DET_BENCH_VERSION kalau aturannya berubah.
DET_BENCH_VERSION = "1"
DET_BENCH_RULES = {
    "frame": "frame berlabel di grid 3 fps (tiap 5 frame video)",
    "cocok": "satu GT satu deteksi (Hungarian), IoU >= 0,5",
    "diabaikan": "GT kecil (tinggi < min_h_px, 60 px) dan deteksi yang cocok dengannya; deteksi tak cocok yang kecil "
                 "atau separuh lebih luasnya di area abaikan",
    "tertutup": "orang tertutup tetap dinilai",
    "nms": "tidak ada (sama dengan pipeline hen)",
}
EDGE_PX, CROWD_IOA, SMALL_FRAC, STATIC_REPEATS, IOU_NEAR = 15, 0.3, 0.10, 5, 0.2
FN_ORDER = ["skor rendah", "box meleset", "tertutup", "kerumunan", "tepi frame", "kecil/jauh", "lainnya"]
FP_ORDER = ["box dobel", "box meleset", "objek diam berulang", "latar/lainnya"]


def _ioa_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Porsi luas tiap box a yang ada di dalam tiap box b. a: Nx4, b: Mx4."""
    iw = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    area = np.maximum((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]), 1e-6)
    return iw * ih / area[:, None]


def bench_frames(clip: GtClip, step: int = 5) -> list[int]:
    """Frame (mulai 0) yang dinilai: frame berlabel di grid 3 fps, dimulai dari frame berlabel pertama."""
    lab = np.sort(clip.boxes.frame.unique()) if clip.labelled_frames is None else np.asarray(clip.labelled_frames)
    first = int(lab.min()) if len(lab) else 0
    if clip.labelled_frames is None:
        return list(range(first, clip.n_frames, step))
    return [int(f) for f in lab if (int(f) - first) % step == 0]


def detector_errors(clip: GtClip, dets: dict, conf: float = 0.5, iou_thr: float = 0.5, low: float = 0.1,
                    frames: list[int] | None = None) -> pd.DataFrame:
    """Satu baris per orang GT dan per deteksi di frame yang dinilai: kind TP / FN / FP / IGN, sebab utama
    (`sebab`) dan semua tanda (`tanda`). dets: {frame mulai 1: Nx5 [x1, y1, x2, y2, skor]} di resolusi label."""
    from scipy.optimize import linear_sum_assignment
    from .lite import iou_matrix

    W, H = clip.width or 1, clip.height or 1
    min_h = float(clip.meta.get("min_h_px", 60)) if clip.meta else 60.0
    frames = bench_frames(clip) if frames is None else frames
    by_f = {int(f): g for f, g in clip.boxes.groupby("frame")}
    empty = np.zeros((0, 5), np.float32)
    rows = []
    for f in frames:
        g = by_f.get(f)
        gb = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32) if g is not None else np.zeros((0, 4), np.float32)
        small = (g["small"].to_numpy(bool) if g is not None and "small" in g else np.zeros(len(gb), bool)) | \
                ((gb[:, 3] - gb[:, 1]) < min_h)
        occl = g["occluded"].to_numpy(bool) if g is not None else np.zeros(len(gb), bool)
        gid = g["id"].to_numpy() if g is not None else np.zeros(0)
        d_all = dets.get(f + 1, empty)
        d = d_all[d_all[:, 4] >= conf]
        dl = d_all[(d_all[:, 4] >= low) & (d_all[:, 4] < conf)]
        db = d[:, :4].astype(np.float32)
        ious = iou_matrix(gb, db) if len(gb) and len(db) else np.zeros((len(gb), len(db)), np.float32)
        gm, pm = [], []
        if ious.size:
            r, c = linear_sum_assignment(-ious)
            ok = ious[r, c] >= iou_thr
            gm, pm = list(r[ok]), list(c[ok])
        crowd = (_ioa_xyxy(gb, gb) - np.eye(len(gb))).max(1) >= CROWD_IOA if len(gb) > 1 else np.zeros(len(gb), bool)
        regions = [rb for f0, f1, rb, _ in clip.ignore_regions if f0 <= f <= f1]
        t = f / (clip.fps or 15)
        base = {"klip": clip.name, "frame": f, "t_s": round(t, 2)}
        for gi, pi in zip(gm, pm):
            rows.append({**base, "kind": "IGN" if small[gi] else "TP", "gt_id": gid[gi], "box": gb[gi].tolist(),
                         "skor": float(d[pi, 4]), "iou": float(ious[gi, pi]), "tinggi_px": float(gb[gi, 3] - gb[gi, 1]),
                         "tertutup": bool(occl[gi]), "sebab": "", "tanda": ""})
        for gi in range(len(gb)):
            if gi in gm:
                continue
            b = gb[gi]
            tags = []
            if len(dl) and iou_matrix(b[None], dl[:, :4].astype(np.float32)).max() >= iou_thr:
                tags.append("skor rendah")
            if ious.shape[1] and ious[gi].max() >= IOU_NEAR:
                tags.append("box meleset")
            if occl[gi]:
                tags.append("tertutup")
            if crowd[gi]:
                tags.append("kerumunan")
            if b[0] <= EDGE_PX or b[1] <= EDGE_PX or b[2] >= W - EDGE_PX or b[3] >= H - EDGE_PX:
                tags.append("tepi frame")
            if (b[3] - b[1]) < SMALL_FRAC * H:
                tags.append("kecil/jauh")
            tags = tags or ["lainnya"]
            rows.append({**base, "kind": "IGN" if small[gi] else "FN", "gt_id": gid[gi], "box": b.tolist(), "skor": np.nan,
                         "iou": float(ious[gi].max()) if ious.shape[1] else 0.0, "tinggi_px": float(b[3] - b[1]),
                         "tertutup": bool(occl[gi]), "sebab": tags[0], "tanda": ";".join(tags)})
        for pi in range(len(db)):
            if pi in pm:
                continue
            b = db[pi]
            ign = (b[3] - b[1]) < min_h
            if regions:
                ign = ign or _ioa_xyxy(b[None], np.array(regions, np.float32)).max() >= 0.5
            best = float(ious[:, pi].max()) if ious.shape[0] else 0.0
            dup = (ious.shape[0] and any(ious[gi, pi] >= iou_thr for gi in gm)) or \
                  (len(pm) and iou_matrix(b[None], db[pm]).max() >= iou_thr)
            tags = (["box dobel"] if dup else []) + (["box meleset"] if best >= IOU_NEAR and not dup else [])
            rows.append({**base, "kind": "IGN" if ign else "FP", "gt_id": None, "box": b.tolist(), "skor": float(d[pi, 4]),
                         "iou": best, "tinggi_px": float(b[3] - b[1]), "tertutup": None, "sebab": "", "tanda": ";".join(tags)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    fp = df.index[df.kind == "FP"]
    if len(fp):   # box di tempat yang sama di banyak frame = objek diam (rak, tiang, handle hand pallet, ...)
        boxes = np.array(df.loc[fp, "box"].tolist(), np.float32)
        same = (iou_matrix(boxes, boxes) >= 0.7).sum(1) - 1
        for idx, n in zip(fp, same):
            tags = [t for t in df.at[idx, "tanda"].split(";") if t]
            if n >= STATIC_REPEATS:
                tags.append("objek diam berulang")
            tags = tags or ["latar/lainnya"]
            df.at[idx, "tanda"] = ";".join(tags)
            df.at[idx, "sebab"] = next(t for t in FP_ORDER if t in tags)
    return df


def average_precision(clip: GtClip, dets: dict, frames: list[int] | None = None, iou_thr: float = 0.5,
                      min_score: float = 0.05) -> float:
    """AP50 (interpolasi semua titik, gaya VOC) di frame yang dinilai. GT kecil dan area abaikan tidak dihitung."""
    from .lite import iou_matrix

    min_h = float(clip.meta.get("min_h_px", 60)) if clip.meta else 60.0
    frames = bench_frames(clip) if frames is None else frames
    by_f = {int(f): g for f, g in clip.boxes.groupby("frame")}
    scores, tps, n_gt = [], [], 0
    for f in frames:
        g = by_f.get(f)
        gb = g[["x1", "y1", "x2", "y2"]].to_numpy(np.float32) if g is not None else np.zeros((0, 4), np.float32)
        small = ((gb[:, 3] - gb[:, 1]) < min_h) | (g["small"].to_numpy(bool) if g is not None and "small" in g else False)
        n_gt += int((~small).sum())
        d = dets.get(f + 1, np.zeros((0, 5), np.float32))
        d = d[d[:, 4] >= min_score]
        d = d[np.argsort(-d[:, 4])]
        regions = [rb for f0, f1, rb, _ in clip.ignore_regions if f0 <= f <= f1]
        used = np.zeros(len(gb), bool)
        ious = iou_matrix(d[:, :4].astype(np.float32), gb) if len(d) and len(gb) else np.zeros((len(d), len(gb)))
        for k in range(len(d)):
            j = -1
            if ious.shape[1]:
                cand = np.where(~used & (ious[k] >= iou_thr))[0]
                if len(cand):
                    j = int(cand[np.argmax(ious[k, cand])])
            if j >= 0:
                used[j] = True
                if small[j]:
                    continue
                scores.append(d[k, 4]); tps.append(1)
            else:
                b = d[k, :4]
                if (b[3] - b[1]) < min_h or (regions and _ioa_xyxy(b[None], np.array(regions, np.float32)).max() >= 0.5):
                    continue
                scores.append(d[k, 4]); tps.append(0)
    if not n_gt:
        return float("nan")
    o = np.argsort(-np.array(scores))
    tp = np.array(tps)[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0], rec, [1]])
    mpre = np.concatenate([[1], prec, [0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    i = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1]))


def detector_benchmark(clip: GtClip, dets: dict, detector: str, conf: float = 0.5,
                       sweep=(0.2, 0.3, 0.4, 0.5, 0.6)) -> tuple[dict, pd.DataFrame]:
    """Ringkasan satu klip untuk satu detektor: recall, presisi, F1 di `conf`, AP50, recall di beberapa ambang,
    jumlah FN dan FP per sebab. Hasil: (baris ringkasan, tabel kesalahan dari detector_errors)."""
    err = detector_errors(clip, dets, conf)
    n = err.kind.value_counts() if len(err) else pd.Series(dtype=int)
    tp, fn, fp = int(n.get("TP", 0)), int(n.get("FN", 0)), int(n.get("FP", 0))
    row = {"versi": DET_BENCH_VERSION, "klip": clip.name, "detektor": detector, "gt_version": clip.meta.get("gt_version", ""),
           "frame": len(bench_frames(clip)), "orang": tp + fn, "ambang": conf, "TP": tp, "FN": fn, "FP": fp,
           "recall": tp / max(tp + fn, 1), "presisi": tp / max(tp + fp, 1),
           "AP50": average_precision(clip, dets)}
    row["F1"] = 2 * row["recall"] * row["presisi"] / max(row["recall"] + row["presisi"], 1e-9)
    for c in sweep:
        if c == conf:
            row[f"recall@{c:g}"], row[f"FP@{c:g}"] = row["recall"], fp
            continue
        e = detector_errors(clip, dets, c)
        k = e.kind.value_counts() if len(e) else pd.Series(dtype=int)
        row[f"recall@{c:g}"] = int(k.get("TP", 0)) / max(int(k.get("TP", 0)) + int(k.get("FN", 0)), 1)
        row[f"FP@{c:g}"] = int(k.get("FP", 0))
    for s in FN_ORDER:
        row[f"FN {s}"] = int(((err.kind == "FN") & (err.sebab == s)).sum()) if len(err) else 0
    for s in FP_ORDER:
        row[f"FP {s}"] = int(((err.kind == "FP") & (err.sebab == s)).sum()) if len(err) else 0
    return row, err


def benchmark_table(rows: list[dict], by=("detektor",)) -> pd.DataFrame:
    """Gabungkan baris detector_benchmark (jumlah dijumlah, rasio dihitung ulang; AP50 = rata-rata per klip)."""
    df = pd.DataFrame(rows)
    cnt = [c for c in df.columns if c in ("frame", "orang", "TP", "FN", "FP") or c.startswith(("FN ", "FP ", "FP@"))]
    g = df.groupby(list(by))
    out = g[cnt].sum()
    out["klip"] = g.size()
    out["recall"] = out.TP / out.orang.clip(lower=1)
    out["presisi"] = out.TP / (out.TP + out.FP).clip(lower=1)
    out["F1"] = 2 * out.recall * out.presisi / (out.recall + out.presisi).clip(lower=1e-9)
    out["AP50 (rata-rata klip)"] = g["AP50"].mean()
    for c in [c for c in df.columns if c.startswith("recall@")]:
        out[c] = (g.apply(lambda x, c=c: (x[c] * x["orang"]).sum() / max(x["orang"].sum(), 1)))
    return out
