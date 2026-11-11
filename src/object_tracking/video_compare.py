"""Video perbandingan tracker dari video MILIK SENDIRI (upload atau path), untuk notebook.

Beda dengan `mot17_video`: video sendiri tidak punya label manusia maupun deteksi publik, jadi orang dideteksi
dengan detektor D-FINE proyek ini (sekali per frame yang diproses), lalu deteksi yang sama dibagikan ke semua
tracker. Grid: video asli, deteksi mentah, satu panel per tracker. Tanpa label, ID switch tidak bisa dihitung
pasti; sebagai gantinya panel menandai "ID baru di tengah" (label kuning), yang biasanya tanda ID switch.

Alur di notebook:
    picker = video_picker(upload_dir, [folder_video_lain])    # upload / pilih / tempel path
    res = render(detector, picker.selected(), specs, out_path, fps=5)
    display(fps_table(res)); display(download_link(res["video"]))
"""

from __future__ import annotations

import base64
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .live_compare import LiveCompare, list_videos, video_meta
from .zoo import TrackerSpec


def video_picker(upload_dir: str | Path, video_dirs=(), default: str | Path | None = None):
    """Widget: upload video, pilih dari daftar, atau tempel path. `.selected()` mengembalikan path terpilih."""
    import ipywidgets as W

    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dirs = [upload_dir] + [Path(d) for d in video_dirs]

    def options():
        return [(f"{p.name}  ({p.parent.name}, {p.stat().st_size / 1e6:.0f} MB)", str(p)) for p in list_videos(dirs)]

    opts = options()
    dd = W.Dropdown(options=opts, description="Video", layout=W.Layout(width="560px"),
                    value=str(default) if default and str(default) in [v for _, v in opts] else (opts[0][1] if opts else None))
    up = W.FileUpload(accept="video/*,.mp4,.mov,.avi,.mkv,.m4v,.webm", multiple=False, description="Upload video")
    txt = W.Text(placeholder="atau tempel path lengkap video (cara paling aman untuk file besar)",
                 description="Path", layout=W.Layout(width="560px"))
    info = W.HTML()

    def describe(p):
        try:
            m = video_meta(p)
            info.value = (f"<b>{Path(p).name}</b>: {m['width']}x{m['height']}, <b>{m['fps']:.2f} FPS</b>, "
                          f"{m['frames']} frame, {m['duration']:.1f} detik")
        except Exception as e:
            info.value = f"<b style='color:red'>tidak bisa dibuka: {e}</b>"

    def on_upload(change):
        files = up.value
        if not files:
            return
        f = files[0]
        dst = upload_dir / Path(f["name"]).name
        dst.write_bytes(bytes(f["content"]))
        dd.options = options()
        dd.value = str(dst)
        txt.value = ""

    dd.observe(lambda c: c["new"] and describe(c["new"]), names="value")
    txt.observe(lambda c: Path(c["new"].strip()).exists() and describe(c["new"].strip()), names="value")
    up.observe(on_upload, names="value")
    if dd.value:
        describe(dd.value)
    box = W.VBox([W.HBox([up, W.HTML(f"&nbsp;tersimpan ke <code>{upload_dir}</code>")]), dd, txt, info])

    def selected() -> Path:
        p = txt.value.strip().strip("'\"") or dd.value
        if not p or not Path(p).expanduser().exists():
            raise FileNotFoundError("Belum ada video: upload, pilih dari daftar, atau tempel path di sel sebelumnya.")
        return Path(p).expanduser()

    box.selected = selected
    return box


def render(detector, source: str | Path, specs: dict[str, TrackerSpec], out_path: str | Path, fps: float = 5.0,
           panel_width: int = 640, start_sec: float = 0.0, max_sec: float | None = None, show_raw: bool = True,
           progress: bool = True) -> dict:
    """Buat video grid [asli] [deteksi mentah] [tracker ...] dan simpan sebagai mp4 (H.264 kalau tersedia)."""
    named = [s.with_params(name=n) for n, s in specs.items()]     # nama panel = kunci dict
    lc = LiveCompare(detector, named, fps=fps, panel_width=panel_width, show_raw=show_raw, show_original=True)
    meta = video_meta(source)
    span = (meta["duration"] - start_sec) if meta["duration"] else 0
    if max_sec:
        span = min(span, max_sec) if span > 0 else max_sec
    total = int(span * fps) + 1 if span > 0 else None
    bar = None
    if progress:
        try:
            from tqdm.auto import tqdm
            bar = tqdm(total=total, desc="render", unit="frame")
        except ImportError:
            pass
    out_path = Path(out_path)
    t0 = time.perf_counter()
    res = lc.run(source, start_sec=start_sec, max_sec=max_sec, realtime=False, save_path=out_path,
                 show=(lambda _b: bar.update(1)) if bar is not None else None)
    if bar is not None:
        bar.close()
    wall = time.perf_counter() - t0
    snap = out_path.with_suffix(".jpg")
    if lc.snapshot is not None:
        cv2.imwrite(str(snap), lc.snapshot, [cv2.IMWRITE_JPEG_QUALITY, 88])
    det_ms = float(np.median(lc.det_ms)) if lc.det_ms else 0.0
    dec_ms = float(np.median(lc.decode_ms)) if lc.decode_ms else 0.0
    return {"video": out_path, "snapshot": snap, "source": meta, "proc_fps": lc.proc_fps, "frames": res["frames"],
            "rows": res["rows"], "det_ms": det_ms, "decode_ms": dec_ms, "wall_sec": wall,
            "size": None if lc.snapshot is None else lc.snapshot.shape[1::-1]}


def fps_table(res: dict) -> pd.DataFrame:
    """Keterangan FPS: video sumber, proses, video hasil, dan kecepatan nyata di mesin ini."""
    src = res["source"]
    trk = max((r["ms tracker/frame"] for r in res["rows"]), default=0.0)
    per_frame = res["det_ms"] + trk
    rows = [
        ("FPS video asli (sumber)", f"{src['fps']:.2f}", f"{src['width']}x{src['height']}, {src['duration']:.1f} detik"),
        ("FPS proses (tracker dan detektor)", f"{res['proc_fps']:.2f}",
         f"1 dari tiap {round(src['fps'] / res['proc_fps'])} frame sumber diproses"),
        ("FPS video hasil", f"{res['proc_fps']:.2f}", "diputar dengan kecepatan sama seperti video asli (tidak dipercepat)"),
        ("Frame diproses", f"{res['frames']}", f"selesai dalam {res['wall_sec']:.0f} detik (termasuk menggambar)"),
        ("Detektor (median)", f"{res['det_ms']:.1f} ms/frame", "D-FINE, sekali per frame, dipakai bersama semua tracker"),
        ("Tracker paling lambat", f"{trk:.2f} ms/frame", "lihat tabel per tracker"),
        ("FPS maksimum 1 kamera di mesin ini", f"{1000 / per_frame:.1f}" if per_frame else "-",
         "1000 / (detektor + tracker), tanpa decode dan gambar"),
    ]
    return pd.DataFrame(rows, columns=["", "nilai", "keterangan"]).set_index("")


def download_link(path: str | Path, max_mb: float = 80.0):
    """Tombol unduh di output sel (file disisipkan sebagai data URI, jalan di Jupyter dan VS Code).
    Kalau file lebih besar dari max_mb, hanya path yang ditampilkan supaya notebook tidak membengkak."""
    from IPython.display import HTML

    path = Path(path)
    mb = path.stat().st_size / 1e6
    style = ("display:inline-block;padding:8px 16px;background:#2e7d32;color:#fff;border-radius:6px;"
             "text-decoration:none;font-weight:600")
    loc = f"<div style='margin-top:6px'>file: <code>{path}</code> ({mb:.1f} MB)</div>"
    if mb > max_mb:
        return HTML(f"<div>File {mb:.0f} MB terlalu besar untuk tombol unduh di notebook. "
                    f"Ambil langsung dari folder ini (kecilkan PANEL_WIDTH atau durasi untuk file lebih kecil).</div>{loc}")
    b64 = base64.b64encode(path.read_bytes()).decode()
    return HTML(f"<a download='{path.name}' href='data:video/mp4;base64,{b64}' style='{style}'>"
                f"&#11015; Unduh {path.name}</a>{loc}")
