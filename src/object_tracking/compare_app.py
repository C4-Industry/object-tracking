"""Aplikasi Gradio: bandingkan beberapa tracker pada satu video CCTV, tampil langsung (live).

Jalankan:  uv run tracker-compare                 (buka http://127.0.0.1:7861)
      atau  uv run python -m object_tracking.compare_app
      atau  dari notebooks/04_bandingkan_tracker_video.ipynb

Alur: pilih video (daftar video PT SRP atau upload), pilih tracker, tekan Proses. Grid video tampil langsung
sambil diproses, lengkap dengan FPS yang sedang berjalan. Setelah selesai (atau dihentikan), video hasil bisa
diunduh. Detektor D-FINE jalan sekali per frame yang diproses, lalu deteksi yang sama dibagikan ke semua tracker,
jadi beda antar panel murni dari tracker.
"""

from __future__ import annotations

import argparse
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from . import ROOT
from .live_compare import VIDEO_EXT, LiveCompare, list_videos, video_meta
from .zoo import all_presets

VIDEO_DIRS = [Path.home() / "PPE - PT SRP" / "Dataset_Momen_PPE_Anonim", ROOT / "data" / "videos"]
UPLOAD_DIR = ROOT / "data" / "videos"
OUT_DIR = ROOT / "outputs" / "compare"
DET_MODEL = ROOT / "models" / "dfine-n-960x540-stride-8.onnx"
DEFAULT_TRACKERS = ["Lite tuned", "ByteTrack tuned", "ByteTrack (sekarang)"]
UI_INTERVAL_SEC = 0.12          # gambar ke UI paling sering sekitar 8 kali per detik

LEGEND = """
**Cara membaca panel**
- **Video asli**: frame sumber. **Deteksi mentah**: box dari D-FINE yang jadi input semua tracker (hijau skor 0,5 ke atas, oranye 0,2 sampai 0,5).
- **Panel tracker**: warna dan nomor = ID. Orang yang sama seharusnya memegang warna yang sama sepanjang video. Garis = jejak gerak 3 detik terakhir.
- **Label kuning** = ID baru yang muncul di tengah video. Kalau orangnya sudah terlihat sebelumnya, itu ID switch.
- **Box hanya sudut** = posisi tebakan tracker di frame tanpa deteksi.
"""


class _State:
    """State aplikasi (satu pengguna lokal): detektor, video, dan proses yang sedang berjalan."""

    def __init__(self, video_dirs, det_model, detector=None):
        self.video_dirs = [Path(d) for d in video_dirs]
        self.det_model = Path(det_model)
        self.detector = detector
        self.det_info = "" if detector is None else "detektor dari pemanggil"
        self.stop = threading.Event()
        self.lock = threading.Lock()

    def get_detector(self):
        with self.lock:
            if self.detector is None:
                from .detector import OnnxDFineDetector

                if not self.det_model.exists():
                    raise gr.Error(f"Model detektor tidak ada: {self.det_model}")
                try:
                    self.detector = OnnxDFineDetector(self.det_model, runtime="coreml", resize="area")
                except Exception:
                    self.detector = OnnxDFineDetector(self.det_model, runtime="cpu", resize="area")
                self.det_info = f"{self.det_model.name}, {self.detector.active_providers[0]}"
            return self.detector


def _video_choices(state: _State) -> list[tuple[str, str]]:
    out = []
    for p in list_videos(state.video_dirs):
        out.append((f"{p.name}   ({p.parent.name}, {p.stat().st_size / 1e6:.0f} MB)", str(p)))
    return out


def _video_info(path: str | None) -> str:
    if not path:
        return "Belum ada video. Taruh video di salah satu folder daftar, atau upload."
    try:
        m = video_meta(path)
    except Exception as e:
        return f"Video tidak bisa dibuka: {e}"
    return f"{m['width']}x{m['height']}  |  {m['fps']:.1f} FPS  |  {m['duration']:.1f} detik  |  {m['frames']} frame"


def _fmt(x: float, nd: int = 1) -> str:
    return f"{x:,.{nd}f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _live_stats(lc: LiveCompare, n: int, t_video: float, span: float, wall: float, recent: deque, det_info: str,
                done: bool = False, stopped: bool = False) -> str:
    now_fps = 0.0
    if len(recent) >= 2 and recent[-1] > recent[0]:
        now_fps = (len(recent) - 1) / (recent[-1] - recent[0])
    avg_fps = (n - 1) / wall if wall > 0 and n > 1 else 0.0     # wall = waktu frame pertama sampai terakhir
    proc = lc.proc_fps or 1.0
    det_ms = float(np.mean(lc.det_ms[-20:])) if lc.det_ms else 0.0
    trk_ms = max((float(np.mean(ps.ms[-20:])) for ps in lc.panels if ps.ms), default=0.0)
    left = max(0.0, span - t_video)
    eta = left * proc / now_fps if now_fps > 0 else 0.0
    if done:
        head = "**Dihentikan.**" if stopped else "**Selesai.**"
        speed = f"Rata-rata **{_fmt(avg_fps)} frame/detik** ({_fmt(avg_fps / proc, 2)}x kecepatan video)"
    else:
        speed = (f"Sekarang **{_fmt(now_fps)} frame/detik** ({_fmt(now_fps / proc, 2)}x kecepatan video)"
                 if now_fps > 0 else "menghitung kecepatan...")
        head = "**Sedang diproses...**"
    pct = min(100.0, 100 * t_video / span) if span > 0 else 0.0
    lines = [
        f"{head} {speed}",
        f"- FPS proses: {_fmt(proc, 2)} (video sumber {_fmt(lc.source.get('fps', 0), 1)} FPS)",
        f"- Progres: {_fmt(t_video)} dari {_fmt(span)} detik video ({_fmt(pct, 0)}%), frame ke-{n}"
        + ("" if done or eta <= 0 else f", sisa sekitar {_fmt(eta, 0)} detik"),
        f"- Detektor: {_fmt(det_ms)} ms/frame  |  tracker paling lambat: {_fmt(trk_ms, 2)} ms/frame",
        f"- Batas 1 kamera di mesin ini: {_fmt(1000 / (det_ms + trk_ms), 1) if det_ms + trk_ms > 0 else '-'} FPS "
        "(detektor + tracker saja)",
    ]
    if det_info:
        lines.append(f"- Model: {det_info}")
    return "\n".join(lines)


def _table(lc: LiveCompare, n: int) -> pd.DataFrame:
    rows = lc.summary(n)["rows"]
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows).drop(columns=["deteksi/detik"], errors="ignore")
    return df.sort_values(["ID baru di tengah", "ID dibuat"]).reset_index(drop=True)


def build_app(video_dirs=VIDEO_DIRS, det_model=DET_MODEL, detector=None) -> gr.Blocks:
    state = _State(video_dirs, det_model, detector)
    presets = all_presets()
    names = list(presets)
    default_trk = [n for n in DEFAULT_TRACKERS if n in presets] or names[:3]

    with gr.Blocks(title="Bandingkan Tracker") as demo:
        gr.Markdown("# Bandingkan Tracker\nPilih video CCTV, pilih tracker, lalu tekan **Proses**. "
                    "Hasil tampil langsung dan bisa diunduh setelah selesai.")
        with gr.Row():
            with gr.Column(scale=3):
                choices = _video_choices(state)
                video = gr.Dropdown(choices, value=choices[0][1] if choices else None, label="Video",
                                    info="Video PT SRP dan video yang sudah di-upload")
                info = gr.Markdown(_video_info(choices[0][1] if choices else None))
                with gr.Accordion("Upload video lain", open=False):
                    upload = gr.File(label="Video (mp4, mov, avi, mkv)", file_types=[".mp4", ".mov", ".avi", ".mkv", ".m4v", ".webm"])
                    gr.Markdown(f"Disimpan ke `{UPLOAD_DIR.relative_to(ROOT)}/` lalu otomatis terpilih.")
            with gr.Column(scale=2):
                trackers = gr.CheckboxGroup(names, value=default_trk, label="Tracker yang dibandingkan")
                with gr.Row():
                    fps = gr.Slider(1, 12, value=5, step=1, label="FPS proses", info="CCTV kita: 2 sampai 6")
                    dur = gr.Number(60, label="Durasi (detik)", info="0 = sampai video habis", minimum=0)
                with gr.Accordion("Pengaturan lanjutan", open=False):
                    start = gr.Number(0, label="Mulai dari detik ke", minimum=0)
                    panel_w = gr.Radio([480, 640, 800], value=640, label="Lebar tiap panel (piksel)")
                    show_raw = gr.Checkbox(True, label="Tampilkan panel deteksi mentah")
                    realtime = gr.Checkbox(False, label="Batasi ke kecepatan video asli",
                                           info="Mati = diproses secepat mungkin, jadi FPS yang tampil adalah kemampuan mesin ini")
        with gr.Row():
            go = gr.Button("Proses", variant="primary", scale=3)
            stop = gr.Button("Hentikan", variant="stop", scale=1)
            download = gr.DownloadButton("Unduh video hasil", visible=False, scale=2)
        stats = gr.Markdown("Belum ada proses.")
        live = gr.Image(label="Perbandingan live", type="numpy", format="jpeg", interactive=False)
        table = gr.Dataframe(label="Ringkasan per tracker (ID baru di tengah: makin kecil makin baik)", interactive=False)
        gr.Markdown(LEGEND)

        def on_video(path):
            return _video_info(path)

        def on_upload(f):
            if not f:
                return gr.update(), gr.update()
            src = Path(f if isinstance(f, str) else f.name)
            if src.suffix.lower() not in VIDEO_EXT:
                raise gr.Error(f"Format {src.suffix} tidak didukung")
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            dst = UPLOAD_DIR / src.name
            shutil.copy(src, dst)
            if UPLOAD_DIR not in state.video_dirs:
                state.video_dirs.append(UPLOAD_DIR)
            return gr.update(choices=_video_choices(state), value=str(dst)), _video_info(str(dst))

        def on_stop():
            state.stop.set()
            return "Menghentikan... video hasil sampai titik ini tetap bisa diunduh."

        def process(path, chosen, fps_v, dur_v, start_v, pw, raw, rt):
            if not path or not Path(path).exists():
                raise gr.Error("Pilih video dulu.")
            if not chosen:
                raise gr.Error("Pilih minimal satu tracker.")
            yield (gr.update(), "Memuat detektor...", gr.update(), gr.DownloadButton(visible=False))
            det = state.get_detector()
            specs = [presets[n].with_params(name=n) for n in chosen]
            lc = LiveCompare(det, specs, fps=float(fps_v), panel_width=int(pw), show_raw=bool(raw), show_original=True)
            meta = video_meta(path)
            start_s = float(start_v or 0)
            span = max(0.0, meta["duration"] - start_s)
            if dur_v:
                span = min(span, float(dur_v)) if span > 0 else float(dur_v)
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            out = OUT_DIR / f"{Path(path).stem}_{float(fps_v):g}fps_{datetime.now():%Y%m%d_%H%M%S}.mp4"

            state.stop.clear()
            box = {"jpg": None, "n": 0, "fresh": False, "err": None, "done": False, "t_first": None}
            recent: deque = deque(maxlen=30)

            def show(jpg):
                if box["t_first"] is None:
                    box["t_first"] = time.perf_counter()   # waktu muat (impor tracker, model) tidak ikut dihitung
                box["jpg"] = jpg
                box["n"] += 1
                box["fresh"] = True
                recent.append(time.perf_counter())

            def work():
                try:
                    lc.run(path, start_sec=start_s, max_sec=float(dur_v) if dur_v else None, show=show,
                           realtime=bool(rt), save_path=out, stop_event=state.stop)
                except Exception as e:          # ditampilkan di UI
                    box["err"] = e
                finally:
                    box["done"] = True

            th = threading.Thread(target=work, daemon=True)
            th.start()
            last_tab = 0.0
            while not box["done"]:
                time.sleep(UI_INTERVAL_SEC)
                if not box["fresh"]:
                    continue
                box["fresh"] = False
                img = cv2.cvtColor(cv2.imdecode(np.frombuffer(box["jpg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
                n = box["n"]
                t_video = (n - 1) / (lc.proc_fps or 1.0)
                s = _live_stats(lc, n, t_video, span, time.perf_counter() - box["t_first"], recent, state.det_info)
                tab = gr.update()
                if time.perf_counter() - last_tab > 1.0:
                    tab, last_tab = _table(lc, n), time.perf_counter()
                yield img, s, tab, gr.update()
            th.join()
            if box["err"] is not None:
                raise gr.Error(f"Proses gagal: {box['err']}")
            n = box["n"]
            img = gr.update()
            if box["jpg"] is not None:
                img = cv2.cvtColor(cv2.imdecode(np.frombuffer(box["jpg"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            wall = (recent[-1] - box["t_first"]) if box["t_first"] is not None and recent else 0.0
            stopped = state.stop.is_set()
            t_end = (n - 1) / (lc.proc_fps or 1.0) if stopped else span
            s = _live_stats(lc, n, t_end, span, wall, recent,
                            state.det_info, done=True, stopped=stopped)
            dl = gr.DownloadButton(visible=False)
            if out.exists() and out.stat().st_size > 0:
                s += f"\n- Video hasil: `{out.relative_to(ROOT)}` ({_fmt(out.stat().st_size / 1e6)} MB)"
                dl = gr.DownloadButton(f"Unduh video hasil ({_fmt(out.stat().st_size / 1e6)} MB)", value=str(out), visible=True)
            yield img, s, _table(lc, n), dl

        video.change(on_video, video, info)
        upload.upload(on_upload, upload, [video, info])
        go.click(process, [video, trackers, fps, dur, start, panel_w, show_raw, realtime], [live, stats, table, download],
                 concurrency_limit=1)
        stop.click(on_stop, None, stats)

    demo.compare_state = state
    return demo


def launch(demo: gr.Blocks | None = None, port: int = 7861, host: str = "127.0.0.1", inbrowser: bool = True,
           block: bool = True, **kw):
    demo = demo or build_app()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    demo.queue().launch(server_name=host, server_port=port, inbrowser=inbrowser, prevent_thread_lock=not block,
                        allowed_paths=[str(OUT_DIR)], **kw)
    return demo


def main() -> None:
    ap = argparse.ArgumentParser(description="Bandingkan tracker pada satu video, live")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=7861)
    ap.add_argument("--video-dir", action="append", help="folder video tambahan (boleh diulang)")
    ap.add_argument("--model", default=str(DET_MODEL), help="model detektor ONNX")
    args = ap.parse_args()
    dirs = VIDEO_DIRS + [Path(d).expanduser() for d in (args.video_dir or [])]
    launch(build_app(dirs, args.model), port=args.port, host=args.host)


if __name__ == "__main__":
    main()
