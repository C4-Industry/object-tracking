"""Live view perbandingan tracker dari SATU sumber video.

Detektor jalan sekali per frame yang diproses, lalu deteksi yang sama dibagikan ke semua tracker. Jadi beda
antar panel murni dari tracker. Tiap panel menampilkan:
  - box orang dengan warna tetap per ID, nomor ID, dan jejak gerak beberapa detik terakhir,
  - box tebakan (frame tanpa deteksi, untuk det/2 dst.) digambar hanya sudutnya,
  - ID baru yang muncul setelah detik awal diberi label KUNING selama 1,5 detik. Kalau orangnya sebenarnya
    sudah ada sebelumnya, itu tanda ID switch / fragmentasi,
  - header: jumlah orang aktif, total ID yang pernah dibuat, dan waktu tracker per frame.

Bisa dipakai tanpa widget (LiveCompare.run, misalnya untuk menyimpan video) atau lewat `live_ui()` di notebook
(pilih/upload video, pilih tracker, Start/Stop, tampil langsung di output sel).
"""

from __future__ import annotations

import colorsys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .video_io import VIDEO_EXT, iter_frames, list_videos, open_writer, video_meta  # noqa: F401
from .zoo import TrackerRunner, TrackerSpec



def id_color(i: int) -> tuple[int, int, int]:
    h = (i * 0.61803398875) % 1.0
    r, g, b = colorsys.hsv_to_rgb(h, 0.85, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _text(img, s, x, y, scale=0.5, fg=(255, 255, 255), bg=(0, 0, 0), thick=1):
    (tw, th), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    y0 = max(0, y - th - 6)
    cv2.rectangle(img, (x, y0), (x + tw + 6, y0 + th + 6 + base // 2), bg, -1)
    cv2.putText(img, s, (x + 3, y0 + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale, fg, thick, cv2.LINE_AA)


def _corners(img, x1, y1, x2, y2, col, t):
    L = max(6, int(min(x2 - x1, y2 - y1) * 0.25))
    for (ax, ay, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(img, (ax, ay), (ax + dx * L, ay), col, t)
        cv2.line(img, (ax, ay), (ax, ay + dy * L), col, t)


@dataclass
class PanelState:
    spec: TrackerSpec
    runner: TrackerRunner
    trails: dict = field(default_factory=dict)       # id -> list (t, x, y) titik kaki
    first_seen: dict = field(default_factory=dict)   # id -> t pertama muncul
    ms: list = field(default_factory=list)
    active: list = field(default_factory=list)
    new_mid: int = 0                                 # ID baru yang muncul setelah grace period


class LiveCompare:
    def __init__(self, detector, specs: list[TrackerSpec], fps: float = 4.0, panel_width: int = 640,
                 trail_sec: float = 3.0, show_raw: bool = True, grace_sec: float = 2.0, det_conf: float = 0.05,
                 show_original: bool = False):
        self.detector = detector
        self.specs = list(specs)
        self.fps = float(fps)
        self.panel_width = int(panel_width)
        self.trail_sec = trail_sec
        self.show_raw = show_raw
        self.grace_sec = grace_sec
        self.det_conf = det_conf
        self.show_original = show_original
        self.source: dict = {}           # info video sumber dari run() terakhir: fps, frame, ukuran, durasi
        self.proc_fps = self.fps         # FPS proses sebenarnya = fps sumber / lompatan frame (bulat)
        self.decode_ms: list = []
        self.snapshot = None             # grid di tengah video (untuk gambar cuplikan)
        self.det_cache: dict = {}        # (path, t) -> Nx5 orang, supaya replay dengan tracker lain instan
        self.det_ms: list = []
        self.panels: list[PanelState] = []

    # ---------- deteksi ----------
    def _detect(self, path: str, t: float, frame: np.ndarray) -> np.ndarray:
        key = (path, round(t, 3))
        if key in self.det_cache:
            return self.det_cache[key]
        t0 = time.perf_counter()
        persons, _, _, _ = self.detector.detect_timed(frame, conf_thresh=self.det_conf)
        self.det_ms.append((time.perf_counter() - t0) * 1000)
        p = np.array([[b.x1, b.y1, b.x2, b.y2, b.conf] for b in persons], dtype=np.float32).reshape(-1, 5)
        self.det_cache[key] = p
        return p

    # ---------- gambar ----------
    def _draw_raw(self, small, persons, s):
        img = small.copy()
        persons = persons[persons[:, 4] >= 0.1]   # di bawah 0,1 tidak dipakai tracker mana pun
        for x1, y1, x2, y2, c in persons:
            col = (0, 200, 0) if c >= 0.5 else (0, 165, 255) if c >= 0.2 else (140, 140, 140)
            p1, p2 = (int(x1 * s), int(y1 * s)), (int(x2 * s), int(y2 * s))
            cv2.rectangle(img, p1, p2, col, 1)
            _text(img, f"{c:.2f}", p1[0], p1[1], 0.4, (0, 0, 0), col)
        hi = int((persons[:, 4] >= 0.5).sum()) if len(persons) else 0
        return self._header(img, "Deteksi mentah (input semua tracker)",
                            f"skor >=0,5: {hi} | skor 0,1-0,5: {len(persons) - hi}  (hijau >=0,5, oranye >=0,2)")

    def _header(self, img, title, sub):
        """Strip header di ATAS gambar (tidak menutupi orang di tepi atas frame). Mengembalikan gambar baru."""
        bar = np.full((40, img.shape[1], 3), 25, np.uint8)
        cv2.putText(bar, title, (8, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, sub, (8, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
        return np.vstack([bar, img])

    def _draw_panel(self, small, ps: PanelState, tracks, t, t0, s):
        img = small.copy()
        th = 2
        live_ids = set()
        for tr in tracks:
            live_ids.add(tr.id)
            x1, y1, x2, y2 = (int(v * s) for v in tr.box)
            col = id_color(tr.id)
            fresh = ps.first_seen.get(tr.id, t) > t0 + self.grace_sec and (t - ps.first_seen.get(tr.id, t)) < 1.5
            if tr.predicted:
                _corners(img, x1, y1, x2, y2, col, th)
            else:
                cv2.rectangle(img, (x1, y1), (x2, y2), col, th)
            pts = ps.trails.get(tr.id, [])
            if len(pts) > 1:
                cv2.polylines(img, [np.array([(int(x * s), int(y * s)) for _, x, y in pts], np.int32)], False, col, 2)
            _text(img, f"#{tr.id}" + ("*" if tr.predicted else ""), x1, y1, 0.5,
                  (0, 0, 0), (0, 255, 255) if fresh else col, 1)
        ms = np.mean(ps.ms[-20:]) if ps.ms else 0.0
        extra = f" | det/{ps.spec.det_every}" if ps.spec.det_every > 1 else ""
        return self._header(img, ps.spec.name, f"aktif {len(live_ids)} | ID dibuat {len(ps.first_seen)} | "
                                               f"ID baru di tengah {ps.new_mid} | {ms:.2f} ms{extra}")

    @staticmethod
    def _grid(tiles, cols):
        h, w = tiles[0].shape[:2]
        rows = (len(tiles) + cols - 1) // cols
        out = np.full((rows * h, cols * w, 3), 255, np.uint8)
        for i, tl in enumerate(tiles):
            r, c = divmod(i, cols)
            out[r * h:(r + 1) * h, c * w:(c + 1) * w] = tl
            cv2.rectangle(out, (c * w, r * h), ((c + 1) * w - 1, (r + 1) * h - 1), (255, 255, 255), 2)
        return out

    # ---------- jalan ----------
    def run(self, source: str | Path, start_sec: float = 0.0, max_sec: float | None = 60.0, show=None,
            realtime: bool = True, speed: float = 1.0, save_path: str | Path | None = None,
            stop_event: threading.Event | None = None, status=None) -> dict:
        """show: fungsi(jpeg_bytes) dipanggil tiap frame (mis. untuk widgets.Image). status: fungsi(str).
        Mengembalikan ringkasan per tracker."""
        path = str(source)
        self.source = video_meta(path)
        src_fps = self.source["fps"]
        self.proc_fps = src_fps / max(1, int(round(src_fps / self.fps)))   # sama dengan lompatan di iter_frames
        self.decode_ms = []
        self.panels = [PanelState(sp, TrackerRunner(sp, self.proc_fps)) for sp in self.specs]
        n_tiles = len(self.specs) + int(self.show_raw) + int(self.show_original)
        cols = 2 if n_tiles <= 4 else 3
        foot = f"sumber {src_fps:.1f} FPS | proses {self.proc_fps:.2f} FPS"
        span = self.source["duration"] - start_sec
        if max_sec:
            span = min(span, max_sec) if span > 0 else max_sec
        mid_t = start_sec + max(0.0, span) / 2
        self.snapshot = None
        writer, wall0, t0, n = None, None, None, 0
        for t, frame, dec_ms in iter_frames(path, self.fps, max_sec, start_sec):
            self.decode_ms.append(dec_ms)
            if stop_event is not None and stop_event.is_set():
                break
            if t0 is None:
                t0, wall0 = t, time.perf_counter()
            persons = None
            if self.show_raw or any(ps.runner.uses_detection() for ps in self.panels):
                persons = self._detect(path, t, frame)
            s = self.panel_width / frame.shape[1]
            small = cv2.resize(frame, (self.panel_width, int(round(frame.shape[0] * s))), interpolation=cv2.INTER_AREA)
            tiles = []
            if self.show_original:
                tiles.append(self._header(small.copy(), "Video asli",
                                          f"{frame.shape[1]}x{frame.shape[0]} | {src_fps:.1f} FPS asli | "
                                          f"diproses {self.proc_fps:.2f} FPS"))
            if self.show_raw:
                tiles.append(self._draw_raw(small, persons, s))
            for ps in self.panels:
                t1 = time.perf_counter()
                tracks = ps.runner.step(t, persons if ps.runner.uses_detection() else None, frame)
                ps.ms.append((time.perf_counter() - t1) * 1000)
                ps.active.append(len(tracks))
                for tr in tracks:
                    if tr.id not in ps.first_seen:
                        ps.first_seen[tr.id] = t
                        if t > t0 + self.grace_sec:
                            ps.new_mid += 1
                    tl = ps.trails.setdefault(tr.id, [])
                    tl.append((t, float((tr.box[0] + tr.box[2]) / 2), float(tr.box[3])))
                    while tl and t - tl[0][0] > self.trail_sec:
                        tl.pop(0)
                tiles.append(self._draw_panel(small, ps, tracks, t, t0, s))
            grid = self._grid(tiles, cols)
            _text(grid, f"t = {t:6.1f} s | {foot}", 6, grid.shape[0] - 6, 0.5)
            if save_path is not None:
                if writer is None:
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    writer = open_writer(save_path, self.proc_fps, grid.shape[1::-1])
                writer.write(grid)
            if self.snapshot is None or t <= mid_t:      # cuplikan = frame di tengah durasi yang diproses
                self.snapshot = grid
            if show is not None:
                show(cv2.imencode(".jpg", grid, [cv2.IMWRITE_JPEG_QUALITY, 80])[1].tobytes())
            n += 1
            if status is not None and n % max(1, int(self.fps)) == 0:
                dm = np.mean(self.det_ms[-20:]) if self.det_ms else 0.0
                status(f"t = {t:.1f} s | frame {n} | detektor {dm:.0f} ms | cache deteksi {len(self.det_cache)} frame")
            if realtime and wall0 is not None:
                wait = (t - t0) / max(speed, 1e-3) - (time.perf_counter() - wall0)
                if wait > 0:
                    time.sleep(wait)
        if writer is not None:
            writer.release()
        return self.summary(n)

    def summary(self, frames: int | None = None) -> dict:
        rows = []
        for ps in self.panels:
            rows.append({"tracker": ps.spec.name, "ID dibuat": len(ps.first_seen), "ID baru di tengah": ps.new_mid,
                         "orang aktif (rata-rata)": round(float(np.mean(ps.active)), 2) if ps.active else 0.0,
                         "ms tracker/frame": round(float(np.mean(ps.ms)), 3) if ps.ms else 0.0,
                         "deteksi/detik": round(self.proc_fps / max(1, ps.spec.det_every), 2)})
        return {"frames": frames, "rows": rows,
                "det_ms_median": float(np.median(self.det_ms)) if self.det_ms else None}


# ---------------- UI notebook ----------------

def live_ui(detector, specs: dict[str, TrackerSpec], video_dirs, upload_dir, default_specs=None,
            default_video: str | Path | None = None, out_dir: str | Path | None = None):
    """Widget: pilih/upload video, pilih tracker, Start/Stop. Proses jalan di thread terpisah, jadi tombol Stop
    tetap bisa ditekan dan sel lain tetap bisa dijalankan."""
    import ipywidgets as W
    import pandas as pd
    from IPython.display import display

    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dirs = [upload_dir] + [Path(d) for d in video_dirs]

    def options():
        return [(f"{p.name}  ({p.parent.name})", str(p)) for p in list_videos(dirs)]

    opts = options()
    vid = W.Dropdown(options=opts, description="Video", layout=W.Layout(width="520px"),
                     value=str(default_video) if default_video and any(v == str(default_video) for _, v in opts)
                     else (opts[0][1] if opts else None))
    path_txt = W.Text(placeholder="atau tempel path video di sini", description="Path", layout=W.Layout(width="520px"))
    up = W.FileUpload(accept="video/*,.mp4,.mov,.avi,.mkv", multiple=False, description="Upload video")
    refresh = W.Button(description="Muat ulang daftar", icon="refresh")
    names = list(specs)
    pick = W.SelectMultiple(options=names, value=tuple(n for n in (default_specs or names[:3]) if n in specs),
                            rows=min(10, len(names)), description="Tracker", layout=W.Layout(width="520px"))
    fps = W.FloatSlider(value=4.0, min=1.0, max=12.5, step=0.5, description="FPS proses")
    start = W.IntText(value=0, description="Mulai (s)")
    dur = W.IntText(value=60, description="Durasi (s)")
    width = W.IntSlider(value=560, min=320, max=960, step=40, description="Lebar panel")
    raw = W.Checkbox(value=True, description="Panel deteksi mentah")
    rt = W.Checkbox(value=True, description="Kecepatan asli (real-time)")
    save = W.Checkbox(value=False, description="Simpan video perbandingan")
    go = W.Button(description="Start", button_style="success", icon="play")
    stop = W.Button(description="Stop", button_style="danger", icon="stop", disabled=True)
    img = W.Image(format="jpeg", layout=W.Layout(max_width="100%"))
    status = W.HTML()
    table = W.Output()
    state = {"thread": None, "stop": threading.Event(), "lc": None}

    def on_upload(change):
        files = up.value
        if not files:
            return
        f = files[0]
        dst = upload_dir / f["name"]
        dst.write_bytes(bytes(f["content"]))
        vid.options = options()
        vid.value = str(dst)
        status.value = f"tersimpan: <code>{dst}</code>"

    def on_refresh(_):
        cur = vid.value
        vid.options = options()
        if cur in [v for _, v in vid.options]:
            vid.value = cur

    def worker(src, chosen, save_path):
        lc = state["lc"]
        if lc is None or lc.fps != fps.value or lc.panel_width != width.value:
            lc = LiveCompare(detector, chosen, fps=fps.value, panel_width=width.value, show_raw=raw.value)
            if state["lc"] is not None and state["lc"].fps == fps.value:
                lc.det_cache = state["lc"].det_cache
        lc.specs, lc.show_raw = chosen, raw.value
        state["lc"] = lc
        try:
            res = lc.run(src, start_sec=start.value, max_sec=dur.value or None,
                         show=lambda b: setattr(img, "value", b), realtime=rt.value, save_path=save_path,
                         stop_event=state["stop"], status=lambda s: setattr(status, "value", s))
            table.clear_output()
            with table:
                display(pd.DataFrame(res["rows"]))
            status.value = "selesai" + (f" | video: <code>{save_path}</code>" if save_path else "")
        except Exception as e:
            status.value = f"<b style='color:red'>error: {e!r}</b>"
        finally:
            go.disabled, stop.disabled = False, True

    def on_go(_):
        src = path_txt.value.strip() or vid.value
        if not src or not Path(src).exists():
            status.value = "<b style='color:red'>video tidak ditemukan</b>"
            return
        chosen = [specs[n] for n in pick.value]
        if not chosen:
            status.value = "<b style='color:red'>pilih minimal satu tracker</b>"
            return
        save_path = None
        if save.value and out_dir is not None:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            save_path = Path(out_dir) / f"live_{Path(src).stem}_{fps.value:g}fps_{time.strftime('%H%M%S')}.mp4"
        state["stop"] = threading.Event()
        go.disabled, stop.disabled = True, False
        state["thread"] = threading.Thread(target=worker, args=(src, chosen, save_path), daemon=True)
        state["thread"].start()

    def on_stop(_):
        state["stop"].set()

    up.observe(on_upload, names="value")
    refresh.on_click(on_refresh)
    go.on_click(on_go)
    stop.on_click(on_stop)
    ui = W.VBox([
        W.HBox([vid, refresh]), W.HBox([path_txt, up]), pick,
        W.HBox([fps, width]), W.HBox([start, dur]), W.HBox([raw, rt, save]),
        W.HBox([go, stop]), status, img, table,
    ])
    ui.state = state  # untuk akses dari sel lain (mis. state["lc"].summary())
    return ui
