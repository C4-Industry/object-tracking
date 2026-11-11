"""Video perbandingan dari video sendiri: render, keterangan FPS, dan tombol unduh (detektor palsu, tanpa model)."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking import video_compare as vc  # noqa: E402
from object_tracking.boxes import BBox  # noqa: E402
from object_tracking.zoo import LITE_DEFAULT, PRESETS  # noqa: E402


class FakeDetector:
    """Satu orang berjalan dari kiri ke kanan."""

    def __init__(self):
        self.k = 0

    def detect_timed(self, frame, conf_thresh):
        x = 50 + 20 * self.k
        self.k += 1
        return [BBox(x, 100, x + 40, 200, 0.9, 0)], [], [], {}


def test_render_dan_fps(tmp_path):
    src = tmp_path / "cctv.mp4"
    wr = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 25, (640, 360))
    for i in range(25 * 4):
        wr.write(np.full((360, 640, 3), i % 255, np.uint8))
    wr.release()
    specs = {"Lite": LITE_DEFAULT, "ByteTrack": PRESETS["ByteTrack"]}
    r = vc.render(FakeDetector(), src, specs, tmp_path / "out" / "cmp.mp4", fps=4, panel_width=320, progress=False)
    assert r["video"].stat().st_size > 0 and r["snapshot"].exists()
    assert abs(r["proc_fps"] - 25 / 6) < 1e-6                      # 4 FPS diminta -> lompat 6 frame -> 4,17 FPS
    assert [row["tracker"] for row in r["rows"]] == ["Lite", "ByteTrack"]
    assert all(row["ID dibuat"] == 1 for row in r["rows"])         # satu orang, satu ID
    t = vc.fps_table(r)
    assert t.loc["FPS video asli (sumber)", "nilai"] == "25.00"
    assert t.loc["FPS proses (tracker dan detektor)", "nilai"] == "4.17"
    html = vc.download_link(r["video"]).data
    assert "download='cmp.mp4'" in html and "base64," in html
    assert "terlalu besar" in vc.download_link(r["video"], max_mb=0).data
