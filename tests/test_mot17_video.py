"""Video perbandingan MOT17: render jalan, dan hitungan ID switch per frame sama dengan IDSW resmi."""

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_tracking import metrics as mm, mot17, mot17_video as mv  # noqa: E402
from object_tracking.zoo import LITE_DEFAULT, PRESETS  # noqa: E402
from synth_mot import make_fake_mot17  # noqa: E402


def _fake_video(path, n, w=960, h=540):
    wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
    for i in range(n):
        wr.write(np.full((h, w, 3), i % 255, np.uint8))
    wr.release()


def test_idsw_events_sama_dengan_idsw(tmp_path):
    root = make_fake_mot17(tmp_path, length=90, n_ped=12, seed=3)
    seq = mot17.load_seq(root, "MOT17-02", "SDP")
    for spec in (LITE_DEFAULT, PRESETS["ByteTrack"]):
        for fps in (None, 5, 2):
            hyp, _ = mot17.run_tracker(spec, seq, fps)
            g, h = mot17.to_eval_frames(seq, hyp)
            assert sum(len(e) for e in mv.idsw_events(g, h)) == mm.evaluate_sequence(g, h)["IDSW"]


def test_render(tmp_path):
    root = make_fake_mot17(tmp_path, length=60, n_ped=6)
    seq = mot17.load_seq(root, "MOT17-02", "SDP")
    vid = tmp_path / "v.mp4"
    _fake_video(vid, seq.length)
    specs = {"Lite": LITE_DEFAULT, "ByteTrack": PRESETS["ByteTrack"], "OC-SORT": PRESETS["OC-SORT"]}
    r = mv.render_comparison(seq, vid, specs, tmp_path / "out" / "cmp.mp4", target_fps=5, panel_width=320)
    assert r["video"].stat().st_size > 0 and r["snapshot"].exists()
    assert r["size"] == (3 * 320, 2 * (180 + mv.HEADER_H))           # 5 panel -> grid 3 x 2
    for row in r["table"]:
        assert row["IDSW di video"] == row["IDSW"]
    cap = cv2.VideoCapture(str(r["video"]))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == len(range(1, seq.length + 1, 6))
