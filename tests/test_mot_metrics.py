"""Uji cepat metrik MOT dan loader MOT17. Jalankan: uv run python -m pytest tests -q

Angka di metrics.py sudah dicocokkan dengan TrackEval resmi (selisih 0 untuk HOTA, IDF1, MOTA, ID switch)
saat modul dibuat. Tes di sini menjaga perilaku dasarnya tanpa perlu memasang TrackEval.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_tracking import metrics as mm, mot17  # noqa: E402
from synth_mot import make_fake_mot17  # noqa: E402


def _frames(ids_boxes):
    return [(np.array(i, int), np.array(b, float).reshape(-1, 4)) for i, b in ids_boxes]


def test_sempurna():
    gt = _frames([([1, 2], [[0, 0, 10, 10], [50, 50, 10, 10]])] * 5)
    m = mm.summarize(mm.evaluate_sequence(gt, gt))
    assert m["IDF1"] == 1 and m["MOTA"] == 1 and m["IDSW"] == 0
    assert abs(m["HOTA"] - 1) < 1e-9


def test_id_switch():
    gt = _frames([([1], [[0, 0, 10, 10]])] * 4)
    hyp = _frames([([7], [[0, 0, 10, 10]])] * 2 + [([8], [[0, 0, 10, 10]])] * 2)
    m = mm.summarize(mm.evaluate_sequence(gt, hyp))
    assert m["IDSW"] == 1
    assert abs(m["IDF1"] - 0.5) < 1e-9          # hanya separuh frame memegang ID yang terpilih
    assert abs(m["MOTA"] - 0.75) < 1e-9         # (4 TP - 0 FP - 1 IDSW) / 4


def test_tracker_kosong():
    gt = _frames([([1], [[0, 0, 10, 10]])] * 3)
    hyp = _frames([([], [])] * 3)
    m = mm.summarize(mm.evaluate_sequence(gt, hyp))
    assert m["IDF1"] == 0 and m["MOTA"] == 0 and m["HOTA"] == 0


def test_mot17_palsu(tmp_path):
    root = make_fake_mot17(tmp_path, "MOT17-02", "SDP", length=60)
    seq = mot17.load_seq(root, "MOT17-02", "SDP")
    assert seq.fps == 30 and seq.length == 60 and seq.static
    spec = mot17.TrackerSpec("Lite", "lite", {})
    rows = mot17.run_benchmark({"Lite": spec}, [seq], fps_list=(None, 5))
    tab = mot17.summary(rows, by=("tracker", "target_fps"), sort=None)
    assert len(tab) == 2
    assert tab.loc[("Lite", "asli"), "IDF1"] > 0.7
