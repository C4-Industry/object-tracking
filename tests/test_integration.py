"""API integrasi: PersonTracker, CameraTrackers, dan pengganti tracker di ComplianceTracker."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from object_tracking import (LITE_PRESETS, LITE_TUNED_DETECTOR, CameraTrackers,  # noqa: E402
                             LiteForComplianceTracker, PersonTracker, load_lite_config)
from object_tracking.boxes import TrackerBoxes  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def _scene(t):
    """Dua orang berjalan berlawanan arah, plus satu box dobel berskor rendah di atas orang pertama."""
    a = [100 + 40 * t, 200, 160 + 40 * t, 360]
    b = [700 - 40 * t, 210, 760 - 40 * t, 370]
    dup = [a[0] + 3, a[1] + 2, a[2] + 3, a[3] + 2]
    return np.array([a, b, dup], np.float32), np.array([0.9, 0.85, 0.3], np.float32)


def test_parameter_bawaan_sama_dengan_json():
    """configs/tracker_tuned.json (hasil notebook 02 terakhir) harus sama dengan preset di integration.py
    untuk detektor yang dipakai saat tuning. Kalau gagal: salin hasil tuning ke LITE_PRESETS."""
    info = json.loads((ROOT / "configs" / "tracker_tuned.json").read_text()).get("info", {})
    det = Path(info.get("detector", LITE_TUNED_DETECTOR)).stem
    cfg_mod, inp_mod = load_lite_config(detector=det)
    cfg_json, inp_json = load_lite_config(ROOT / "configs" / "tracker_tuned.json")
    assert cfg_mod == cfg_json and inp_mod == inp_json
    assert det == LITE_TUNED_DETECTOR, "tuning terakhir memakai detektor lain dari LITE_TUNED_DETECTOR"
    for d in LITE_PRESETS:
        cfg, _ = load_lite_config(detector=d)
        assert set(LITE_PRESETS[d]["params"]) <= set(cfg.__dataclass_fields__)


@pytest.mark.parametrize("detector", list(LITE_PRESETS))
def test_id_stabil_dan_det_idx_benar(detector):
    trk = PersonTracker.tuned(detector=detector)
    ids = []
    for k in range(12):                      # 4 FPS, 3 detik
        t = k / 4
        boxes, scores = _scene(t)
        out = trk.update(boxes, scores, t)
        if k == 0:
            assert len(out) == 2              # frame pertama langsung tampil
        for o in out:
            assert o.det_idx in (0, 1)        # box dobel (indeks 2) dibuang NMS
            assert np.allclose(o.box, boxes[o.det_idx], atol=25)
        ids.append(sorted(o.id for o in out))
    assert all(i == ids[0] for i in ids)      # ID tidak berganti
    assert trk.ids_shown == 2


@pytest.mark.parametrize("detector", list(LITE_PRESETS))
def test_tertutup_lalu_kembali(detector):
    trk = PersonTracker.tuned(detector=detector)
    first = None
    for k in range(20):
        t = k / 4
        boxes, scores = _scene(t)
        if 6 <= k < 10:                        # orang pertama tertutup 1 detik
            boxes, scores = boxes[1:2], scores[1:2]
        out = trk.update(boxes, scores, t)
        if k == 2:
            first = min(out, key=lambda o: o.box[0]).id
    assert first in {o.id for o in out}       # ID lama kembali setelah tertutup


def test_predict_dan_waktu_mundur():
    trk = PersonTracker.tuned()
    for k in range(3):
        trk.update(*_scene(k / 4), k / 4)
    pred = trk.predict(0.75)
    assert pred and all(p.predicted and p.det_idx == -1 for p in pred)
    with pytest.raises(ValueError):
        trk.update(*_scene(0.0), 0.1)


def test_camera_trackers_terpisah():
    cams = CameraTrackers()
    a = cams.update("cam1", *_scene(0), 0.0)
    b = cams.update("cam2", *_scene(0), 0.0)
    assert [o.id for o in a] == [o.id for o in b]   # ID dihitung per kamera
    assert len(cams.by_camera) == 2


def test_pengganti_tracker_ultralytics():
    eng = LiteForComplianceTracker(fps=4)
    for k in range(4):
        boxes, scores = _scene(k / 4)
        xywh = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
                         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]], 1)
        res = eng.update(TrackerBoxes(xywh, scores, np.zeros(3)), timestamp_sec=k / 4)
        assert res.shape == (2, 8)
        assert set(res[:, 7].astype(int)) == {0, 1}   # kolom idx menunjuk ke deteksi input
    assert eng.update(TrackerBoxes(), timestamp_sec=1.0).shape == (0, 8)
