"""Loader ground truth CVAT dan evaluasi tracker terhadap label manusia."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from object_tracking import ground_truth as gtm, mot17  # noqa: E402
from object_tracking.zoo import LITE_DEFAULT  # noqa: E402
from synth_cvat import FakeDetector, make_cvat_clip  # noqa: E402


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    root = tmp_path_factory.mktemp("gt")
    make_cvat_clip(root, "cam01_xml", duration=12, fmt="xml")
    make_cvat_clip(root, "cam02_mot", duration=6, fmt="mot")
    return root, gtm.find_clips(root)


def test_baca_xml_dan_mot(clips):
    _, cl = clips
    xml, mot = [c for c in cl if c.name == "cam01_xml"][0], [c for c in cl if c.name == "cam02_mot"][0]
    assert xml.source == "cvat_xml" and mot.source == "mot_txt"
    assert set(xml.ids) == {0, 1, 2, 3, 4}                 # forklift diabaikan, 5 orang
    assert xml.boxes.groupby("id").frame.diff().dropna().max() == 1   # keyframe tiap 5 frame sudah diinterpolasi
    assert xml.boxes.occluded.any()
    assert any("forklift" in n for n in xml.notes)
    assert xml.video is not None and mot.video is not None


def test_cek_label(clips):
    _, cl = clips
    info = gtm.check(cl[0])
    assert info["orang (ID)"] == 5 and "frame video" in info


def test_oracle_hampir_sempurna(clips):
    """Dengan box label sebagai deteksi, tracker harus hampir sempurna (sisa kesalahan = tracker)."""
    _, cl = clips
    c = [c for c in cl if c.name == "cam01_xml"][0]
    seq = gtm.to_seq(c, gtm.oracle_dets(c), "label", occluded="nilai")
    rows = mot17.run_benchmark({"Lite": LITE_DEFAULT}, [seq], fps_list=(None, 5))
    tab = mot17.summary(rows, by=("target_fps",), sort=None)
    assert (tab["IDF1"] > 0.9).all()


def test_detektor_palsu_dan_abaikan_tertutup(clips, tmp_path):
    _, cl = clips
    c = [c for c in cl if c.name == "cam01_xml"][0]
    dets = gtm.detect_clip(FakeDetector(), c, cache_path=tmp_path / "d.npz")
    again = gtm.detect_clip(None, c, cache_path=tmp_path / "d.npz")          # dari cache
    assert len(dets) == len(again) == 300
    nilai = mot17.summary(mot17.run_benchmark({"Lite": LITE_DEFAULT}, [gtm.to_seq(c, dets, "palsu", "nilai")], (5,)))
    abai = mot17.summary(mot17.run_benchmark({"Lite": LITE_DEFAULT}, [gtm.to_seq(c, dets, "palsu", "abaikan")], (5,)))
    assert abai["IDF1"].iloc[0] >= nilai["IDF1"].iloc[0]   # orang tertutup tidak dihitung sebagai terlewat
    assert abai["IDF1"].iloc[0] > 0.7


def test_format_frozen_dan_output_pipeline(tmp_path):
    from synth_cvat import make_frozen_clip
    root = make_frozen_clip(tmp_path, ignore=[{"from_s": 0, "to_s": 12, "bbox": [0, 0, 300, 720], "reason": "tes"}])
    c = gtm.find_clips(root / "frozen", [root])[0]
    assert c.source == "frozen" and c.video is not None and c.fps == 15
    assert c.ignore_regions and c.ignore_regions[0][1] >= 179
    seq = gtm.to_seq(c, gtm.oracle_dets(c), "label")
    assert seq.ignore_regions and seq.eval_frames is None
    raw = gtm.load_run(root / "runs" / "palsu", c.name, 15, shift=0)
    shift, score = gtm.estimate_shift(c, raw)
    assert shift == 2 and score[2] > score[0]
    good = mot17.mm.summarize(gtm.eval_run(seq, root / "runs" / "palsu", 15, shift)["raw"])
    bad = mot17.mm.summarize(gtm.eval_run(seq, root / "runs" / "palsu", 15, 0)["raw"])
    assert good["IDF1"] > 0.9 and good["IDF1"] >= bad["IDF1"]


def test_label_sebagian_frame(tmp_path):
    from synth_cvat import make_frozen_clip
    root = make_frozen_clip(tmp_path, every=5)
    c = gtm.find_clips(root / "frozen", [root])[0]
    assert c.labelled_frames is not None and len(c.labelled_frames) == 36
    seq = gtm.to_seq(c, gtm.oracle_dets(c), "label")
    rows = mot17.run_benchmark({"Lite": LITE_DEFAULT}, [seq], fps_list=(None, 3))
    tab = mot17.summary(rows, by=("target_fps",), sort=None)
    assert (tab["FP"] == 0).all()              # frame tanpa label tidak dihitung sebagai deteksi palsu


def test_sampler_ffmpeg_dari_run_json(tmp_path):
    """run.json dengan config.sampler = "ffmpeg": sampel k jatuh di frame floor((k + 0,5) x 5) = 5k + 2,
    jadi tanpa pergeseran manual pun box pipeline cocok dengan label."""
    import json
    from synth_cvat import make_frozen_clip
    root = make_frozen_clip(tmp_path)
    rj = root / "runs" / "palsu" / "run.json"
    d = json.loads(rj.read_text())
    d["config"]["sampler"] = "ffmpeg"
    rj.write_text(json.dumps(d))
    assert [gtm.sample_frame(k, 3, 15, "ffmpeg") for k in range(3)] == [2, 7, 12]
    assert [gtm.sample_frame(k, 3, 15, "exact") for k in range(3)] == [0, 5, 10]
    c = gtm.find_clips(root / "frozen", [root])[0]
    raw = gtm.load_run(root / "runs" / "palsu", c.name, 15)
    shift, _ = gtm.estimate_shift(c, raw)
    assert shift == 0
    seq = gtm.to_seq(c, gtm.oracle_dets(c), "label")
    assert mot17.mm.summarize(gtm.eval_run(seq, root / "runs" / "palsu", 15)["raw"])["IDF1"] > 0.9


def test_run_di_klip_label_sebagian(tmp_path):
    """Label hanya tiap 5 frame dan pipeline jatuh di 5k + 2: frame digeser ke frame berlabel terdekat."""
    import json
    from synth_cvat import make_frozen_clip
    root = make_frozen_clip(tmp_path, every=5)
    rj = root / "runs" / "palsu" / "run.json"
    d = json.loads(rj.read_text())
    d["config"]["sampler"] = "ffmpeg"
    rj.write_text(json.dumps(d))
    c = gtm.find_clips(root / "frozen", [root])[0]
    assert c.labelled_frames is not None
    seq = gtm.to_seq(c, gtm.oracle_dets(c), "label")
    row = gtm.eval_run(seq, root / "runs" / "palsu", 15, clip=c)
    assert row["label_digeser"] and row["frames"] > 0
    assert mot17.mm.summarize(row["raw"])["IDF1"] > 0.8


def test_geser_label_dan_tolok_ukur_detektor(tmp_path):
    """Label yang terlambat 2 frame terdeteksi dan bisa digeser; tolok ukur detektor dengan deteksi = label
    memberi recall 1, sedangkan deteksi dengan skor 0,3 tercatat sebagai 'skor rendah'."""
    from synth_cvat import make_frozen_clip
    root = make_frozen_clip(tmp_path)
    c = gtm.find_clips(root / "frozen", [root])[0]
    dets = gtm.oracle_dets(c)
    late = gtm.shift_labels(c, -2)                       # label sekarang 2 frame lebih awal dari video
    off, score = gtm.label_offset(late, dets)
    assert off == 2 and score[2] > score[0]
    fixed = gtm.shift_labels(late, off)
    assert gtm.label_offset(fixed, dets)[0] == 0
    row, err = gtm.detector_benchmark(c, dets, "label")
    assert row["recall"] > 0.95 and row["FP"] == 0 and row["AP50"] > 0.95
    low = {f: np.c_[v[:, :4], np.full(len(v), 0.3)] for f, v in dets.items()}
    row2, err2 = gtm.detector_benchmark(c, low, "skor 0,3")
    assert row2["recall"] == 0 and row2["FN skor rendah"] == row2["FN"] > 0
    assert row2["recall@0.2"] > 0.95
