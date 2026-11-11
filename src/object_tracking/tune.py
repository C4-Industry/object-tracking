"""Tuning parameter tracker dari cache deteksi, tanpa label ID manual.

Skor tiap kandidat = rata-rata IDF1 terhadap DUA referensi pada 12,5 FPS (ByteTrack dan OC-SORT), dihitung di
beberapa FPS proses dan beberapa klip. Dua referensi dipakai supaya tracker tidak dihargai hanya karena
perilakunya mirip satu tracker tertentu.

Klip tuning sebaiknya dari kamera yang BERBEDA dengan klip penilaian akhir, supaya angka akhir tidak bias.

Pencarian: random search, lalu penyempurnaan di sekitar kandidat terbaik (parameter digeser sedikit).
Dijalankan paralel di beberapa proses.
"""

from __future__ import annotations

import os
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np

from .reference_eval import compare, load_cache
from .zoo import TrackerSpec, run_spec

# ---------------- ruang pencarian ----------------
# Tiap entri: ("float", lo, hi) | ("int", lo, hi) | ("choice", [..]). Kunci "spec." mengubah field TrackerSpec,
# kunci lain mengubah params tracker.

LITE_SPACE = {
    "high_thresh": ("float", 0.35, 0.65),
    "new_gap": ("float", 0.0, 0.15),            # new_thresh = high_thresh + new_gap
    "low_thresh": ("choice", [0.05, 0.1, 0.2]),
    "min_iou": ("float", 0.05, 0.45),
    "min_iou_low": ("float", 0.1, 0.5),
    "buffers": ("choice", [(0.0,), (0.1,), (0.2,), (0.3,), (0.1, 0.4), (0.2, 0.6), (0.1, 0.3, 0.6)]),
    "max_lost_sec": ("float", 1.5, 8.0),
    "std_vel": ("float", 0.2, 1.5),
    "std_pos": ("float", 0.02, 0.10),
    "lost_velocity_decay_sec": ("float", 0.3, 2.5),
    "confirm_hits": ("choice", [1, 2, 3]),
    "nsa": ("choice", [True, False]),
    "oru": ("choice", [True, False]),
    "ocr": ("choice", [True, False]),
    "cost": ("choice", ["iou", "diou"]),
    "max_h_ratio": ("choice", [0.0, 1.5, 2.0]),
    "spec.person_min": ("choice", [0.05, 0.1, 0.2, 0.3]),
    "spec.nms_iou": ("choice", [0.5, 0.6, 0.7]),
}

ULTRA_SPACE = {
    "track_high_thresh": ("float", 0.3, 0.65),
    "new_gap": ("float", 0.0, 0.15),
    "track_low_thresh": ("choice", [0.05, 0.1, 0.2]),
    "match_thresh": ("float", 0.6, 0.95),
    "spec.buffer_sec": ("float", 1.5, 8.0),
    "spec.person_min": ("choice", [0.05, 0.1, 0.2, 0.3]),
    "spec.nms_iou": ("choice", [0.0, 0.5, 0.6, 0.7]),
}
EXTRA_SPACE = {
    "ocsort": {"inertia": ("float", 0.05, 0.4), "use_byte": ("choice", [True, False])},
    "tracktrack": {"tai_thr": ("float", 0.4, 0.7)},
}


def space_for(kind: str) -> dict:
    if kind == "lite":
        return LITE_SPACE
    return {**ULTRA_SPACE, **EXTRA_SPACE.get(kind, {})}


def _sample(space: dict, rng: random.Random) -> dict:
    out = {}
    for k, (typ, *a) in space.items():
        if typ == "float":
            out[k] = round(rng.uniform(a[0], a[1]), 3)
        elif typ == "int":
            out[k] = rng.randint(a[0], a[1])
        else:
            out[k] = rng.choice(a[0])
    return out


def _perturb(x: dict, space: dict, rng: random.Random, p: float = 0.35) -> dict:
    out = dict(x)
    for k, (typ, *a) in space.items():
        if rng.random() > p:
            continue
        if typ == "float":
            span = (a[1] - a[0]) * 0.15
            out[k] = round(float(np.clip(out[k] + rng.gauss(0, span), a[0], a[1])), 3)
        elif typ == "int":
            out[k] = int(np.clip(out[k] + rng.choice([-1, 1]), a[0], a[1]))
        else:
            out[k] = rng.choice(a[0])
    return out


def build_spec(base: TrackerSpec, x: dict, name: str | None = None) -> TrackerSpec:
    params = dict(base.params)
    spec_kw = {}
    hi_key = "high_thresh" if base.kind == "lite" else "track_high_thresh"
    new_key = "new_thresh" if base.kind == "lite" else "new_track_thresh"
    for k, v in x.items():
        if k == "new_gap":
            continue
        if k.startswith("spec."):
            spec_kw[k[5:]] = v
        else:
            params[k] = tuple(v) if k == "buffers" else v
    if "new_gap" in x:
        params[new_key] = round(min(0.95, params.get(hi_key, 0.5) + x["new_gap"]), 3)
    return replace(base, name=name or base.name, params=params, **spec_kw)


# ---------------- evaluasi ----------------

OCSORT_REF = TrackerSpec("ref OC-SORT", "ocsort", {"track_high_thresh": 0.5, "track_low_thresh": 0.1,
                                                   "new_track_thresh": 0.55, "use_byte": True})


def references(cache: dict) -> dict:
    """Dua referensi pada FPS cache (12,5): ByteTrack Ultralytics dan OC-SORT, keduanya dengan deteksi skor >= 0,1."""
    from .reference_eval import run_reference

    return {"ByteTrack": run_reference(cache), "OC-SORT": run_spec(cache, cache["ref_fps"], OCSORT_REF)[0]}


_W: dict = {}


def _init_worker(cache_paths: dict, refs: dict) -> None:
    _W["caches"] = {c: load_cache(p) for c, p in cache_paths.items()}
    _W["refs"] = refs


def score_spec(spec: TrackerSpec, caches: dict, refs: dict, fps_list) -> dict:
    """refs: {klip: {nama_ref: hasil run_reference}}. Rata-rata IDF1 di semua klip x FPS x referensi."""
    idf1, idsw, ids, ids_ref, cov = [], 0, 0, 0, []
    for c, cache in caches.items():
        for fps in fps_list:
            hyp, _ = run_spec(cache, fps, spec)
            for rname, ref in refs[c].items():
                r = compare(ref, hyp)
                idf1.append(r["IDF1"])
                cov.append(r["coverage"])
                if rname == next(iter(refs[c])):
                    idsw += r["id_switches"]
                    ids += r["ids_hyp"]
                    ids_ref += r["ids_ref"]
    return {"score": float(np.mean(idf1)), "coverage": float(np.mean(cov)), "id_switches": idsw,
            "ids": ids, "ids_ref": ids_ref}


def _job(args):
    spec, fps_list, x = args
    try:
        return {**score_spec(spec, _W["caches"], _W["refs"], fps_list), "x": x, "spec": spec}
    except Exception as e:  # kombinasi parameter yang tidak valid untuk tracker tertentu
        return {"score": -1.0, "error": repr(e), "x": x, "spec": spec}


def search(base: TrackerSpec, cache_paths: dict, refs: dict, fps_list, n_random: int = 120, n_refine: int = 60,
           seed: int = 0, workers: int | None = None, progress: bool = True) -> list[dict]:
    """Random search + penyempurnaan. Hasil diurutkan dari skor terbaik. Kandidat pertama selalu `base` apa adanya."""
    rng = random.Random(seed)
    space = space_for(base.kind)
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    results: list[dict] = []

    def run_batch(xs, desc):
        jobs = [((base if x is None else build_spec(base, x)), fps_list, x) for x in xs]
        it = pool.map(_job, jobs, chunksize=1)
        if progress:
            try:
                from tqdm.auto import tqdm
                it = tqdm(it, total=len(jobs), desc=desc, leave=False)
            except ImportError:
                pass
        results.extend(it)

    with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(cache_paths, refs)) as pool:
        run_batch([None] + [_sample(space, rng) for _ in range(n_random)], f"{base.name}: acak")
        if n_refine > 0:
            top = [r["x"] for r in sorted(results, key=lambda r: -r["score"]) if r["x"] is not None][:5]
            if top:
                run_batch([_perturb(top[i % len(top)], space, rng) for i in range(n_refine)], f"{base.name}: sempurnakan")
    return sorted(results, key=lambda r: (-r["score"], r.get("id_switches", 0)))
