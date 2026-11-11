"""Metrik MOT standar: HOTA, IDF1 (Identity) dan MOTA (CLEAR).

Ditulis ulang mengikuti logika TrackEval (https://github.com/JonathonLuiten/TrackEval, MIT), supaya tidak
butuh dependency tambahan. Paket `motmetrics` tidak jalan di numpy 2 (memakai np.asfarray).
Hasilnya dicek sama dengan TrackEval di tests/test_mot_metrics.py.

Format input per sekuens:
    gt[t]  = (ids: int array N, boxes: Nx4 xywh)
    hyp[t] = (ids: int array M, boxes: Mx4 xywh)
untuk setiap timestep t yang dinilai (urutan list = urutan waktu).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

EPS = np.finfo("float").eps
ALPHAS = np.arange(0.05, 0.99, 0.05)          # 19 ambang lokalisasi HOTA


def box_iou_xywh(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    ax2, ay2 = a[:, 0] + a[:, 2], a[:, 1] + a[:, 3]
    bx2, by2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    iw = np.clip(np.minimum(ax2[:, None], bx2[None]) - np.maximum(a[:, 0, None], b[None, :, 0]), 0, None)
    ih = np.clip(np.minimum(ay2[:, None], by2[None]) - np.maximum(a[:, 1, None], b[None, :, 1]), 0, None)
    inter = iw * ih
    union = (a[:, 2] * a[:, 3])[:, None] + (b[:, 2] * b[:, 3])[None] - inter
    out = np.zeros_like(inter)
    m = union > EPS
    out[m] = inter[m] / union[m]
    return out


def _relabel(frames):
    """id asli -> 0..K-1, supaya bisa dipakai sebagai indeks matriks."""
    uniq = np.unique(np.concatenate([np.asarray(f[0], dtype=np.int64) for f in frames] + [np.zeros(0, np.int64)]))
    lut = {int(u): i for i, u in enumerate(uniq)}
    return [np.array([lut[int(i)] for i in f[0]], dtype=np.int64) for f in frames], len(uniq)


def evaluate_sequence(gt: list, hyp: list) -> dict:
    """Hitung semua angka mentah (yang bisa dijumlah antar sekuens) untuk satu sekuens."""
    assert len(gt) == len(hyp), "jumlah timestep gt dan hyp harus sama"
    gt_ids, n_gt = _relabel(gt)
    tr_ids, n_tr = _relabel(hyp)
    sims = [box_iou_xywh(g[1], h[1]) for g, h in zip(gt, hyp)]
    n_gt_dets = int(sum(len(i) for i in gt_ids))
    n_tr_dets = int(sum(len(i) for i in tr_ids))
    r = {"n_gt_dets": n_gt_dets, "n_tr_dets": n_tr_dets, "n_gt_ids": n_gt, "n_tr_ids": n_tr}

    # ---------------- CLEAR (MOTA, ID switch) ----------------
    tp = fn = fp = idsw = 0
    gt_matched_count = np.zeros(n_gt)
    gt_frames = np.zeros(n_gt)
    prev_tr = np.full(n_gt, np.nan)             # tracker terakhir yang pernah dipasangkan ke gt ini
    prev_step_tr = np.full(n_gt, np.nan)        # tracker yang dipasangkan di timestep sebelumnya
    for g, h, s in zip(gt_ids, tr_ids, sims):
        if len(g) == 0:
            fp += len(h)
            continue
        if len(h) == 0:
            fn += len(g)
            gt_frames[g] += 1
            continue
        gt_frames[g] += 1
        score = 1000 * (h[None, :] == prev_step_tr[g][:, None]) + s
        score[s < 0.5 - EPS] = 0
        rows, cols = linear_sum_assignment(-score)
        ok = score[rows, cols] > 0 + EPS
        rows, cols = rows[ok], cols[ok]
        mg, mt = g[rows], h[cols]
        prev_m = prev_tr[mg]
        is_sw = (~np.isnan(prev_m)) & (mt != prev_m)
        idsw += int(is_sw.sum())
        gt_matched_count[mg] += 1
        prev_step_tr[:] = np.nan
        prev_step_tr[mg] = mt
        prev_tr[mg] = mt
        tp += len(mg)
        fn += len(g) - len(mg)
        fp += len(h) - len(mg)
    ratio = gt_matched_count[gt_frames > 0] / gt_frames[gt_frames > 0]
    r.update(CLR_TP=tp, CLR_FN=fn, CLR_FP=fp, IDSW=idsw,
             MT=int(np.sum(ratio > 0.8)), PT=int(np.sum((ratio >= 0.2) & (ratio <= 0.8))),
             ML=int(np.sum(ratio < 0.2)))

    # ---------------- Identity (IDF1) ----------------
    pot = np.zeros((n_gt, n_tr))
    gt_cnt = np.zeros(n_gt)
    tr_cnt = np.zeros(n_tr)
    for g, h, s in zip(gt_ids, tr_ids, sims):
        if len(g) and len(h):
            mg, mt = np.nonzero(s >= 0.5 - EPS)
            np.add.at(pot, (g[mg], h[mt]), 1)
        np.add.at(gt_cnt, g, 1)
        np.add.at(tr_cnt, h, 1)
    if n_gt == 0 or n_tr == 0:
        idfn, idfp = int(gt_cnt.sum()), int(tr_cnt.sum())
    else:
        n = n_gt + n_tr
        fp_mat = np.zeros((n, n))
        fn_mat = np.zeros((n, n))
        fp_mat[n_gt:, :n_tr] = 1e10
        fn_mat[:n_gt, n_tr:] = 1e10
        for i in range(n_gt):
            fn_mat[i, n_tr + i] = gt_cnt[i]
        for j in range(n_tr):
            fp_mat[n_gt + j, j] = tr_cnt[j]
        fn_mat[:n_gt, :n_tr] = gt_cnt[:, None] - pot
        fp_mat[:n_gt, :n_tr] = tr_cnt[None, :] - pot
        rows, cols = linear_sum_assignment(fn_mat + fp_mat)
        idfn = int(round(fn_mat[rows, cols].sum()))
        idfp = int(round(fp_mat[rows, cols].sum()))
    r.update(IDTP=int(gt_cnt.sum()) - idfn, IDFN=idfn, IDFP=idfp)

    # ---------------- HOTA ----------------
    na = len(ALPHAS)
    h_tp, h_fn, h_fp = np.zeros(na), np.zeros(na), np.zeros(na)
    loca = np.zeros(na)
    if n_gt == 0 or n_tr == 0:
        h_fn[:] = n_gt_dets
        h_fp[:] = n_tr_dets
        ass_a = ass_re = ass_pr = np.zeros(na)
    else:
        pot = np.zeros((n_gt, n_tr))
        for g, h, s in zip(gt_ids, tr_ids, sims):
            if len(g) and len(h):
                denom = s.sum(0)[None, :] + s.sum(1)[:, None] - s
                sim_iou = np.zeros_like(s)
                m = denom > EPS
                sim_iou[m] = s[m] / denom[m]
                pot[g[:, None], h[None, :]] += sim_iou
        gas = pot / (gt_cnt[:, None] + tr_cnt[None, :] - pot)
        matches = [np.zeros((n_gt, n_tr)) for _ in ALPHAS]
        for g, h, s in zip(gt_ids, tr_ids, sims):
            if len(g) == 0:
                h_fp += len(h)
                continue
            if len(h) == 0:
                h_fn += len(g)
                continue
            score = gas[g[:, None], h[None, :]] * s
            rows, cols = linear_sum_assignment(-score)
            for a, alpha in enumerate(ALPHAS):
                ok = s[rows, cols] >= alpha - EPS
                ar, ac = rows[ok], cols[ok]
                k = len(ar)
                h_tp[a] += k
                h_fn[a] += len(g) - k
                h_fp[a] += len(h) - k
                if k:
                    loca[a] += s[ar, ac].sum()
                    matches[a][g[ar], h[ac]] += 1
        ass_a, ass_re, ass_pr = np.zeros(na), np.zeros(na), np.zeros(na)
        for a in range(na):
            mc = matches[a]
            tpa = max(1.0, h_tp[a])
            ass_a[a] = np.sum(mc * (mc / np.maximum(1, gt_cnt[:, None] + tr_cnt[None, :] - mc))) / tpa
            ass_re[a] = np.sum(mc * (mc / np.maximum(1, gt_cnt[:, None]))) / tpa
            ass_pr[a] = np.sum(mc * (mc / np.maximum(1, tr_cnt[None, :]))) / tpa
    r.update(HOTA_TP=h_tp, HOTA_FN=h_fn, HOTA_FP=h_fp, LocA_sum=loca, AssA=ass_a, AssRe=ass_re, AssPr=ass_pr)
    return r


_SUM_KEYS = ("n_gt_dets", "n_tr_dets", "n_gt_ids", "n_tr_ids", "CLR_TP", "CLR_FN", "CLR_FP", "IDSW",
             "MT", "PT", "ML", "IDTP", "IDFN", "IDFP")


def combine(raws: list[dict]) -> dict:
    """Gabungkan angka mentah beberapa sekuens (cara TrackEval: jumlahkan hitungan, AssA dirata-rata berbobot TP)."""
    out = {k: sum(r[k] for r in raws) for k in _SUM_KEYS}
    tp = sum(r["HOTA_TP"] for r in raws)
    out["HOTA_TP"], out["HOTA_FN"], out["HOTA_FP"] = tp, sum(r["HOTA_FN"] for r in raws), sum(r["HOTA_FP"] for r in raws)
    out["LocA_sum"] = sum(r["LocA_sum"] for r in raws)
    for k in ("AssA", "AssRe", "AssPr"):
        out[k] = sum(r[k] * r["HOTA_TP"] for r in raws) / np.maximum(1e-10, tp)
    return out


def summarize(raw: dict) -> dict:
    """Angka mentah (satu sekuens atau gabungan) -> metrik akhir. Semua rasio dalam 0..1."""
    tp, fn, fp = raw["HOTA_TP"], raw["HOTA_FN"], raw["HOTA_FP"]
    det_a = tp / np.maximum(1, tp + fn + fp)
    hota = np.sqrt(det_a * raw["AssA"])
    n_gt = max(1, raw["n_gt_dets"])
    idtp, idfn, idfp = raw["IDTP"], raw["IDFN"], raw["IDFP"]
    return {
        "HOTA": float(hota.mean()),
        "DetA": float(det_a.mean()),
        "AssA": float(np.mean(raw["AssA"])),
        "LocA": float(np.mean(np.maximum(1e-10, raw["LocA_sum"]) / np.maximum(1e-10, tp))),
        "IDF1": idtp / max(1.0, idtp + 0.5 * idfp + 0.5 * idfn),
        "IDP": idtp / max(1.0, idtp + idfp),
        "IDR": idtp / max(1.0, idtp + idfn),
        "MOTA": (raw["CLR_TP"] - raw["CLR_FP"] - raw["IDSW"]) / n_gt,
        "IDSW": int(raw["IDSW"]),
        "FP": int(raw["CLR_FP"]),
        "FN": int(raw["CLR_FN"]),
        "MT": int(raw["MT"]),
        "ML": int(raw["ML"]),
        "GT_IDs": int(raw["n_gt_ids"]),
        "IDs": int(raw["n_tr_ids"]),
    }
