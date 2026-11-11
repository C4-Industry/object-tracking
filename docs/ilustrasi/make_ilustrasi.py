import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Ellipse, FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).resolve().parent   # gambar ditulis di folder ini (docs/ilustrasi)
OUT.mkdir(parents=True, exist_ok=True)

# Warna
C_TRACK = {3: "#185FA5", 5: "#0F6E56", 7: "#993556", 12: "#854F0B"}
C_YAKIN = "#3B6D11"
C_RAGU = "#BA7517"
C_LOW = "#888780"
C_BG = "#F4F2EC"
C_FLOOR = "#E8E5DC"
C_TEXT = "#2C2C2A"
C_MUTED = "#5F5E5A"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11})

TH = {"min": 0.05, "high": 0.517, "low": 0.10, "new": 0.535, "buf": 0.10, "decay": 1.5, "lost": 4.2,
      "min_iou": 0.365, "min_iou_low": 0.386, "ocr_iou": 0.30, "nms": 0.7}


def f2(x):
    return f"{x:.2f}".replace(".", ",")


def iou(a, b, buf=0.0):
    def ex(r):
        x1, y1, x2, y2 = r
        w, h = x2 - x1, y2 - y1
        return (x1 - buf * w, y1 - buf * h, x2 + buf * w, y2 + buf * h)
    a, b = ex(a), ex(b)
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i
    return i / u if u > 0 else 0


def scene(ax, forklift=True, title_frame=None):
    ax.set_xlim(0, 100)
    ax.set_ylim(62, 0)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 100, 62, fc=C_FLOOR, ec="#B4B2A9", lw=1))
    for x in range(0, 100, 10):
        ax.plot([x, x + 6], [62, 30], color="#D3D1C7", lw=0.6, zorder=0)
    if forklift:
        ax.add_patch(FancyBboxPatch((50, 26), 16, 16, boxstyle="round,pad=0.4", fc="#EF9F27", ec="#854F0B", lw=1.2, zorder=4))
        ax.add_patch(Rectangle((52, 20), 2, 22, fc="#5F5E5A", zorder=4))
        ax.add_patch(Ellipse((54, 43), 4, 4, fc="#2C2C2A", zorder=5))
        ax.add_patch(Ellipse((63, 43), 4, 4, fc="#2C2C2A", zorder=5))
        ax.text(58, 34, "forklift", ha="center", va="center", fontsize=10, color="#412402", zorder=6)
    if title_frame:
        ax.text(1.5, 3, title_frame, fontsize=10, color=C_MUTED, va="top")


def person(ax, cx, top, h, alpha=1.0, z=3):
    w = h * 0.38
    ax.add_patch(Ellipse((cx, top + h * 0.1), h * 0.17, h * 0.17, fc="#D3D1C7", ec="#888780", lw=0.6, alpha=alpha, zorder=z))
    ax.add_patch(Rectangle((cx - w / 2, top + h * 0.2), w, h * 0.45, fc="#B5D4F4", ec="#888780", lw=0.6, alpha=alpha, zorder=z))
    ax.add_patch(Rectangle((cx - w / 2.6, top + h * 0.65), w / 2.8, h * 0.35, fc="#888780", alpha=alpha, zorder=z))
    ax.add_patch(Rectangle((cx + w / 2.6 - w / 2.8, top + h * 0.65), w / 2.8, h * 0.35, fc="#888780", alpha=alpha, zorder=z))


def pbox(cx, top, h):
    w = h * 0.5
    return (cx - w / 2, top, cx + w / 2, top + h)


def box(ax, r, color, ls="-", lw=2.0, label=None, label_pos="top", alpha=1.0, z=7, fill=False):
    x1, y1, x2, y2 = r
    ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=fill, fc=color if fill else "none", ec=color, ls=ls, lw=lw,
                           alpha=alpha, zorder=z))
    if label:
        y = y1 - 0.8 if label_pos == "top" else y2 + 0.8
        va = "bottom" if label_pos == "top" else "top"
        ax.text(x1, y, label, color="white", fontsize=9, va=va, ha="left", zorder=z + 1,
                bbox=dict(boxstyle="square,pad=0.15", fc=color, ec="none", alpha=alpha))


def expand(r, b):
    x1, y1, x2, y2 = r
    w, h = x2 - x1, y2 - y1
    return (x1 - b * w, y1 - b * h, x2 + b * w, y2 + b * h)


def arrow(ax, p, q, color=C_TEXT, ls="-", lw=1.6, z=9, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw, ls=ls, zorder=z))


def side(ax, heading, lines, foot=None):
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(1, 0)
    ax.text(0, 0.02, heading, fontsize=13, fontweight="bold", color=C_TEXT, va="top")
    y = 0.12
    for ln in lines:
        if ln.startswith("##"):
            ax.text(0, y, ln[2:].strip(), fontsize=11, fontweight="bold", color=C_TEXT, va="top", wrap=True)
            y += 0.065
            continue
        t = ax.text(0.0, y, ln, fontsize=10.5, color=C_TEXT, va="top", linespacing=1.35)
        y += 0.055 * (ln.count("\n") + 1) + 0.025
    if foot:
        ax.text(0, 0.97, foot, fontsize=9.5, color=C_MUTED, va="bottom", style="italic", linespacing=1.3)


def legend_row(fig, items, y=0.035):
    x = 0.04
    for color, ls, txt in items:
        fig.lines.append(plt.Line2D([x, x + 0.025], [y, y], transform=fig.transFigure, color=color, ls=ls, lw=2.2))
        fig.text(x + 0.03, y, txt, va="center", fontsize=9.5, color=C_TEXT)
        x += 0.03 + 0.0075 * len(txt) + 0.03


def new_fig(step, title):
    fig = plt.figure(figsize=(13, 7.4), dpi=140, facecolor="white")
    fig.text(0.04, 0.955, f"Langkah {step}", fontsize=11, color=C_MUTED)
    fig.text(0.04, 0.915, title, fontsize=17, fontweight="bold", color=C_TEXT)
    ax = fig.add_axes([0.03, 0.09, 0.60, 0.78])
    sx = fig.add_axes([0.66, 0.09, 0.31, 0.78])
    return fig, ax, sx


def save(fig, name):
    fig.savefig(OUT / name, dpi=140, facecolor="white")
    plt.close(fig)


# ---------------- Langkah 0 ----------------
fig, ax, sx = new_fig(0, "Saring deteksi: buang skor sangat rendah dan box dobel")
scene(ax, forklift=False, title_frame="Output detektor mentah di frame t")
person(ax, 22, 20, 30)
person(ax, 72, 24, 26)
b_a = pbox(22, 20, 30)
b_dup = (b_a[0] + 1.2, b_a[1] + 1.5, b_a[2] + 1.6, b_a[3] - 0.5)
b_b = pbox(72, 24, 26)
b_noise = (40, 6, 46, 16)
box(ax, b_a, C_YAKIN, label="orang 0,91")
box(ax, b_dup, C_LOW, ls="--", lw=1.6, label="dobel 0,22", label_pos="bottom")
box(ax, b_b, C_YAKIN, label="orang 0,78")
box(ax, b_noise, C_LOW, ls=":", lw=1.6, label="0,03", label_pos="bottom")
ax.plot([40, 46], [6, 16], color="#A32D2D", lw=2, zorder=9)
ax.plot([46, 40], [6, 16], color="#A32D2D", lw=2, zorder=9)
ax.text(47.5, 11, "dibuang: skor < 0,05", color="#A32D2D", fontsize=9.5, va="center", zorder=9)
ax.text(b_dup[2] + 1, b_dup[3] - 4, f"dibuang: IoU {f2(iou(b_a, b_dup))} >= 0,7\ndengan box 0,91", color="#A32D2D", fontsize=9.5, zorder=9)
side(sx, "Apa yang terjadi", [
    "D-FINE mengeluarkan banyak box per frame,\ntermasuk yang skornya sangat rendah.",
    "Box dengan skor di bawah person_min\ndibuang lebih dulu.",
    "Box dobel (IoU >= 0,7 dengan box lain yang\nlebih yakin) dibuang dengan NMS. Box dobel\nmembuat tracker salah sambung di area ramai.",
    "## Hasil",
    "Tinggal satu box per orang.",
], foot="Catatan: tahap mana pun hanya memakai skor >= 0,1,\njadi person_min 0,05 sama saja dengan 0,1.")
legend_row(fig, [(C_YAKIN, "-", "deteksi dipakai"), (C_LOW, "--", "deteksi dibuang")])
save(fig, "langkah_0_saring.png")

# ---------------- Langkah 1 ----------------
fig, ax, sx = new_fig(1, "Tebak posisi semua orang dengan Kalman berbasis detik")
scene(ax, title_frame="Frame t = 10,0 detik (frame sebelumnya t = 9,75 detik)")
t3_prev = pbox(18, 18, 30); t3_pred = pbox(24, 18, 30)
person(ax, 18, 18, 30, alpha=0.35)
box(ax, t3_prev, C_TRACK[3], ls="-", lw=1.2, alpha=0.4, label="#3 di t-1")
box(ax, t3_pred, C_TRACK[3], ls="--", label="#3 tebakan t", label_pos="bottom")
arrow(ax, (18, 33), (24, 33), C_TRACK[3])
ax.text(21, 31, "v = 24 px/detik", color=C_TRACK[3], fontsize=9, ha="center", va="bottom", zorder=10)
t7_last = pbox(46, 22, 26)
person(ax, 46, 22, 26, alpha=0.35)
box(ax, t7_last, C_TRACK[7], ls="-", lw=1.2, alpha=0.4, label="#7 terakhir terlihat t = 8,0")
box(ax, (t7_last[0] + 0.4, t7_last[1] + 0.4, t7_last[2] + 0.4, t7_last[3] + 0.4), C_TRACK[7], ls="--", label="#7 tebakan t", label_pos="bottom")
ax.text(46, 52, "hilang 2,0 detik > 1,5:\nkecepatan dianggap 0 (diam)", color=C_TRACK[7], fontsize=9, ha="center", va="top", zorder=10)
side(sx, "Apa yang terjadi", [
    "Setiap orang yang sedang diikuti punya\nposisi, ukuran, dan kecepatan (piksel/detik).",
    "Posisinya digeser sejauh kecepatan x selisih\nwaktu nyata antar frame. Jadi tetap benar di\n2 FPS, 4 FPS, atau saat frame terlewat.",
    "Orang yang hilang lebih dari 1,5 detik\ndianggap diam. Tanpa ini, tebakannya terus\nbergerak menjauh dari tempat dia hilang.",
    "## Hasil",
    "Box tebakan (putus-putus) untuk tiap orang,\nsiap dicocokkan dengan deteksi frame ini.",
])
legend_row(fig, [(C_TRACK[3], "-", "posisi frame sebelumnya"), (C_TRACK[3], "--", "posisi tebakan Kalman")])
save(fig, "langkah_1_kalman.png")

# ---------------- Langkah 2 ----------------
fig, ax, sx = new_fig(2, "Bagi deteksi menjadi YAKIN dan RAGU")
scene(ax, title_frame="Deteksi orang di frame t setelah disaring")
dets = [((24, 18, 30), 0.88), ((72, 24, 26), 0.81), ((44, 30, 20), 0.32), ((84, 8, 14), 0.61), ((36, 10, 12), 0.14)]
for (cx, top, h), s in dets:
    person(ax, cx, top, h)
for (cx, top, h), s in dets:
    c = C_YAKIN if s >= TH["high"] else C_RAGU
    box(ax, pbox(cx, top, h), c, label=f2(s))
ax.add_patch(Rectangle((40, 26), 26, 16, fc=C_FLOOR, alpha=0.0))
side(sx, "Apa yang terjadi", [
    "Deteksi dibagi dua menurut skornya:",
    "YAKIN: skor >= 0,517\nBoleh dipakai untuk orang lama maupun\nmembuat ID baru.",
    "RAGU: skor 0,1 sampai 0,517\nHanya dipakai untuk menyambung orang yang\nbaru saja terlihat. Tidak pernah membuat ID\nbaru, supaya box palsu tidak jadi orang.",
    "## Kenapa RAGU tetap dipakai",
    "Orang yang setengah tertutup atau buram\nsering dapat skor rendah. Kalau dibuang,\nID-nya putus (inti ide ByteTrack).",
])
legend_row(fig, [(C_YAKIN, "-", "YAKIN (>= 0,517)"), (C_RAGU, "-", "RAGU (0,1 sampai 0,517)")])
save(fig, "langkah_2_bagi_deteksi.png")

# ---------------- Langkah 3 ----------------
fig, ax, sx = new_fig(3, "Tahap 1: orang dikenal dicocokkan dengan deteksi YAKIN")
scene(ax, forklift=False, title_frame="Zoom ke satu orang yang berjalan cepat (2 FPS)")
pred = (20, 12, 34, 40)
det = (27.25, 13, 41.25, 41)
person(ax, 34.25, 13, 28)
box(ax, pred, C_TRACK[3], ls="--", label="#3 tebakan")
box(ax, det, C_YAKIN, label="deteksi 0,88", label_pos="bottom")
box(ax, expand(pred, TH["buf"]), C_TRACK[3], ls=":", lw=1.3, alpha=0.8)
box(ax, expand(det, TH["buf"]), C_YAKIN, ls=":", lw=1.3, alpha=0.8)
ex_p, ex_d = expand(pred, TH["buf"]), expand(det, TH["buf"])
ov = (max(ex_p[0], ex_d[0]), max(ex_p[1], ex_d[1]), min(ex_p[2], ex_d[2]), min(ex_p[3], ex_d[3]))
ax.add_patch(Rectangle((ov[0], ov[1]), ov[2] - ov[0], ov[3] - ov[1], fc="#7F77DD", alpha=0.25, zorder=2))
ax.text(56, 14, f"IoU box asli     = {f2(iou(pred, det))}", fontsize=11, color=C_TEXT, zorder=10, family="DejaVu Sans Mono")
ax.text(56, 19, f"IoU diperbesar = {f2(iou(pred, det, TH['buf']))}", fontsize=11, color="#3C3489", zorder=10, family="DejaVu Sans Mono")
ax.text(56, 24, "batas tahap 1  = 0,365", fontsize=11, color=C_MUTED, zorder=10, family="DejaVu Sans Mono")
ax.text(56, 31, "Box asli: di bawah batas, tidak\ntersambung, jadi ID baru.\nDiperbesar 10%: lewat batas,\ntetap #3.", fontsize=10, color=C_TEXT, zorder=10, va="top")
side(sx, "Apa yang terjadi", [
    "Semua orang yang sudah pasti dicocokkan\ndengan deteksi YAKIN memakai Hungarian\n(pasangan terbaik).",
    "Kedua box diperbesar 10% di tiap sisi\n(buffered IoU / C-BIoU) sebelum IoU dihitung.",
    "Di FPS rendah orang bergeser cukup jauh,\nsehingga IoU box asli bisa turun di bawah\nbatas 0,365. Box diperbesar menaikkan IoU\nsedikit, cukup untuk pergeseran sedang.\nBox yang sama sekali tidak bersentuhan\ntetap tidak tersambung.",
    "## Hasil",
    "Yang cocok langsung ke langkah 7.\nSisanya lanjut ke tahap berikutnya.",
])
legend_row(fig, [(C_TRACK[3], "--", "tebakan #3"), (C_YAKIN, "-", "deteksi YAKIN"), (C_MUTED, ":", "diperbesar 10%"), ("#7F77DD", "-", "irisan")])
save(fig, "langkah_3_tahap1_biou.png")

# ---------------- Langkah 4 ----------------
fig, ax, sx = new_fig(4, "Tahap 2: orang yang baru saja terlihat dicocokkan dengan deteksi RAGU")
scene(ax, title_frame="#5 berjalan ke belakang forklift")
t5_pred = pbox(48, 14, 30)
person(ax, 48, 14, 30, alpha=0.9, z=2)
box(ax, t5_pred, C_TRACK[5], ls="--", label="#5 tebakan (terlihat di frame lalu)")
half = (t5_pred[0] + 0.5, t5_pred[1] + 0.3, t5_pred[2] - 0.3, 27)
box(ax, half, C_RAGU)
ax.annotate("deteksi RAGU 0,32\n(hanya setengah badan terlihat)", xy=(half[0], (half[1] + half[3]) / 2), xytext=(8, 22),
            fontsize=9.5, color=C_RAGU, arrowprops=dict(arrowstyle="-", color=C_RAGU, lw=0.8), zorder=10)
ax.text(4, 56, f"IoU (diperbesar 10%) tebakan vs deteksi RAGU = {f2(iou(t5_pred, half, TH['buf']))}, batas 0,386  ->  tersambung ke #5", fontsize=10.5, color=C_TEXT, zorder=10)
side(sx, "Apa yang terjadi", [
    "Orang yang cocok di frame sebelumnya tapi\nbelum dapat pasangan di tahap 1 dicocokkan\ndengan deteksi RAGU.",
    "Kasus khas: badan bawah tertutup forklift,\npalet, atau orang lain. Detektor hanya melihat\nsebagian, jadi skornya turun ke 0,3-an.",
    "Hanya orang yang baru saja terlihat yang\nboleh memakai deteksi RAGU, supaya box ragu\ntidak menghidupkan lagi orang yang sudah\nlama hilang.",
    "## Hasil",
    "#5 tetap #5 walau setengah tertutup.",
])
legend_row(fig, [(C_TRACK[5], "--", "tebakan #5"), (C_RAGU, "-", "deteksi RAGU")])
save(fig, "langkah_4_tahap2_ragu.png")

# ---------------- Langkah 5 ----------------
fig, ax, sx = new_fig(5, "Tahap 2b (OCR): cocokkan dengan posisi terakhir yang benar-benar terlihat")
scene(ax, title_frame="#7 tertutup forklift selama 2 detik, lalu muncul lagi")
last_obs = pbox(44, 20, 26)
kalman = (last_obs[0] + 11, last_obs[1], last_obs[2] + 11, last_obs[3])
reappear = (last_obs[0] - 1.5, last_obs[1] + 0.5, last_obs[2] - 1.0, last_obs[3] + 0.8)
person(ax, (reappear[0] + reappear[2]) / 2, reappear[1], reappear[3] - reappear[1], z=5)
box(ax, last_obs, C_TRACK[7], ls="-", lw=1.6, alpha=0.6, z=7)
box(ax, kalman, C_TRACK[7], ls="--", z=7)
box(ax, reappear, C_YAKIN, label="deteksi 0,84 (t = 10,0)", label_pos="bottom", z=8)
ax.annotate("posisi terakhir nyata #7 (t = 8,0)", xy=(last_obs[0] + 1, last_obs[1]), xytext=(8, 8),
            fontsize=9.5, color=C_TRACK[7], arrowprops=dict(arrowstyle="-", color=C_TRACK[7], lw=0.8), zorder=10)
ax.annotate("tebakan Kalman #7 (sudah jalan 1,5 detik ke kanan)", xy=(kalman[2] - 1, kalman[1]), xytext=(60, 12),
            fontsize=9.5, color=C_TRACK[7], arrowprops=dict(arrowstyle="-", color=C_TRACK[7], lw=0.8), zorder=10)
ax.text(4, 54, f"IoU deteksi vs tebakan Kalman        = {f2(iou(reappear, kalman, TH['buf']))}   (gagal di tahap 1)", fontsize=10.5, color=C_TEXT, zorder=10, family="DejaVu Sans Mono")
ax.text(4, 58.5, f"IoU deteksi vs posisi terakhir nyata = {f2(iou(reappear, last_obs, TH['buf']))}   (cocok, tetap #7)", fontsize=10.5, color="#3C3489", zorder=10, family="DejaVu Sans Mono")
side(sx, "Apa yang terjadi", [
    "Saat seseorang tertutup, Kalman tetap\nmenebak posisinya. Tebakan ini bisa meleset\njauh, apalagi kalau orangnya berhenti atau\nberbalik di balik penghalang.",
    "Sisa orang yang belum cocok dicocokkan\nlagi, kali ini memakai box terakhir yang benar-\nbenar terlihat (observation-centric, dari\nOC-SORT), bukan tebakan Kalman.",
    "## Dampak",
    "Ini langkah paling berpengaruh: tanpa OCR,\nID switch naik dari 76 ke 104.",
])
legend_row(fig, [(C_TRACK[7], "-", "posisi terakhir nyata"), (C_TRACK[7], "--", "tebakan Kalman"), (C_YAKIN, "-", "deteksi baru")])
save(fig, "langkah_5_tahap2b_ocr.png")

# ---------------- Langkah 6 ----------------
fig, ax, sx = new_fig(6, "Tahap 3: kandidat orang baru dicocokkan dengan sisa deteksi")
scene(ax, forklift=False, title_frame="Orang masuk dari kanan. Frame lalu baru jadi kandidat")
cand = pbox(86, 16, 28)
newdet = (cand[0] - 3, cand[1] + 0.3, cand[2] - 3, cand[3] + 0.3)
person(ax, (newdet[0] + newdet[2]) / 2, newdet[1], 28)
box(ax, cand, C_LOW, ls=":", label="kandidat (terlihat 1x)")
box(ax, newdet, C_YAKIN, label="sisa deteksi 0,74", label_pos="bottom")
ax.text(4, 55, f"IoU = {f2(iou(cand, newdet))}  ->  terlihat 2x, kandidat menjadi orang pasti dan mulai ditampilkan", fontsize=10.5, color=C_TEXT, zorder=10)
side(sx, "Apa yang terjadi", [
    "Kandidat adalah deteksi yang baru sekali\nmuncul. Belum ditampilkan dan belum dinilai\nhelm/vest.",
    "Kandidat dicocokkan dengan deteksi YAKIN\nyang masih tersisa setelah tahap 1 sampai 2b.",
    "Kalau cocok, kandidat sudah terlihat 2 kali\ndan menjadi orang pasti dengan ID tetap.",
    "## Kenapa perlu",
    "Box palsu sesaat (bayangan, pantulan, poster)\nbiasanya tidak muncul dua kali, jadi tidak\npernah menjadi ID.",
])
legend_row(fig, [(C_LOW, ":", "kandidat"), (C_YAKIN, "-", "sisa deteksi")])
save(fig, "langkah_6_tahap3_kandidat.png")

# ---------------- Langkah 7 ----------------
fig, ax, sx = new_fig(7, "Update posisi, dan hitung ulang kecepatan setelah hilang (ORU)")
scene(ax, title_frame="#7 tersambung lagi lewat OCR di t = 10,0")
last_obs = pbox(44, 20, 26)
reappear = (last_obs[0] - 1.5, last_obs[1] + 0.5, last_obs[2] - 1.0, last_obs[3] + 0.8)
person(ax, (reappear[0] + reappear[2]) / 2, reappear[1], reappear[3] - reappear[1], z=3)
box(ax, last_obs, C_TRACK[7], ls="-", lw=1.4, alpha=0.5)
ax.annotate("posisi nyata terakhir (t = 8,0)", xy=(last_obs[2], last_obs[1] + 2), xytext=(70, 12),
            fontsize=9.5, color=C_TRACK[7], arrowprops=dict(arrowstyle="-", color=C_TRACK[7], lw=0.8), zorder=10)
box(ax, reappear, C_TRACK[7], lw=2.4, label="#7 diperbarui t = 10,0", label_pos="bottom")
arrow(ax, (54, 50), (68, 50), "#A32D2D", ls="--")
ax.text(61, 52, "kecepatan lama Kalman: ke kanan (salah)", fontsize=9, color="#A32D2D", ha="center", va="top", zorder=10)
ax.text(4, 56.3, "ORU: kecepatan baru = (posisi sekarang - posisi\nterakhir nyata) / 2,0 detik  ->  hampir diam", fontsize=10, color=C_TRACK[7], zorder=10, va="top")
side(sx, "Apa yang terjadi", [
    "Setiap orang yang tersambung di tahap 1,\n2, 2b, atau 3 diperbarui dengan box\ndeteksinya (koreksi Kalman).",
    "Kalau orang itu sempat hilang, Kalman\nmasih membawa kecepatan lama yang\nsudah tidak benar.",
    "ORU (observation-centric re-update, dari\nOC-SORT) menghitung ulang gerak dari\nposisi nyata terakhir ke posisi nyata\nsekarang.",
    "## Hasil",
    "Tebakan di frame berikutnya tidak lagi\nmeleset karena kecepatan yang keliru.",
])
legend_row(fig, [(C_TRACK[7], "-", "posisi nyata #7"), ("#A32D2D", "--", "kecepatan lama")])
save(fig, "langkah_7_update_oru.png")

# ---------------- Langkah 8 ----------------
fig = plt.figure(figsize=(13, 7.4), dpi=140, facecolor="white")
fig.text(0.04, 0.955, "Langkah 8", fontsize=11, color=C_MUTED)
fig.text(0.04, 0.915, "Buang kandidat yang tidak muncul lagi dan orang yang hilang terlalu lama", fontsize=17, fontweight="bold", color=C_TEXT)
ax = fig.add_axes([0.10, 0.12, 0.52, 0.72])
sx = fig.add_axes([0.66, 0.09, 0.31, 0.78])
ax.set_xlim(0, 7); ax.set_ylim(-0.6, 3.2); ax.invert_yaxis()
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.set_yticks([0, 1, 2, 3])
ax.set_yticklabels(["#3", "#7", "#9", "kandidat"], fontsize=11)
ax.set_xlabel("waktu (detik)")
rows = [(0, [(0, 7)], [], C_TRACK[3]), (1, [(0, 1.0), (3.0, 7)], [(1.0, 3.0)], C_TRACK[7]),
        (2, [(0, 1.2)], [(1.2, 5.4)], "#534AB7"), (3, [(2.0, 2.25)], [], C_LOW)]
for y, seen, lost, c in rows:
    for a, b in seen:
        ax.add_patch(Rectangle((a, y - 0.2), b - a, 0.4, fc=c, ec="none"))
    for a, b in lost:
        ax.add_patch(Rectangle((a, y - 0.2), b - a, 0.4, fc="none", ec=c, hatch="///", lw=1))
ax.axvline(5.4, color="#A32D2D", ls="--", lw=1.4)
ax.text(5.45, 2.0, "hilang 4,2 detik:\n#9 dihapus", color="#A32D2D", fontsize=10, va="center")
ax.text(2.35, 3.0, "tidak muncul lagi di frame\nberikutnya: kandidat dibuang", color=C_MUTED, fontsize=10, va="center")
ax.text(1.05, 0.62, "hilang 2 detik, masih diingat, lalu tersambung lagi (langkah 5)", color=C_TRACK[7], fontsize=9.5)
side(sx, "Apa yang terjadi", [
    "Kandidat yang tidak tersambung di frame\ndeteksi berikutnya langsung dihapus.",
    "Orang pasti yang tidak terlihat tetap\ndiingat (box tebakan dan posisi terakhir)\nsampai 4,2 detik, supaya bisa disambung\nlagi oleh tahap 1 sampai 2b.",
    "Setelah 4,2 detik tanpa terlihat, orang itu\ndihapus. Kalau dia muncul lagi sesudahnya,\ndia mendapat ID baru.",
    "## Kenapa 4,2 detik",
    "Cukup untuk lewat di balik forklift atau\npilar, tapi tidak terlalu lama sehingga orang\nlain yang lewat di tempat sama tidak salah\ndisambung.",
])
fig.lines.append(plt.Line2D([0.04, 0.065], [0.035, 0.035], transform=fig.transFigure, color=C_TEXT, lw=6))
fig.text(0.07, 0.035, "terlihat", va="center", fontsize=9.5)
fig.patches.append(Rectangle((0.15, 0.025), 0.025, 0.02, transform=fig.transFigure, fc="none", ec=C_TEXT, hatch="///", figure=fig))
fig.text(0.18, 0.035, "hilang tapi masih diingat", va="center", fontsize=9.5)
save(fig, "langkah_8_buang.png")

# ---------------- Langkah 9 ----------------
fig, ax, sx = new_fig(9, "Sisa deteksi dengan skor tinggi menjadi ID baru")
scene(ax, title_frame="Deteksi yang tidak cocok dengan siapa pun setelah semua tahap")
person(ax, 84, 14, 26)
person(ax, 30, 8, 14, alpha=0.6)
box(ax, pbox(84, 14, 26), "#854F0B", label="0,61: kandidat #12", label_pos="bottom")
box(ax, pbox(30, 8, 14), C_RAGU, ls="--", label="0,48: tidak jadi ID", label_pos="bottom")
ax.text(4, 52, "0,61 >= 0,535  ->  kandidat #12. Ditampilkan kalau terlihat lagi di frame berikutnya.", fontsize=10.5, color=C_TEXT, zorder=10)
ax.text(4, 57, "0,48 < 0,535  ->  diabaikan. Box ragu tidak pernah membuat orang baru.", fontsize=10.5, color=C_TEXT, zorder=10)
side(sx, "Apa yang terjadi", [
    "Deteksi yang tersisa setelah tahap 1 sampai 3\nberarti tidak cocok dengan siapa pun.",
    "Hanya yang skornya >= 0,535 yang dijadikan\nkandidat ID baru. Ambang ini sedikit di atas\nYAKIN (0,517), jadi membuat orang baru lebih\nketat daripada menyambung orang lama.",
    "Kandidat baru tampil setelah terlihat 2 kali\n(langkah 6 di frame berikutnya).",
    "## Inti tracker ini",
    "Seseorang punya 4 kesempatan disambung\nke ID lama (tahap 1, 2, 2b, 3) sebelum dia\ndianggap orang baru.",
])
legend_row(fig, [("#854F0B", "-", "kandidat ID baru"), (C_RAGU, "--", "tidak cukup yakin")])
save(fig, "langkah_9_id_baru.png")

# ---------------- Ringkasan ----------------
import matplotlib.image as mpimg
files = sorted(OUT.glob("langkah_*.png"))
fig, axes = plt.subplots(5, 2, figsize=(16, 23), dpi=110, facecolor="white")
for a, f in zip(axes.ravel(), files):
    a.imshow(mpimg.imread(f)); a.axis("off")
fig.suptitle("Tracker per frame: langkah 0 sampai 9", fontsize=20, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
fig.savefig(OUT / "ringkasan_semua_langkah.png", facecolor="white")
print(sorted(p.name for p in OUT.glob("*.png")))
