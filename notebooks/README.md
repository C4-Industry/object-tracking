# Notebook

| Notebook | Isi | Butuh |
|---|---|---|
| `01_mengenal_lite_tracker.ipynb` | Cara kerja Lite tracker di adegan CCTV sintetis, pengaruh tiap pengaturan, dan contoh kode untuk memasang ke pipeline | Tidak ada (jalan dalam beberapa detik) |
| `02_tuning_cctv.ipynb` | Tuning dan penilaian semua tracker (Lite, ByteTrack, OC-SORT, BoT-SORT, TrackTrack, FastTrack) di klip CCTV PT SRP, ablasi Lite, perkiraan kapasitas kamera | Detektor + video PT SRP |
| `03_benchmark_mot17.ipynb` | Benchmark di MOT17 dengan label ID dari manusia: FPS asli, FPS rendah, kamera diam vs bergerak, validasi silang, video perbandingan | Internet (label MOT17 sekitar 10 MB) |
| `04_bandingkan_tracker_video.ipynb` | Membuka aplikasi Gradio "Bandingkan Tracker": pilih video, lihat semua tracker secara live dengan FPS, unduh hasilnya | Detektor + video |
| `05_evaluasi_ground_truth.ipynb` | Penilaian terhadap label ID manusia (CVAT): cek label, HOTA/IDF1/ID switch per FPS, detektor sempurna vs D-FINE, per klip, validasi silang, video dengan tanda ID switch | Detektor + klip berlabel di `data/ground_truth/` |

## Menjalankan di VS Code

```bash
cd ~/Documents/GitHub/object-tracking
uv sync          # sekali: membuat .venv dengan semua paket notebook
```

Buka notebook, klik nama kernel di kanan atas, pilih **.venv (object-tracking)**. Sel pertama tiap notebook mengecek kernel dan paketnya.

Untuk cek cepat dengan data lebih sedikit, set `QUICK = True` di sel pengaturan (atau jalankan VS Code dengan `OT_QUICK=1`).

## Hasil

- `experiments/02_tuning_cctv/<waktu>/`, `experiments/03_mot17/<waktu>/`, `experiments/05_ground_truth/<waktu>/`: `results.json`, CSV, grafik, video (video tidak masuk git)
- `configs/tracker_tuned.json`: semua spec hasil tuning notebook 02 (dibaca notebook lain dan aplikasi)
- `outputs/cache/`: cache deteksi (boleh dihapus, akan dibuat ulang). `outputs/compare/`: video hasil aplikasi

## Data ground truth (notebook 05)

Bawaannya membaca `~/PPE - PT SRP/ground truth/` (ganti `GT_ROOT` kalau pindah):

```
ground truth/
  <klip>_15fps.mp4                       video 15 FPS (juga varian _blur dan _reenc)
  frozen/<klip>/gt.jsonl, meta.json      label beku: orang per frame, area diabaikan, status helm/vest
  runs/<run>/<klip>/observations.jsonl   output pipeline yang sudah dijalankan (dinilai di bagian 4)
```

Export CVAT biasa juga bisa: satu folder per klip berisi `annotations.xml` ("CVAT for video 1.1") atau `gt/gt.txt` ("MOT 1.1"), plus videonya.

Untuk menilai pipeline dengan Lite terpasang, jalankan dulu `scripts/run_pipeline_lite.py` (lihat `docs/INTEGRASI.md`). Run barunya masuk ke `runs/` dan otomatis dinilai di bagian 4.

Bagian 6 (tolok ukur detektor) menambahkan hasil setiap detektor ke `experiments/05_ground_truth/tolok_ukur_detektor.csv`. Satu baris per klip, file detektor (sha), dan versi label, jadi detektor baru tinggal ditambahkan ke `DETECTORS` lalu dibandingkan dengan riwayatnya.

Klip GT baru yang sedang disiapkan ada di `ground truth/kandidat_baru/` (video 15 fps dan pre-label CVAT, belum berlabel). Notebook baru membacanya setelah di-freeze ke `frozen/`.

