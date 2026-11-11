# Notebook

| Notebook | Isi | Butuh |
|---|---|---|
| `01_mengenal_lite_tracker.ipynb` | Cara kerja Lite tracker di adegan CCTV sintetis, pengaruh tiap pengaturan, dan contoh kode untuk memasang ke pipeline | Tidak ada (jalan dalam beberapa detik) |
| `02_tuning_cctv.ipynb` | Tuning dan penilaian semua tracker (Lite, ByteTrack, OC-SORT, BoT-SORT, TrackTrack, FastTrack) di klip CCTV PT SRP, ablasi Lite, perkiraan kapasitas kamera | Detektor + video PT SRP |
| `03_benchmark_mot17.ipynb` | Benchmark di MOT17 dengan label ID dari manusia: FPS asli, FPS rendah, kamera diam vs bergerak, validasi silang, video perbandingan | Internet (label MOT17 sekitar 10 MB) |
| `04_bandingkan_tracker_video.ipynb` | Membuka aplikasi Gradio "Bandingkan Tracker": pilih video, lihat semua tracker secara live dengan FPS, unduh hasilnya | Detektor + video |

## Menjalankan di VS Code

```bash
cd ~/Documents/GitHub/object-tracking
uv sync          # sekali: membuat .venv dengan semua paket notebook
```

Buka notebook, klik nama kernel di kanan atas, pilih **.venv (object-tracking)**. Sel pertama tiap notebook mengecek kernel dan paketnya.

Untuk cek cepat dengan data lebih sedikit, set `QUICK = True` di sel pengaturan (atau jalankan VS Code dengan `OT_QUICK=1`).

## Hasil

- `experiments/02_tuning_cctv/<waktu>/` dan `experiments/03_mot17/<waktu>/`: `results.json`, CSV, grafik, video (video tidak masuk git)
- `configs/tracker_tuned.json`: semua spec hasil tuning notebook 02 (dibaca notebook lain dan aplikasi)
- `outputs/cache/`: cache deteksi (boleh dihapus, akan dibuat ulang). `outputs/compare/`: video hasil aplikasi
