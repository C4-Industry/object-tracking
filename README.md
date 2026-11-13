# object-tracking

Lite tracker untuk melacak orang di CCTV dengan FPS rendah (2 sampai 6 FPS per kamera), untuk proyek deteksi APD (helm dan vest) PT SRP. Repo ini berisi tracker-nya, API untuk memasangnya ke pipeline utama, serta alat untuk evaluasi, tuning, dan perbandingan dengan tracker lain.

## Kenapa Lite tracker

Supaya 12 kamera muat di satu mesin, tiap kamera hanya diproses beberapa frame per detik. Tracker yang umum (ByteTrack, OC-SORT, BoT-SORT) dirancang untuk 25 sampai 30 FPS dan menghitung gerak per frame, sehingga ID orang mudah tertukar di FPS rendah. Lite tracker:

- menghitung gerak dalam **detik**, jadi tebakan posisi tetap benar di 2 FPS maupun 25 FPS,
- tetap memakai **deteksi skor rendah** untuk menyambung orang yang setengah tertutup (ide ByteTrack),
- membandingkan box yang **diperbesar** supaya pergeseran antar frame tetap tersambung (ide C-BIoU),
- **pulih setelah tertutup** dengan posisi asli terakhir (OCR dan ORU, ide OC-SORT),
- membuang **box dobel** dari detektor dan menampilkan ID baru setelah terlihat 2 kali.

Hasil di 5 klip kamera test PT SRP (2 sampai 6 FPS), dinilai terhadap tracker referensi (belum label manusia):

| Detektor | Lite tuned | ByteTrack tuned | ByteTrack yang dipakai sekarang |
|---|---|---|---|
| `dfine-n-960x540-stride-8` (sekarang) | IDF1 **0,901**, 59 ID switch | 0,823, 105 | 0,764, 210 |
| `dfine-n-1248x704-commercial-v3` | IDF1 **0,828**, 76 ID switch | 0,766, 108 | 0,649, 223 |

**Dengan label ID manusia** (9 klip PT SRP, pipeline hen-detect-ppe dengan Lite terpasang, detektor dan frame sama, 3 FPS): IDF1 **0,743** vs 0,644 untuk ByteTrack yang dipakai sekarang, ID switch 30 vs 47. Di klip klien bersih IDF1 setara (0,785 vs 0,786), tapi orang yang terlewat turun dari 46 ke 17. Di label manusia, Lite bawaan lebih baik daripada Lite tuned. Rinciannya di [docs/EVALUASI_GROUND_TRUTH.md](docs/EVALUASI_GROUND_TRUTH.md).

Bandingkan antar tracker di baris yang sama saja. Antar detektor tidak sebanding, karena referensinya juga dibuat dari detektor masing-masing. Di MOT17 (label manusia), Lite setara tracker lain di FPS asli dan unggul di 2 sampai 5 FPS. Rinciannya di [docs/TRACKING_SUMMARY.md](docs/TRACKING_SUMMARY.md). Penilaian terhadap label ID manusia di klip PT SRP ada di notebook 05.

Inti tracker hanya butuh **numpy dan scipy**, ditulis sendiri, dan boleh dipakai komersial. Lisensi semua tracker yang dibandingkan ada di [docs/TRACKER_LISENSI_DAN_SUMBER.md](docs/TRACKER_LISENSI_DAN_SUMBER.md).

## Mulai

```bash
git clone https://github.com/anarusdiati/object-tracking.git
cd object-tracking
uv sync                      # membuat .venv dengan semua paket notebook
uv run pytest                # tes cepat
```

Lalu buka `notebooks/01_mengenal_lite_tracker.ipynb` di VS Code dengan kernel `.venv (object-tracking)`. Notebook ini tidak butuh data. Untuk notebook 02, 04, dan 05, taruh model detektor dulu (lihat [models/README.md](models/README.md)).

## Pakai di pipeline

```python
from object_tracking import PersonTracker

trk = PersonTracker.tuned()                   # satu instance per kamera, parameter "Lite tuned" untuk detektor sekarang
for t, frame in kamera:                        # t = waktu frame dalam detik
    boxes, scores = detektor_orang(frame)      # Nx4 xyxy dan N skor, SEMUA orang dengan skor >= 0,05
    for p in trk.update(boxes, scores, t):
        p.id, p.box, p.det_idx                 # det_idx = indeks deteksi input (untuk menyambung hasil helm/vest)
```

Untuk pipeline utama (hen-detect-ppe), `LiteForComplianceTracker` bisa langsung menggantikan BYTETracker di dalam `ComplianceTracker`. Cukup salin `lite.py` dan `integration.py`, lalu ikuti [docs/INTEGRASI.md](docs/INTEGRASI.md).

## Isi repo

```
src/object_tracking/
  lite.py            Lite tracker (inti, numpy + scipy)            <- dipasang ke produksi
  integration.py     PersonTracker, CameraTrackers, LiteForComplianceTracker, parameter Lite tuned  <- dipasang ke produksi
  boxes.py           struktur box bersama
  zoo.py             satu antarmuka untuk Lite dan tracker Ultralytics (pembanding), preset dan hasil tuning
  tune.py            tuning parameter tracker dari cache deteksi
  reference_eval.py  evaluasi tanpa label manusia (dibandingkan dengan tracker referensi)
  metrics.py         HOTA, IDF1, MOTA standar (hasilnya sama persis dengan TrackEval)
  mot17.py, mot17_video.py   benchmark dan video perbandingan di MOT17
  synthetic.py       adegan CCTV sintetis dengan label ID, untuk belajar dan tes
  ground_truth.py    baca label (frozen JSONL, CVAT, MOT), cek dan geser label, nilai run pipeline, tolok ukur detektor
  detector.py        detektor D-FINE ONNX (CPU atau CoreML)
  video_io.py        baca dan tulis video
  live_compare.py, video_compare.py, compare_app.py   video grid perbandingan dan aplikasi Gradio
scripts/
  run_pipeline_lite.py   jalankan pipeline hen-detect-ppe dengan Lite (tanpa mengubah kodenya) di klip ground truth
  cari_klip_kandidat.py  cari potongan video ramai dan banyak orang tertutup untuk klip GT baru
  siapkan_klip_gt.py     potong klip 15 fps dan buat pre-label CVAT (D-FINE + Lite)
  analisis_kamera.py     ukuran orang per kamera dan box yang diam (untuk uji resolusi tinggi dan mask)
configs/tracker_tuned.json   parameter hasil tuning semua tracker
notebooks/           01 pengenalan, 02 tuning CCTV, 03 benchmark MOT17, 04 aplikasi perbandingan video, 05 evaluasi ground truth
docs/                penjelasan Lite tracker, hasil, lisensi, panduan integrasi, ilustrasi 10 langkah
tests/               tes metrik, integrasi, dan video perbandingan
```

## Dependency

| Bagian | Paket | Catatan |
|---|---|---|
| Inti (`lite.py`, `integration.py`) | numpy, scipy | Satu-satunya yang dipasang ke produksi |
| `video` | opencv, onnxruntime, pandas | Detektor dan baca video |
| `app` | gradio | Aplikasi perbandingan |
| `compare` | ultralytics, lapx | Tracker pembanding. **AGPL-3.0**: hanya untuk evaluasi, jangan ikut dikirim ke klien |

`uv sync` memasang semuanya untuk eksplorasi. Proyek lain yang hanya butuh tracker cukup memasang paket intinya.

## Asal kode

Dipindahkan dari repo eksplorasi `nana-detect-ppe` (notebook 01 bagian tracking, notebook 02 MOT17, notebook 03 aplikasi). Algoritma Lite terinspirasi dari SORT, ByteTrack, C-BIoU, dan OC-SORT. Kodenya ditulis ulang, tidak disalin dari repo mana pun. Referensi paper ada di [docs/TRACKER_LISENSI_DAN_SUMBER.md](docs/TRACKER_LISENSI_DAN_SUMBER.md).
