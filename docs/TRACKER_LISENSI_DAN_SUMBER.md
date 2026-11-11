# Tracker: lisensi, sumber ilmiah, dan cara membaca IDF1

Pelengkap [LITE_TRACKER.md](LITE_TRACKER.md) dan [TRACKING_SUMMARY.md](TRACKING_SUMMARY.md).

## 1. Lisensi tracker yang sudah diuji

Semua tracker pembanding di notebook 02 dan 03 dijalankan lewat library **Ultralytics** (`ultralytics/trackers/*.py`), yang berlisensi **AGPL-3.0**. Yang menentukan boleh atau tidaknya dipakai di produk adalah lisensi kode yang benar-benar kita pakai, bukan hanya lisensi repo asli algoritmanya.

| Tracker | IDF1 (test) | Kode yang dipakai | Lisensi kode itu | Repo asli dan lisensinya | Produk komersial tertutup? |
|---|---|---|---|---|---|
| **Lite tuned** | **0,828** | `src/object_tracking/lite.py` (ditulis sendiri, hanya numpy dan scipy) | Milik proyek ini | Tidak ada, ide dari paper | **Ya** |
| ByteTrack tuned | 0,766 | Ultralytics `byte_tracker.py` | AGPL-3.0 | ifzhang/ByteTrack, MIT | Versi Ultralytics: tidak, kecuali Enterprise License |
| OC-SORT | 0,754 | Ultralytics `oc_sort.py` | AGPL-3.0 | noahcao/OC_SORT, MIT | Sama |
| BoT-SORT | 0,754 | Ultralytics `bot_sort.py` | AGPL-3.0 | NirAharon/BoT-SORT, MIT | Sama |
| TrackTrack tuned | 0,743 | Ultralytics `track_tracker.py` | AGPL-3.0 | kamkyu94/TrackTrack, MIT | Sama |
| FastTrack tuned | 0,741 | Ultralytics `fast_tracker.py` | AGPL-3.0 | Hamidreza-Hashempoor/FastTracker, belum diverifikasi | Sama |
| ByteTrack (sekarang) | 0,649 | Ultralytics, setelan lama | AGPL-3.0 | Sama dengan ByteTrack | Sama |

Catatan:
- Ultralytics mensyaratkan seluruh proyek dibuka sebagai AGPL-3.0, atau membeli Enterprise License. Ini berlaku juga untuk pemakaian internal.
- Di repo ini `ultralytics` hanya ada di extra `compare` (khusus evaluasi). Inti tracker (`lite.py` dan `integration.py`) hanya butuh numpy dan scipy, jadi yang dipasang ke pipeline produksi tidak membawa Ultralytics.
- Pembanding lain dengan lisensi permisif: Roboflow `trackers` (Apache-2.0), berisi SORT, ByteTrack, OC-SORT, BoT-SORT, dan C-BIoU.
- Dokumen ini bukan nasihat hukum. Daftar lisensi semua dependency final (termasuk detektor D-FINE dan bobotnya) perlu dicek pihak legal sebelum serah terima.

## 2. Lite tracker: dibuat sendiri, terinspirasi paper

Lite tidak punya repo atau paper sendiri. Nama "Lite" adalah nama internal proyek ini. Kodenya ditulis dari nol; yang diambil dari paper adalah idenya.

| Langkah Lite | Asal ide | Bedanya dengan aslinya |
|---|---|---|
| 0. Saring deteksi (NMS orang) | Tambahan Lite | Tidak ada di tracker lain |
| 1. Tebak posisi (Kalman) | SORT | Satuan detik, bukan frame |
| 2. Pisah deteksi YAKIN dan RAGU | ByteTrack | Sama |
| 3. Cocokkan YAKIN dengan box diperbesar | C-BIoU | Buffer 10%, satu ukuran |
| 4. Cocokkan RAGU ke track yang belum ketemu | ByteTrack | Juga memakai box diperbesar |
| 5. OCR, cocokkan ke posisi asli terakhir | OC-SORT | Sama idenya |
| 6. Perbarui Kalman + ORU setelah hilang | OC-SORT | Sama idenya |
| 7. Tahan track hilang | ByteTrack (track buffer) | Dalam detik (4,2 s), kecepatan diredam setelah 1,5 s |
| 8. ID baru, tampil setelah 2 kali | SORT (min hits) dan ByteTrack (ambang skor) | Sama |

Tidak dipakai: OCM dari OC-SORT, Re-ID dan kompensasi gerak kamera dari BoT-SORT (kamera statis, seragam pekerja mirip, dan brief melarang identifikasi orang).

Kenapa Lite unggul di data kita: gerak dihitung per detik sehingga tetap benar di 2 sampai 4 FPS, box diperbesar membantu pergeseran antar frame, OCR dan ORU menangani orang tertutup forklift atau rak, NMS membuang box ganda D-FINE, dan semua parameter disetel di data CCTV sendiri.

Kenapa boleh dipakai komersial: kode ditulis sendiri dan hanya bergantung pada numpy dan scipy (BSD). Hak cipta melindungi kode, bukan algoritma. Repo asli paper-paper di atas pun berlisensi MIT. Paten belum dicek.

Kekurangannya: belum diuji komunitas dan belum dinilai di benchmark publik. Langkah yang disarankan sebelum produksi: unit test untuk kasus inti, label ID manual di 2 sampai 3 klip, dan pembanding C-BIoU dari Roboflow `trackers`.

## 3. Cara membaca IDF1

IDF1 mengukur seberapa konsisten setiap orang memegang ID yang benar sepanjang video. Setiap ID dari tracker dipasangkan ke satu orang referensi, lalu dihitung porsi kemunculan yang ID-nya benar. IDF1 0,83 berarti kira-kira 83% dari waktu, setiap orang memegang ID yang benar. Metrik ini menghukum ID yang tertukar dan orang yang tidak terlacak.

Kolom lain di tabel hasil:
- **ID switch**: berapa kali ID seseorang berganti. Makin rendah makin baik.
- **Cakupan**: porsi kemunculan orang yang diberi ID.
- **ID dibuat vs ID referensi**: jauh di atas referensi berarti tracker terlalu sering membuat ID baru untuk orang yang sama.

Penting: referensi di evaluasi ini adalah tracker yang dijalankan pada FPS tinggi (12,5 FPS), bukan label manusia. Angkanya valid untuk membandingkan tracker, belum untuk akurasi absolut.

## Referensi

- Bewley dkk., *Simple Online and Realtime Tracking* (SORT), ICIP 2016. arXiv:1602.00763
- Zhang dkk., *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*, ECCV 2022. arXiv:2110.06864
- Yang dkk., *Hard to Track Objects with Irregular Motions and Similar Appearances? Make It Easier by Buffering the Matching Space* (C-BIoU), WACV 2023. arXiv:2211.14317
- Cao dkk., *Observation-Centric SORT: Rethinking SORT for Robust Multi-Object Tracking*, CVPR 2023. arXiv:2203.14360
- Ristani dkk., *Performance Measures and a Data Set for Multi-Target, Multi-Camera Tracking* (IDF1), ECCV 2016 Workshops. arXiv:1609.01775
- Lisensi Ultralytics: https://www.ultralytics.com/license
- Roboflow trackers (Apache-2.0): https://github.com/roboflow/trackers
