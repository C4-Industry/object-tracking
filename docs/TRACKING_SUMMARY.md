# Ringkasan tracker orang untuk CCTV FPS rendah

Tanggal: 9 November 2026
Notebook: `notebooks/02_tuning_cctv.ipynb` (aslinya notebook 01 di repo nana-detect-ppe, bagian B3 dan C)
Kode: `src/object_tracking/zoo.py` (semua tracker), `tune.py` (tuning), `lite.py` (tracker lite)
Hasil tuning: `configs/tracker_tuned.json`
Hasil mentah: repo nana-detect-ppe, `experiments/notebook01/20261109_1427_helm_tracking/C4_tracking_test_*.csv`

## Ringkasan

| Peringkat | Tracker | IDF1 | ID switch | ID dibuat (referensi 196) | Catatan |
|---|---|---|---|---|---|
| 1 | **Lite tuned** | **0,828** | **76** | 170 | Tracker sendiri (`tracker_lite`) + tuning |
| 2 | Lite BIoU | 0,802 | 86 | 175 | Tracker sendiri, setting bawaan |
| 3 | **ByteTrack tuned** | **0,766** | 108 | 162 | Tracker standar terbaik |
| 4 | OC-SORT | 0,754 | 121 | 183 | Setting bawaan |
| 5 | BoT-SORT | 0,754 | 135 | 200 | Setting bawaan |
| 6 | ByteTrack | 0,751 | 146 | 212 | Setting bawaan + deteksi skor rendah |
| 7 | TrackTrack tuned | 0,743 | 138 | 214 | |
| 8 | FastTrack tuned | 0,741 | 120 | 161 | |
| 9 | OC-SORT tuned | 0,734 | 129 | 183 | Lebih buruk dari bawaannya |
| 10 | BoT-SORT tuned | 0,733 | 117 | 168 | Lebih buruk dari bawaannya |
| 11 | Lite tuned det/2 | 0,701 | 108 | 168 | Detektor tiap 2 frame |
| 12 | **ByteTrack (pipeline sekarang)** | **0,649** | **223** | 271 | Hanya orang skor >= 0,5 |

Nilai = rata-rata 5 klip kamera test x 4 FPS proses (2, 3, 4, 6 FPS). ID switch dan ID dibuat = total.

Kesimpulan singkat:
1. **Pipeline sekarang paling lemah** (IDF1 0,649). Penyebab utamanya bukan algoritma ByteTrack, tapi orang dengan skor di bawah 0,5 dibuang sebelum masuk tracker. Orang yang setengah tertutup jadi "hilang", lalu muncul lagi dengan ID baru (271 ID untuk 196 orang).
2. **Tracker standar terbaik: ByteTrack tuned** (IDF1 0,766, ID switch turun dari 223 ke 108 dibanding pipeline sekarang). Paling mudah dipasang karena tetap memakai ByteTrack yang sudah ada di pipeline; hanya parameternya yang berubah.
3. **Terbaik secara keseluruhan: Lite tuned** (IDF1 0,828). Unggul paling jauh di FPS rendah (2 sampai 4 FPS), karena gerak dihitung per detik dan ada tahap matching ulang dari posisi terakhir yang terlihat (OCR).
4. **Di 6 FPS semua tracker yang baik hampir sama** (IDF1 0,85 sampai 0,87). Pilihan tracker paling penting kalau FPS per kamera rendah.
5. **Jangan menjalankan detektor tiap 2 frame.** IDF1 turun dari 0,828 ke 0,701. Lebih baik turunkan FPS proses dan tetap deteksi di setiap frame.

## Cara menilai

Belum ada label ID manual, jadi "kebenaran" diambil dari dua tracker kuat yang dijalankan pada **12,5 FPS** (gerak antar frame kecil, sehingga ID-nya relatif stabil): ByteTrack dan OC-SORT, keduanya diberi semua deteksi orang skor >= 0,1. Setiap tracker yang diuji dijalankan pada FPS rendah (2 sampai 6 FPS), lalu dibandingkan dengan kedua referensi di waktu yang sama.

| Metrik | Artinya |
|---|---|
| **IDF1** | Seberapa konsisten ID tiap orang dibanding referensi (1 = sama persis). Rata-rata terhadap dua referensi |
| **ID switch** | Berapa kali ID seseorang berganti |
| **ID dibuat** | Jumlah ID selama klip. Makin dekat ke jumlah ID referensi makin baik; jauh di atasnya berarti orang yang sama dipecah jadi beberapa ID |

Data:
- **Tuning**: 10 klip dari 10 kamera **non-test** (v03, v04, v05, v08, v09, v11, v14, v16, v17, v18), satu klip paling ramai per kamera, 60 detik pertama.
- **Penilaian**: 5 klip dari 5 kamera **test** (v01, v02, v07, v10, v12), sama dengan split test `commercial_combined_v3`. Kamera ini tidak dipakai saat tuning, jadi angka di atas tidak bias karena tuning.
- Semua tracker memakai hasil deteksi yang **sama** (D-FINE-N 1248x704, di-cache), jadi bedanya murni dari tracker.
- Pilihan "terbaik" ditentukan dari skor di klip tuning, bukan dari klip test.

## Tracker yang dicoba

| Tracker | Ide utama | Kelebihan | Kelemahan di CCTV FPS rendah |
|---|---|---|---|
| **ByteTrack** | Matching dua tahap: deteksi skor tinggi dulu, lalu deteksi skor rendah untuk menyambung track yang sudah ada | Sederhana, cepat, sudah dipakai pipeline | Model gerak per frame; di FPS rendah orang bergeser jauh antar frame sehingga box tidak bertumpuk lagi |
| **OC-SORT** | ByteTrack + koreksi berbasis observasi: kecepatan dihitung ulang setelah orang hilang (ORU), matching ulang dari posisi terakhir yang terlihat (OCR), konsistensi arah gerak | Lebih tahan saat orang tertutup sebentar | Bobot arah gerak kurang berguna saat gerak antar frame besar dan tidak mulus |
| **BoT-SORT** | ByteTrack + Kalman yang memperkirakan lebar/tinggi langsung (+ kompensasi gerak kamera dan Re-ID, keduanya dimatikan di sini) | Box lebih pas | Tanpa Re-ID dan dengan kamera diam, praktis sama dengan ByteTrack |
| **TrackTrack** | Matching iteratif dengan banyak biaya (IoU, skor, sudut) dan inisialisasi track yang sadar track lain | Kuat di benchmark MOT 30 FPS | Banyak parameter per frame; bawaannya membuang deteksi < 0,6 sehingga hancur (IDF1 0,652, 420 ID) |
| **FastTrack** | ByteTrack + penanganan oklusi (box diperbesar saat tertutup, track yang baru tertutup bisa ditemukan lagi) | Bagus untuk oklusi singkat di FPS tinggi | Jendela oklusi dalam satuan frame, kurang cocok di FPS rendah |
| **Lite (`tracker_lite`)** | Ditulis untuk proyek ini: Kalman berbasis **detik**, buffered IoU (box diperbesar saat matching), deteksi skor rendah dipakai, box orang dobel dibuang, bisa menebak posisi di frame tanpa deteksi, + opsi ORU/OCR dari OC-SORT | Perilaku sama di FPS berapa pun; paling unggul di 2 sampai 4 FPS | Kode sendiri (perlu dirawat); belum ada Re-ID (disengaja, karena brief melarang identifikasi orang) |

Semua tracker dijalankan tanpa Re-ID dan tanpa kompensasi gerak kamera (CCTV diam). Tracker Ultralytics bekerja dalam satuan frame, jadi batas waktunya (misalnya berapa lama orang hilang tetap diingat) ditulis dalam detik lalu diubah ke jumlah frame sesuai FPS proses.

## ByteTrack tuned: apa yang diubah

Titik awal tuning adalah ByteTrack yang sudah diberi deteksi skor rendah (baris "ByteTrack" di tabel). Tuning memakai 91 kandidat: setting awal, 60 kandidat acak, lalu 30 kandidat di sekitar 5 terbaik.

| Parameter | Arti | Pipeline sekarang | Bawaan Ultralytics | Titik awal tuning | **Hasil tuning** | Ruang pencarian |
|---|---|---|---|---|---|---|
| `person_min` | Deteksi orang di bawah skor ini dibuang sebelum tracker | 0,5 | (tidak ada) | 0,1 | **0,1** | 0,05 / 0,1 / 0,2 / 0,3 |
| `nms_iou` | Box orang dobel (IoU >= nilai ini dengan box yang lebih yakin) dibuang | mati | (tidak ada) | 0,6 | **0,6** | mati / 0,5 / 0,6 / 0,7 |
| `track_high_thresh` | Deteksi >= ini ikut matching tahap 1 | 0,5 | 0,25 | 0,5 | **0,458** | 0,30 sampai 0,65 |
| `track_low_thresh` | Deteksi antara ini dan `high` ikut tahap 2 (menyambung track yang ada) | 0,1 | 0,1 | 0,1 | **0,1** | 0,05 / 0,1 / 0,2 |
| `new_track_thresh` | Skor minimal untuk membuat ID baru | 0,5 | 0,25 | 0,55 | **0,557** | `high` + 0 sampai 0,15 |
| `match_thresh` | Batas biaya matching (makin tinggi makin longgar) | 0,8 | 0,8 | 0,8 | **0,921** | 0,60 sampai 0,95 |
| `track_buffer` | Berapa lama orang yang hilang tetap diingat | 3 s | 30 frame | 3 s | **6,7 s** | 1,5 sampai 8 s |
| `fuse_score` | Skor deteksi digabung ke IoU saat matching | ya | ya | ya | ya (tetap) | |

Yang paling berpengaruh dan alasannya:
1. **Deteksi skor rendah masuk tracker** (`person_min` 0,5 ke 0,1, ditambah `nms_iou` 0,6). Ini perubahan terbesar dibanding pipeline sekarang. Orang yang setengah tertutup forklift, rak, atau orang lain sering terdeteksi dengan skor 0,2 sampai 0,4. Kalau dibuang, track-nya putus. NMS diperlukan karena D-FINE kadang mengeluarkan box orang dobel berskor rendah di atas orang yang sama.
2. **Matching lebih longgar** (`match_thresh` 0,8 ke 0,92). Di ByteTrack Ultralytics, biaya = 1 - (IoU x skor). Dengan 0,92, pasangan diterima selama IoU x skor >= 0,08. Di 2 sampai 4 FPS orang bisa bergeser setengah lebar badan antar frame, jadi tumpang tindih box-nya kecil, tapi pasangannya tetap benar.
3. **Ingatan lebih lama** (`track_buffer` 3 detik ke 6,7 detik). Orang yang tertutup beberapa detik masih bisa disambung ke ID lamanya. Jumlah ID turun dari 212 ke 162.
4. **Ambang tahap 1 sedikit lebih rendah** (0,5 ke 0,458), sementara ambang ID baru tetap sekitar 0,56. Deteksi menengah boleh menyambung track yang ada, tapi tidak boleh membuat ID baru.

Hasil ByteTrack tuned per FPS (klip test):

| FPS proses | 2,1 | 3,1 | 4,2 | 6,3 |
|---|---|---|---|---|
| IDF1 ByteTrack tuned | 0,634 | 0,759 | 0,815 | 0,856 |
| IDF1 pipeline sekarang | 0,577 | 0,600 | 0,680 | 0,741 |
| IDF1 Lite tuned | 0,741 | 0,838 | 0,872 | 0,861 |
| ID switch ByteTrack tuned | 45 | 26 | 21 | 16 |
| ID switch pipeline sekarang | 59 | 67 | 44 | 53 |

Catatan: ID dibuat 162 sedikit di bawah referensi 196. ByteTrack tuned cenderung "menyambung" lebih agresif. Ini lebih baik daripada memecah orang, tapi di area ramai bisa sesekali menukar ID dua orang yang berpapasan. Perlu dicek di aplikasi perbandingan video (notebook 04).

## Lite tuned: apa yang diubah

| Parameter | Arti | Bawaan | **Tuned** |
|---|---|---|---|
| `person_min` / `nms_iou` | Filter deteksi orang / buang box dobel | 0,1 / 0,6 | **0,05 / 0,7** |
| `high_thresh` / `new_thresh` | Ambang tahap 1 / ambang ID baru | 0,50 / 0,55 | **0,517 / 0,535** |
| `min_iou` / `min_iou_low` | IoU minimal tahap 1 / tahap 2 | 0,30 / 0,40 | **0,365 / 0,386** |
| `buffers` | Pembesaran box saat matching (C-BIoU) | 0,1 lalu 0,4 | **0,1 saja** |
| `max_lost_sec` | Ingatan orang yang hilang | 3,0 s | **4,2 s** |
| `std_vel` / `std_pos` | Seberapa bebas Kalman mengubah kecepatan / posisi | 0,6 / 0,05 | **0,89 / 0,10** |
| `lost_velocity_decay_sec` | Setelah hilang selama ini, kecepatan dianggap nol | 1,0 s | **1,46 s** |
| `nsa` | Kalman kurang percaya box skor rendah | ya | **tidak** |
| `ocr` | Matching ulang dari box terakhir yang terlihat (OC-SORT) | tidak | **ya** |
| `oru` | Hitung ulang kecepatan setelah orang hilang (OC-SORT) | tidak | **ya** |

Ablasi (satu opsi dimatikan dari Lite tuned, klip test): **OCR paling penting** (tanpa OCR: IDF1 -0,024, ID switch 76 ke 104). Membuang deteksi skor < 0,5 menurunkan IDF1 0,083. Opsi lain (NSA, DIoU, ORU) pengaruhnya di bawah 0,005.

## Catatan dan batasan

- **Referensi bukan label manusia.** Dua referensi sendiri hanya sepakat 0,89 rata-rata di klip test (0,82 sampai 0,98 per klip), dan di klip ramai seperti v08 dan v18 hanya 0,67 sampai 0,78. Selisih kecil (di bawah sekitar 0,02 IDF1) antar tracker sebaiknya tidak dianggap berarti. Untuk angka mutlak (laporan atau paper) perlu label ID manual di beberapa klip (CVAT), dimulai dari v07 dan v12 yang paling sulit.
- **Tuning bisa overfit.** OC-SORT, BoT-SORT, dan FastTrack membaik di klip tuning tapi memburuk atau hampir sama di klip test. ByteTrack tuned dan Lite tuned membaik di keduanya.
- **Klip test tersulit untuk semua tracker**: v07 (IDF1 Lite tuned 0,72) dan v12 (0,77). Keduanya indoor dengan orang yang sering tertutup.
- Waktu tracker di bawah 1 ms per frame untuk semua tracker, jadi tidak berpengaruh ke kapasitas kamera. Bebannya ada di detektor.

## Rekomendasi

| Kebutuhan | Pilihan |
|---|---|
| Perbaikan cepat di pipeline sekarang, risiko kecil | **ByteTrack tuned**: cukup ubah parameter (tabel di atas) dan berhenti membuang orang skor < 0,5 |
| Hasil terbaik, terutama di 2 sampai 4 FPS per kamera | **Lite tuned** |
| FPS proses | 4 FPS dengan deteksi di setiap frame. Jangan pakai deteksi tiap 2 frame |
