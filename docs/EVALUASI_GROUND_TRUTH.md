# Evaluasi dengan ground truth PT SRP

Tanggal: 12 November 2026
Notebook: `notebooks/05_evaluasi_ground_truth.ipynb` (run `20261112_1408`)
Data: `~/PPE - PT SRP/ground truth/` (9 klip utama: v03, v10, v15, v18, dan 5 klip eksternal `sub_*`)
Script: `scripts/run_pipeline_lite.py`, `scripts/cari_klip_kandidat.py`, `scripts/siapkan_klip_gt.py`, `scripts/analisis_kamera.py`

## Ringkasan

1. **Lite sudah dipasang di pipeline hen-detect-ppe dan diuji di 9 klip.** Kode pipeline tidak diubah: tracker Ultralytics diganti Lite saat runtime. Dengan detektor dan frame yang sama, IDF1 naik dari 0,644 ke 0,743, dan ID switch turun dari 47 ke 30.
2. **Di klip klien bersih (v10, v03), IDF1 Lite sama dengan ByteTrack (0,785 vs 0,786).** Tapi orang yang terlewat turun dari 46 ke 17 dan MOTA naik dari 0,774 ke 0,880. Keunggulan IDF1 besar ada di klip bocor (v15, v18) dan klip eksternal.
3. **Deteksi skor rendah ke tracker (tahap 3) belum aman untuk v10.** Di semua klip IDF1 naik ke 0,775, tapi di v10 turun ke 0,721 karena handle hand pallet dan tiang rak terdeteksi sebagai orang dengan skor rendah. Pakai tahap 1 dulu.
4. **Pakai Lite bawaan, bukan Lite tuned.** Di label manusia, Lite bawaan lebih baik untuk kedua detektor (3 FPS, detektor stg2: 0,813 vs 0,681). Tuning notebook 02 terhadap tracker referensi tidak terbawa ke label manusia. Preset `"bawaan"` sudah ditambahkan ke `LITE_PRESETS`.
5. **Masalah terbesar ada di detektor, bukan tracker.** Dengan box label sebagai deteksi, IDF1 Lite 0,956. Di klip klien bersih, hampir semua orang yang terlewat sebenarnya terdeteksi tapi skornya di bawah 0,5, terutama operator forklift di dalam kabin.
6. **Label v10 di folder ini versi lama** (beku 9 November, terlambat 2 frame). Notebook menggesernya otomatis. Minta versi terbaru dari temanmu.

## 1. Lite di dalam pipeline

`scripts/run_pipeline_lite.py` menjalankan pipeline hen-detect-ppe (commit `8aad6db`, detektor `dfine_n_stg2/model.onnx`, sha `78411143`) di klip GT, dengan format output yang sama dengan `run_hen_pipeline.py`. Hasil deteksi disimpan, jadi ketiga run di bawah memakai deteksi yang sama persis di frame yang sama (sampel ffmpeg 3 fps). Perbedaannya hanya tracker.

| Run | Tracker | Orang ke tracker |
|---|---|---|
| `hen_20261112_bytetrack` | ByteTrack (pipeline sekarang) | skor 0,5 ke atas |
| `hen_20261112_lite` | Lite bawaan (tahap 1) | skor 0,5 ke atas |
| `hen_20261112_lite-skorrendah` | Lite bawaan (tahap 3) | skor 0,1 ke atas |

Hasil di 3 FPS (`runs/` di folder ground truth):

| Kelompok | Run | IDF1 | HOTA | MOTA | ID switch | FP | FN | ID dibuat / orang |
|---|---|---|---|---|---|---|---|---|
| klien bersih (v03, v10) | ByteTrack | **0,786** | 0,664 | 0,774 | 7 | 67 | 46 | 22 / 16 |
| | Lite | 0,785 | **0,709** | **0,880** | 8 | 39 | 17 | 20 / 16 |
| | Lite skor rendah | 0,731 | 0,665 | 0,723 | 8 | 127 | 14 | 20 / 16 |
| klien bocor (v15, v18) | ByteTrack | 0,621 | 0,558 | 0,811 | 30 | 28 | 147 | 25 / 8 |
| | Lite | 0,786 | 0,662 | 0,849 | 17 | 19 | 128 | 13 / 8 |
| | Lite skor rendah | **0,869** | **0,734** | **0,911** | 9 | 35 | 52 | 12 / 8 |
| eksternal (5 klip `sub_*`) | ByteTrack | 0,547 | 0,450 | 0,347 | 10 | 85 | 359 | 38 / 28 |
| | Lite | 0,627 | 0,517 | 0,424 | 5 | 68 | 333 | 33 / 28 |
| | Lite skor rendah | **0,655** | **0,549** | **0,425** | 3 | 118 | 288 | 32 / 28 |
| **semua 9 klip** | ByteTrack | 0,644 | 0,556 | 0,662 | 47 | 180 | 552 | 85 / 52 |
| | Lite | 0,743 | 0,633 | **0,727** | 30 | 126 | 478 | 66 / 52 |
| | Lite skor rendah | **0,775** | **0,664** | 0,719 | **20** | 280 | 354 | 64 / 52 |

Klip bocor punya frame di data training detektor, jadi angkanya terlalu optimis untuk detektor. Untuk tracker, perbandingannya tetap adil karena deteksinya sama.

**Kenapa Lite skor rendah buruk di v10.** Dari 127 box salah, sekitar 110 ada di handle hand pallet di kiri lorong (skor 0,2 sampai 0,5). Lite membuat track dari deteksi yakin sesaat, lalu deteksi ragu di benda itu menjaga track tetap hidup. Sebelum tahap 3 dipakai, butuh salah satu dari: penyaring track yang tidak pernah bergerak, mask di tiang rak (bagian 4), atau ambang skor rendah yang lebih tinggi (0,3).

**v03 hampir tidak berarti untuk tracker**: hanya 1 orang (operator forklift), dan detektor hanya memberinya skor di atas 0,5 di sekitar 15 sampai 20 dari 308 frame. Jadi angka "klien bersih" hampir seluruhnya dari v10.

### Sweep FPS di v10 (IDF1)

| Run | 1 | 2 | 3 | 5 | 10 | 15 |
|---|---|---|---|---|---|---|
| ByteTrack (`hen_20261112_bytetrack`) | 0,612 | **0,823** | **0,780** | 0,757 | **0,770** | **0,826** |
| Lite (`hen_20261112_lite`) | **0,862** | 0,803 | 0,778 | **0,770** | 0,728 | 0,801 |
| Lite skor rendah | 0,843 | 0,789 | 0,721 | 0,722 | 0,709 | 0,823 |
| BE (`be_20261110`, detektor lain) | 0,717 | 0,886 | 0,938 | 0,767 | 0,859 | 0,777 |

- Di satu klip, IDF1 tracker naik turun cukup jauh (ByteTrack di v18: 0,535 di run temanmu, 0,414 di run ulang dengan versi onnxruntime yang sedikit berbeda). Selisih di bawah 0,05 di satu klip belum berarti.
- Lite paling unggul di 1 FPS (0,862 vs 0,612). MOTA Lite lebih tinggi di semua FPS (0,80 sampai 0,88 vs 0,64 sampai 0,83), karena lebih sedikit orang yang terlewat.
- BE di 3 FPS (0,938) lebih tinggi dari semua run hen. BE memakai detektor lain (`dfine-n-stg2-960lb`, sha `8db65e31`) dengan NMS 0,7, dan DetA-nya jauh lebih tinggi (0,84 vs 0,69 sampai 0,80). Jadi selisih ini kemungkinan besar dari detektor. **Langkah berikutnya: minta file detektor BE, lalu jalankan Lite dengan detektor itu.**

## 2. Tracker di dalam notebook (tanpa pipeline)

Bagian 3 notebook 05, semua klip, detektor `dfine-n-stg2-960`, IDF1:

| Tracker | 15 FPS | 5 | 3 | 2 |
|---|---|---|---|---|
| **Lite bawaan** | 0,810 | 0,791 | **0,813** | **0,771** |
| Lite tuned | 0,681 | 0,698 | 0,681 | 0,704 |
| TrackTrack tuned | **0,839** | 0,716 | 0,682 | 0,655 |
| OC-SORT tuned | 0,790 | **0,819** | 0,700 | 0,690 |
| ByteTrack tuned | 0,775 | 0,767 | 0,637 | 0,623 |
| ByteTrack (sekarang) | 0,683 | 0,681 | 0,582 | 0,567 |

Lite bawaan unggul jelas di 2 sampai 3 FPS. Dengan detektor sempurna (box label), IDF1 Lite bawaan 0,956 dan ByteTrack (sekarang) 0,898.

## 3. Tolok ukur detektor (bagian 6 notebook 05)

Aturan tetap (versi 1): frame berlabel di grid 3 fps, IoU 0,5, tanpa NMS, orang di bawah 60 px dan area abaikan tidak dinilai. Riwayat semua detektor: `experiments/05_ground_truth/tolok_ukur_detektor.csv`.

| Kelompok | Detektor | Recall @0,5 | Presisi | AP50 | Recall @0,3 |
|---|---|---|---|---|---|
| klien bersih | `dfine-n-stg2-960` | 0,666 | 0,947 | **0,994** | 0,817 |
| | `dfine-n-960x540-stride-8` | 0,696 | 0,919 | 0,982 | 0,986 |
| klien bocor | `dfine-n-stg2-960` | **0,926** | 0,959 | 0,974 | 0,958 |
| | `dfine-n-960x540-stride-8` | 0,749 | 0,943 | 0,946 | 0,837 |
| eksternal | `dfine-n-stg2-960` | 0,614 | 0,828 | 0,689 | 0,707 |
| | `dfine-n-960x540-stride-8` | 0,594 | 0,711 | 0,611 | 0,654 |

- **Klien bersih: AP50 0,99 tapi recall 0,67.** Orangnya ditemukan, tapi skornya di bawah 0,5. Dari 271 orang yang terlewat, 270 berstatus "skor rendah", hampir semuanya operator forklift di v03 (di dalam kabin, sebagian tertutup rangka). Di v10 sendiri recall 0,981.
- **Pilihan perbaikan:** ambang orang lebih rendah khusus untuk tracker (tahap 3, setelah masalah benda diam di v10 beres), atau tambah data training operator forklift di kabin. Menurunkan ambang alert tidak disarankan sebelum presisinya dicek.
- **Eksternal:** sebab utama "skor rendah" (129) dan "tertutup" (80). Box salah terbanyak "objek diam berulang" (mesin dan tiang di pabrik).
- `dfine-n-stg2-960` lebih baik di klip klien bocor (v15: recall 0,916 vs 0,638). Detektor ini yang dipakai pipeline sekarang.

## 4. Kamera untuk uji resolusi tinggi dan mask area

Dari pindaian 1 frame per detik di 22 video (18 nomor video, `outputs/analisis_kamera/per_kamera.csv`). Satu sudut kamera muncul di beberapa nomor video: v04 = v14, v05 = v16, v03 = v13, v07 = v18, v08 = v12, v09 = v11, v02 = v17 (sudut berbeda sedikit).

### Uji resolusi tinggi: halaman luar

Detektor 960 mengecilkan frame 2688 px jadi 960 px (x0,36). Porsi orang terdeteksi yang tingginya di bawah 48 px di input detektor:

| Kamera | Resolusi | Tinggi median (px input) | Di bawah 48 px |
|---|---|---|---|
| v17, v02 (halaman kontainer) | 2688x1520 | 57 sampai 58 | **32 sampai 44%** |
| v14, v04, v01 (halaman terpal biru) | 2688x1520 | 54 sampai 62 | **33 sampai 38%** |
| v05, v16 (halaman, forklift dekat kamera) | 2688x1520 | 71 | 19% |
| gudang indoor (v07, v08, v09, v11, v12, v18) | 2560x1440 | 59 sampai 74 | 0 sampai 6% |
| v10 (lorong rak, 4K) | 3840x2160 | 131 | 1% |

Angka ini hanya dari orang yang **terdeteksi**, jadi orang yang lebih kecil dan terlewat tidak terhitung. Porsi sebenarnya lebih besar.

**Rekomendasi:** uji resolusi tinggi di **halaman kontainer (v17/v02)** dulu, lalu **halaman terpal (v14/v04)**. Caranya bisa detektor dengan input lebih besar, atau frame dipotong 2x2 lalu tiap potongan dideteksi (orang jadi 2x lebih besar di input). Gudang indoor dan v10 tidak perlu.

Untuk mengukurnya butuh label orang kecil. GT sekarang mengabaikan orang di bawah 60 px dan belum ada klip GT dari halaman luar. Dua klip kandidatnya sudah disiapkan (bagian 5).

### Mask area: hanya lorong rak v10

Box "orang" yang muncul di tempat yang sama hampir sepanjang video ditemukan di 16 tempat. Semua sudah dicek dengan gambar: **semuanya orang sungguhan** (operator forklift yang duduk, orang berdiri saat briefing). Mask di kamera-kamera itu justru akan menyembunyikan operator forklift, jadi jangan dipasang.

Satu-satunya benda yang berulang kali terdeteksi sebagai orang ada di **kamera v10 (lorong rak, 4K)**:

- **Tiang rak dan pagar oranye di ujung lorong**, sekitar x 2190 sampai 2310, y 0 sampai 300 (piksel 4K). Skornya kebanyakan 0,2 sampai 0,33, tapi di video lain (v10_momen_001 detik 137 sampai 147) sempat di atas 0,5. Tidak ada orang berlabel di area ini sepanjang klip GT v10. Mask sempit di tiang ini layak diuji. Orang yang berjalan di belakang rak tetap terlihat di kiri dan kanan tiang.
- **Handle hand pallet** di kiri lorong (skor 0,2 sampai 0,5). Benda ini berpindah, jadi mask tidak cocok. Solusinya penyaring track yang tidak pernah bergerak, atau tetap memakai ambang 0,5 untuk tracker.

## 5. Klip GT baru (ramai, tertutup, orang kecil)

Disiapkan di `~/PPE - PT SRP/ground truth/kandidat_baru/` (lihat README di folder itu): 7 klip 60 detik, masing-masing dengan video 15 fps dan pre-label CVAT dari D-FINE + Lite.

| Klip | Untuk | Rata-rata orang | Maks |
|---|---|---|---|
| `v10_momen_001_0083-0143s` | ramai (lorong rak 4K) | 4,4 | 8 |
| `v16_momen_010_0255-0315s` | orang dempet, forklift | 3,6 | 6 |
| `v12_momen_001_0017-0077s` | rombongan, orang agak kecil | 5,0 | 6 |
| `v17_momen_003_0120-0180s` | uji resolusi tinggi | 4,2 | 6 |
| `v05_momen_010`, `v08_momen_001`, `v14_momen_003` | cadangan (sudut sama dengan klip di atas) | | |

Klip GT sekarang hanya 1 sampai 5 orang per klip. Pre-label baru tidak punya pergeseran 2 frame (frame 5k tepat), dan ID-nya disambung Lite dengan deteksi skor 0,1 ke atas. **Sebelum dipakai:** cek kebocoran ke `combined_dataset`, dan untuk klip resolusi tinggi labeli orang sampai sekitar 30 px.

## 6. Masalah data yang ditemukan

- **Label v10 di folder ini versi lama**: beku 9 November, hanya tiap 5 frame, dan box terlambat 2 frame (IoU dengan deteksi tertinggi di +2: 0,902 vs 0,856 di 0). README hen-detect-ppe menyebut v10 sudah di-freeze ulang dengan `--shift-frames 2`. Notebook 05 sekarang mendeteksi dan menggeser label otomatis (`GESER_LABEL_OTOMATIS`), tapi sebaiknya ambil `frozen/` terbaru dari Dropbox.
- **Evaluasi run pipeline di notebook 05 sebelumnya salah memetakan frame.** Sampel ffmpeg ke-k pada 3 fps adalah frame 5k + 2, bukan 5k. `load_run` sekarang membaca `config.sampler` dari `run.json` (sama dengan `evaluate.py` temanmu). Tabel "pipeline vs Lite" lama juga mencampur run dengan klip yang berbeda. Keduanya sudah diperbaiki, jadi angka "Lite 0,87 sampai 0,90 vs pipeline 0,53 di v18" dari 11 November tidak dipakai lagi.
- **`sub_5_te5`**: beberapa label di frame 135 sampai 155 terlihat seperti di atas benda (bukan orang) di galeri FN. Perlu dicek di CVAT.

## Cara mengulang

```bash
cd ~/Documents/GitHub/object-tracking
# 1. pipeline + Lite di semua klip GT (butuh repo hen-detect-ppe dan .venv-nya, atau .venv repo ini + ultralytics)
python scripts/run_pipeline_lite.py --run hen_<tanggal>_bytetrack --tracker bytetrack \
    --hen-repo ../hen-detect-ppe --gt-root "~/PPE - PT SRP/ground truth"
python scripts/run_pipeline_lite.py --run hen_<tanggal>_lite --tracker lite \
    --hen-repo ../hen-detect-ppe --gt-root "~/PPE - PT SRP/ground truth"
# 2. notebook 05 (membaca semua run di runs/, termasuk yang baru)
# 3. kandidat klip GT dan analisis kamera
python scripts/cari_klip_kandidat.py --videos ~/Dataset_Momen_PPE_Anonim --log momen_log.csv --min-persons 4
python scripts/analisis_kamera.py --scan outputs/kandidat_gt --videos ~/Dataset_Momen_PPE_Anonim --out outputs/analisis_kamera
python scripts/siapkan_klip_gt.py --videos ~/Dataset_Momen_PPE_Anonim --out "~/PPE - PT SRP/ground truth/kandidat_baru" \
    --clip v16_momen_010.mp4:255:315
```
