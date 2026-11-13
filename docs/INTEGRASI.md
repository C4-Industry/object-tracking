# Memasang Lite tracker ke pipeline utama (hen-detect-ppe)

Dokumen ini menjelaskan cara mengganti tracker Ultralytics (ByteTrack) di pipeline utama dengan Lite tracker.

> **Cek dulu versi kodenya.** Langkah di bawah disusun dari `src/detect/pipeline/tracking.py` di hen-detect-ppe, branch `feat/sagemaker-training`, commit `ddfc73d` (5 November 2026). Kalau versi final sudah berubah, nama fungsi dan barisnya bisa berbeda, tapi idenya sama.

## Sudah diuji dengan ground truth (12 November 2026)

Pipeline hen-detect-ppe dengan Lite terpasang sudah dijalankan di 9 klip GT, tanpa mengubah kode pipeline (`scripts/run_pipeline_lite.py`). Dengan detektor dan frame yang sama, 3 FPS:

| Tracker di pipeline | IDF1 (9 klip) | ID switch | IDF1 klien bersih (v03, v10) | MOTA klien bersih |
|---|---|---|---|---|
| ByteTrack (sekarang) | 0,644 | 47 | 0,786 | 0,774 |
| Lite bawaan, tahap 1 | 0,743 | 30 | 0,785 | 0,880 |
| Lite bawaan, tahap 1 + 3 | 0,775 | 20 | 0,731 | 0,723 |

Kesimpulannya: **pasang tahap 1 dan 2 dengan Lite bawaan** (`detector="bawaan"`). Tahap 3 tunggu dulu: di v10 deteksi skor rendah membuat handle hand pallet ikut dilacak sebagai orang. Rincian: [EVALUASI_GROUND_TRUTH.md](EVALUASI_GROUND_TRUTH.md).

## Ringkasan

| Tahap | Perubahan | Dampak |
|---|---|---|
| 1 | Tambah pilihan `tracker_type="lite"` di `ComplianceTracker`, kirim `timestamp_sec` ke tracker | Lite aktif, perilaku alert tidak berubah |
| 2 | Samakan batas waktu hapus status orang dengan Lite (3 detik menjadi 4,5 detik) | Status helm/vest tidak hilang saat orang tertutup lebih dari 3 detik |
| 3 | Berikan deteksi orang skor rendah (0,05 sampai 0,5) ke tracker, tanpa dipakai untuk alert | IDF1 naik sekitar 0,07 sampai 0,08 di klip test |
| 4 | Impor Ultralytics dibuat malas (hanya saat dipakai), lalu hapus dari dependency produksi | Lepas dari lisensi AGPL-3.0 |

Tahap 1 dan 2 sudah cukup untuk mulai mencoba. Tahap 3 memberi peningkatan terbesar. Tahap 4 wajib sebelum diserahkan ke klien.

## File yang disalin

Hanya dua file, dan keduanya hanya butuh numpy dan scipy:

```
object-tracking/src/object_tracking/lite.py          -> hen-detect-ppe/src/detect/pipeline/lite_tracker/lite.py
object-tracking/src/object_tracking/integration.py   -> hen-detect-ppe/src/detect/pipeline/lite_tracker/integration.py
(file kosong)                                        -> hen-detect-ppe/src/detect/pipeline/lite_tracker/__init__.py
```

`integration.py` sudah berisi parameter `Lite tuned` per detektor (`LITE_PRESETS`), jadi tidak perlu menyalin folder `configs/`. Yang dipakai bawaan adalah `LITE_TUNED_DETECTOR` (sekarang `dfine-n-960x540-stride-8`). Kalau pipeline memakai detektor lain yang sudah punya preset, berikan namanya: `LiteForComplianceTracker(fps=self.fps, detector="dfine-n-1248x704-commercial-v3")`.

## Tahap 1: pilihan `tracker_type="lite"`

Di `src/detect/pipeline/tracking.py`, kelas `ComplianceTracker`:

**a. `_build_tracker()`**, tambahkan di awal fungsi:

```python
def _build_tracker(self):
    t_type = self.tracker_type
    if t_type == "lite":
        from .lite_tracker.integration import LiteForComplianceTracker
        return LiteForComplianceTracker(fps=self.fps, detector="bawaan")   # Lite tanpa tuning (terbaik di label manusia)
    ...  # cabang bytetrack, ocsort, dst. tetap seperti semula
```

**b. `update()`**, di bagian yang memanggil tracker (langkah 2 di kode aslinya):

```python
# 2. Run Tracker -> returns array (N, 8): [x1, y1, x2, y2, track_id, score, cls, idx]
if self.tracker_type == "lite":
    online_targets = self.tracker.update(tracker_input, timestamp_sec=timestamp_sec)
elif self.tracker_type in ("botsort", "bot_sort", "deepocsort", "deep_oc_sort"):
    online_targets = self.tracker.update(tracker_input, img=frame)
else:
    online_targets = self.tracker.update(tracker_input)
```

Format output `LiteForComplianceTracker` sama persis dengan BYTETracker: array N x 8 `[x1, y1, x2, y2, track_id, score, cls, idx]`, dengan `idx` = indeks di `associated_persons`. Jadi sisa `update()` (state machine helm/vest, alert) tidak perlu diubah.

**c. Config**: set tipe tracker ke `lite` di config pipeline (di repo nana-detect-ppe namanya `tracker.type`).

**Penting soal waktu.** Lite menghitung gerak dalam detik. `timestamp_sec` harus waktu frame yang sebenarnya (misalnya `frame_idx / fps_sumber` atau jam kamera), bukan nomor frame yang diproses. Ini yang membuat Lite tetap benar saat hanya 4 dari 25 frame per detik yang diproses. Kalau `timestamp_sec` tidak dikirim, Lite memakai `frame_idx / fps` dan memberi peringatan, yang hanya benar kalau setiap frame diproses.

## Tahap 2: batas waktu hapus status orang

Di akhir `ComplianceTracker.update()`, status orang (riwayat helm/vest, timer pelanggaran) dihapus kalau orangnya tidak terlihat lebih dari 3 detik:

```python
# 4. Clean up expired tracks (inactive for > 3.0 seconds)
... if (timestamp_sec - tp.last_seen_sec) > 3.0 and tid not in matched_target_ids
```

Lite mengingat orang yang hilang sampai sekitar 4 detik (`max_lost_sec`: 3,9 detik untuk detektor 960x540, 4,2 detik untuk detektor 1248x704). Kalau batasnya tetap 3 detik, orang yang tertutup forklift 3,5 detik kembali dengan ID yang sama, tapi status helmnya mulai dari nol lagi. Ganti `3.0` dengan nilai yang sedikit di atas `max_lost_sec`, misalnya `4.5`.

## Tahap 3: deteksi orang skor rendah

> **Belum disarankan untuk produksi.** Di 9 klip GT, tahap 3 menaikkan IDF1 total (0,743 ke 0,775) tapi menurunkan v10 (0,778 ke 0,721), karena deteksi skor rendah di handle hand pallet menjaga track palsu tetap hidup. Perlu penyaring track yang tidak pernah bergerak, atau ambang skor rendah 0,3, lalu uji ulang.

Di klip test CCTV, Lite tuned dengan semua deteksi (skor 0,05 ke atas) mendapat IDF1 **0,901** dengan detektor 960x540. Kalau hanya diberi orang dengan skor 0,5 ke atas, IDF1 turun ke **0,830**. Dengan detektor 1248x704 polanya sama: 0,828 turun ke 0,745. Orang yang setengah tertutup sering terdeteksi dengan skor rendah, dan deteksi ragu itulah yang menyambung ID-nya.

Sekarang engine menyaring orang dengan ambang `person` (0,5) **sebelum** asosiasi PPE, jadi tracker tidak pernah melihat deteksi skor rendah. Perubahannya:

1. Di engine, ambil orang dengan skor 0,05 ke atas untuk tracker, dan tetap pakai ambang 0,5 untuk keputusan helm/vest.
2. Asosiasikan PPE ke semua orang tersebut (atau hanya ke yang skornya 0,5 ke atas, lalu yang lain diberi `has_helmet_seen=False` dan `pose_vis` dengan `head_visible=False`).
3. Di `ComplianceTracker.update()`, langkah 3 (update state machine), lewati orang yang skornya di bawah ambang alert, supaya deteksi ragu tidak memicu alert:

```python
for tp, ap in matched_results:
    if ap.person_box.conf < self.alert_person_thresh:    # mis. 0.5, sama dengan ambang person sekarang
        continue                                          # tetap dilacak, tapi tidak mengubah status helm/vest
    ...
```

Cara ini membuat deteksi ragu hanya membantu menjaga ID, tanpa memengaruhi status dan alert.

## Tahap 4: lepas dari Ultralytics

`tracking.py` mengimpor tracker Ultralytics di bagian paling atas file, jadi Ultralytics (AGPL-3.0) tetap ikut terpasang walau yang dipakai Lite. Pindahkan impornya ke dalam cabang masing-masing di `_build_tracker()`:

```python
elif t_type in ("bytetrack", "byte", "pose_guided"):
    from ultralytics.trackers.byte_tracker import BYTETracker
    ...
```

Setelah itu `ultralytics` bisa dihapus dari dependency produksi. Cek juga bagian lain pipeline yang mungkin masih mengimpor Ultralytics (misalnya model YOLO pembanding).

## Yang belum diuji

- **`pose_guided`** (box diganti titik tengah badan dari RTMPose) belum pernah dicoba bersama Lite. Mulai tanpa opsi ini.
- **Ambang skor.** Parameter Lite tuned sekarang disetel untuk `dfine-n-960x540-stride-8.onnx` (preset untuk `dfine-n-1248x704-commercial-v3.onnx` tetap ada). Kalau detektor final berbeda lagi (misalnya versi int8), sebaran skornya bisa bergeser. Jalankan ulang notebook 02 dengan detektor final, tambahkan hasilnya ke `LITE_PRESETS`, lalu cek dengan video ground truth (notebook 05).

## Cara mengecek setelah dipasang

1. **Tes cepat**: `tests/test_integration.py` di repo ini memeriksa format output dan indeks deteksi. Adaptor ini juga sudah dicoba bersama `ComplianceTracker` dari salinan kode pipeline utama: ID dua orang tetap stabil, dan alert helm muncul untuk orang yang tepat.
2. **Klip GT, dua tracker, tanpa mengubah kode pipeline**: `scripts/run_pipeline_lite.py` menjalankan pipeline hen-detect-ppe di semua klip GT dan menulis `runs/<run>/` dengan format yang sama dengan `run_hen_pipeline.py`. Hasil deteksi disimpan, jadi run kedua dan seterusnya hanya menjalankan tracker:
   ```bash
   python scripts/run_pipeline_lite.py --run hen_<tanggal>_bytetrack --tracker bytetrack --hen-repo ../hen-detect-ppe --gt-root "~/PPE - PT SRP/ground truth"
   python scripts/run_pipeline_lite.py --run hen_<tanggal>_lite --tracker lite --hen-repo ../hen-detect-ppe --gt-root "~/PPE - PT SRP/ground truth"
   ```
3. **Nilai**: notebook 05 bagian 4 (atau `evaluate.py` temanmu) membaca semua run di `runs/`. Setelah Lite dipasang permanen di pipeline, jalankan `run_hen_pipeline.py --tracker lite` dan bandingkan dengan run di atas: angkanya harus sama.
4. **Kecepatan**: Lite di bawah 1 ms per frame per kamera, jadi beban 12 kamera tetap ditentukan detektor dan decode video.
