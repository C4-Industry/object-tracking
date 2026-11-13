# Model

File model tidak masuk git (lihat `.gitignore`). Daftar dan checksum ada di `MANIFEST.json`.

| File | Dipakai untuk | Wajib? |
|---|---|---|
| `dfine-n-960x540-stride-8.onnx` | Detektor orang yang dipakai sekarang (notebook 02, 04, 05). Parameter `Lite tuned` terbaru disetel dengan detektor ini | Untuk notebook 02, 04, 05 |
| `dfine-n-stg2-960.onnx` | Detektor yang dipakai pipeline hen-detect-ppe sekarang. Dipakai notebook 05 untuk membandingkan Lite dengan output pipeline memakai detektor yang sama | Untuk notebook 05 |
| `dfine-n-1248x704-commercial-v3.onnx` | Detektor sebelumnya. Parameter Lite untuk detektor ini tetap ada di `LITE_PRESETS` | Tidak |
| `dfine-n-544x960-commercial-v3.onnx` | Detektor lebih cepat, sebagai pembanding | Tidak |

Notebook 01 (pengenalan) dan 03 (MOT17) tidak butuh model.

Setiap detektor punya sebaran skor sendiri, jadi parameter Lite disimpan per detektor (`LITE_PRESETS` di `src/object_tracking/integration.py`).

Cara mendapatkan:
```bash
# detektor 960x540-stride-8: dari S3. Detektor 1248x704 v3: dari repo nana-detect-ppe
cp ~/Documents/GitHub/nana-detect-ppe/models/dfine-n-1248x704-commercial-v3.onnx models/
shasum -a 256 models/*.onnx   # bandingkan dengan MANIFEST.json
```

Untuk detektor versi lain (misalnya yang ada di S3), taruh di folder ini lalu ganti `DET_MODEL` di notebook. Kalau detektor final berbeda, ambang skor tracker sebaiknya dicek ulang (notebook 02).
