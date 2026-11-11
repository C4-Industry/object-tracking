# Model

File model tidak masuk git (lihat `.gitignore`). Daftar dan checksum ada di `MANIFEST.json`.

| File | Dipakai untuk | Wajib? |
|---|---|---|
| `dfine-n-1248x704-commercial-v3.onnx` | Detektor orang untuk video CCTV sendiri (notebook 02, aplikasi perbandingan di notebook 04). Parameter `Lite tuned` disetel dengan detektor ini | Untuk notebook 02 dan 04 |
| `dfine-n-544x960-commercial-v3.onnx` | Detektor lebih cepat, sebagai pembanding | Tidak |

Notebook 01 (pengenalan) dan 03 (MOT17) tidak butuh model.

Cara mendapatkan:
```bash
cp ~/Documents/GitHub/nana-detect-ppe/models/dfine-n-1248x704-commercial-v3.onnx models/
# cek checksum
shasum -a 256 models/dfine-n-1248x704-commercial-v3.onnx   # bandingkan dengan MANIFEST.json
```

Untuk detektor versi lain (misalnya yang ada di S3), taruh di folder ini lalu ganti `DET_MODEL` di notebook. Kalau detektor final berbeda, ambang skor tracker sebaiknya dicek ulang (notebook 02).
