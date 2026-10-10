# Kenali-Batikmu-ML

Model *machine learning* untuk **klasifikasi motif batik Nusantara** dari foto. Model dilatih dengan *transfer learning* **MobileNetV2** (TensorFlow/Keras) dan diekspor ke format **TensorFlow Lite** agar ringan dan siap dipakai di aplikasi mobile.

- **20 kelas** motif batik
- **Input:** gambar RGB 224 × 224
- **Output:** probabilitas 20 kelas (softmax)
- **Akurasi:** ±95,7% pada data validasi + test (900 gambar)
- **Ukuran model:** ±9 MB (`.tflite`)

---

## Daftar Isi

- [Struktur Repositori](#struktur-repositori)
- [Kelas Batik](#kelas-batik)
- [Dataset](#dataset)
- [Arsitektur & Pelatihan](#arsitektur--pelatihan)
- [Hasil Evaluasi](#hasil-evaluasi)
- [Cara Menggunakan Model](#cara-menggunakan-model)
- [Menjalankan Ulang Notebook](#menjalankan-ulang-notebook)
- [Keterbatasan](#keterbatasan)

---

## Struktur Repositori

```
Kenali-Batikmu-ML/
├── batik_mobilenetv2.tflite     # Model TFLite hasil konversi
├── labels.txt                   # Daftar 20 label (urutan = indeks output model)
├── model_deteksi_batik.ipynb    # Notebook: preprocessing, training, evaluasi, konversi
└── README.md
```

---

## Kelas Batik

Urutan label pada `labels.txt` **sama persis** dengan urutan indeks output model (alfabetis, sesuai `class_names` dari Keras).

| Idx | Label | Idx | Label |
|----:|-------|----:|-------|
| 0 | `batik_betawi` | 10 | `batik_sidoluhur` |
| 1 | `batik_bokor_kencono` | 11 | `batik_sidomukti` |
| 2 | `batik_buketan` | 12 | `batik_sidomulyo` |
| 3 | `batik_dayak` | 13 | `batik_singa_barong` |
| 4 | `batik_jlamprang` | 14 | `batik_srikaton` |
| 5 | `batik_kawung` | 15 | `batik_tribusono` |
| 6 | `batik_liong` | 16 | `batik_tujuh_rupa` |
| 7 | `batik_mega_mendung` | 17 | `batik_tuntrum` |
| 8 | `batik_parang` | 18 | `batik_wahyu_tumurun` |
| 9 | `batik_sekarjagad` | 19 | `batik_wirasat` |

---

## Dataset

Dataset disimpan dalam struktur folder per kelas (`processed_dataset/`) dan dimuat dengan `tf.keras.utils.image_dataset_from_directory`.

| Split | Folder | Jumlah Gambar |
|-------|--------|--------------:|
| Train | `train_original` | 2.101 |
| Validation | `val` | 600 |
| Test | `test` | 300 |
| **Total** | | **3.001** |

Pada gabungan validation + test, setiap kelas memiliki **45 gambar** (seimbang).

---

## Arsitektur & Pelatihan

**Arsitektur**

```
Input (224×224×3)
  → Data Augmentation (RandomFlip horizontal, RandomRotation 0.1, RandomZoom 0.1, RandomContrast 0.1)
  → MobileNetV2 preprocess_input  (di dalam model)
  → MobileNetV2 (ImageNet, include_top=False)
  → GlobalAveragePooling2D
  → Dropout (0.2)
  → Dense (20, softmax)
```

Total parameter ±2,28 juta.

**Hyperparameter umum**

| Parameter | Nilai |
|-----------|-------|
| Ukuran gambar | 224 × 224 |
| Batch size | 32 |
| Seed | 123 |
| Loss | Categorical Crossentropy |
| Optimizer | Adam |

**Dua tahap pelatihan**

1. **Stage 1 – Feature extraction:** *base model* dibekukan, hanya *classification head* yang dilatih (learning rate `1e-3`, maks. 15 epoch, `EarlyStopping` patience 5).
2. **Stage 2 – Fine-tuning:** 20 layer terakhir MobileNetV2 dibuka (learning rate `1e-4`, maks. 15 epoch, `EarlyStopping` + `ReduceLROnPlateau` faktor 0,5).

Checkpoint terbaik disimpan berdasarkan `val_accuracy`.

---

## Hasil Evaluasi

Evaluasi dilakukan pada gabungan **validation + test (900 gambar)**.

| Metrik | Nilai |
|--------|------:|
| Accuracy | **95,67%** |
| Macro Precision | 95,82% |
| Macro Recall | 95,67% |
| Macro F1-score | 95,56% |
| Salah klasifikasi | 39 dari 900 |

**Ringkasan per kelas**

- **F1 = 1,00:** `batik_liong`, `batik_singa_barong`, `batik_tujuh_rupa`, `batik_tuntrum`
- **Paling sulit:**
  - `batik_sekarjagad` (recall 0,71, F1 0,83)
  - `batik_jlamprang` (F1 0,88)
  - `batik_tribusono` (F1 0,92)
  - `batik_sidoluhur` (F1 0,92)

<details>
<summary>Classification report lengkap</summary>

| Kelas | Precision | Recall | F1-score | Support |
|-------|----------:|-------:|---------:|--------:|
| batik_betawi | 0,9565 | 0,9778 | 0,9670 | 45 |
| batik_bokor_kencono | 0,9783 | 1,0000 | 0,9890 | 45 |
| batik_buketan | 0,9574 | 1,0000 | 0,9783 | 45 |
| batik_dayak | 0,9545 | 0,9333 | 0,9438 | 45 |
| batik_jlamprang | 0,8696 | 0,8889 | 0,8791 | 45 |
| batik_kawung | 0,9130 | 0,9333 | 0,9231 | 45 |
| batik_liong | 1,0000 | 1,0000 | 1,0000 | 45 |
| batik_mega_mendung | 0,9783 | 1,0000 | 0,9890 | 45 |
| batik_parang | 0,9362 | 0,9778 | 0,9565 | 45 |
| batik_sekarjagad | 1,0000 | 0,7111 | 0,8312 | 45 |
| batik_sidoluhur | 0,9750 | 0,8667 | 0,9176 | 45 |
| batik_sidomukti | 0,9362 | 0,9778 | 0,9565 | 45 |
| batik_sidomulyo | 0,8824 | 1,0000 | 0,9375 | 45 |
| batik_singa_barong | 1,0000 | 1,0000 | 1,0000 | 45 |
| batik_srikaton | 1,0000 | 0,9778 | 0,9888 | 45 |
| batik_tribusono | 0,9524 | 0,8889 | 0,9195 | 45 |
| batik_tujuh_rupa | 1,0000 | 1,0000 | 1,0000 | 45 |
| batik_tuntrum | 1,0000 | 1,0000 | 1,0000 | 45 |
| batik_wahyu_tumurun | 0,9375 | 1,0000 | 0,9677 | 45 |
| batik_wirasat | 0,9375 | 1,0000 | 0,9677 | 45 |
| **Accuracy** | | | **0,9567** | 900 |

</details>

> **Catatan:** data validation juga dipakai untuk memilih checkpoint dan *early stopping*, jadi angka di atas sedikit lebih optimistis dibanding evaluasi pada test set yang sepenuhnya terpisah.

---

## Cara Menggunakan Model

### Spesifikasi Input/Output TFLite

| | Shape | Tipe | Keterangan |
|---|---|---|---|
| **Input** | `(1, 224, 224, 3)` | `float32` | Piksel RGB **mentah (0–255)** |
| **Output** | `(1, 20)` | `float32` | Probabilitas tiap kelas |

> ⚠️ **Jangan menormalisasi gambar lagi.** `preprocess_input` MobileNetV2 (skala ke −1…1) sudah menjadi bagian dari model. Cukup *resize* ke 224×224 dan kirim nilai piksel 0–255 sebagai `float32`.

### Contoh Inferensi (Python)

```python
import numpy as np
import tensorflow as tf
from PIL import Image

# Muat label
with open("labels.txt") as f:
    labels = [line.strip() for line in f if line.strip()]

# Muat model TFLite
interpreter = tf.lite.Interpreter(model_path="batik_mobilenetv2.tflite")
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

# Siapkan gambar (tanpa normalisasi tambahan)
img = Image.open("contoh_batik.jpg").convert("RGB").resize((224, 224))
x = np.expand_dims(np.array(img, dtype=np.float32), axis=0)

# Prediksi
interpreter.set_tensor(inp["index"], x)
interpreter.invoke()
probs = interpreter.get_tensor(out["index"])[0]

# Top-3
for i in probs.argsort()[-3:][::-1]:
    print(f"{labels[i]}: {probs[i]:.2%}")
```

> `tf.lite.Interpreter` sudah *deprecated*. Untuk proyek baru, gunakan `ai_edge_litert`:
> `from ai_edge_litert.interpreter import Interpreter`.

### Integrasi Aplikasi Mobile

1. Salin `batik_mobilenetv2.tflite` dan `labels.txt` ke folder *assets* aplikasi.
2. Muat model dengan TensorFlow Lite / LiteRT (Android: `org.tensorflow:tensorflow-lite`).
3. *Resize* gambar kamera/galeri ke 224×224, kirim sebagai `float32` 0–255, lalu petakan indeks output ke baris pada `labels.txt`.

---

## Menjalankan Ulang Notebook

Notebook `model_deteksi_batik.ipynb` ditulis untuk **Google Colab**.

1. Unggah dataset ke Google Drive dengan struktur:
   ```
   dataset_batik_fix/processed_dataset/
   ├── train_original/<nama_kelas>/*.jpg
   ├── val/<nama_kelas>/*.jpg
   └── test/<nama_kelas>/*.jpg
   ```
2. Sesuaikan `DRIVE_DATASET_PATH` pada sel konfigurasi.
3. Jalankan seluruh sel secara berurutan. Di akhir, model dikonversi ke `batik_mobilenetv2.tflite` dan diunduh.

**Dependensi utama:** `tensorflow` (diuji pada 2.21), `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`.

---

## Keterbatasan

- Model hanya mengenali **20 motif** di atas; gambar di luar daftar tetap akan diprediksi ke salah satu kelas tersebut. Pertimbangkan ambang batas probabilitas (misalnya, tolak jika skor tertinggi < 0,5) di aplikasi.
- Motif dengan pola mirip (mis. `batik_sekarjagad`, `batik_jlamprang`) lebih sering tertukar.
- Performa bisa menurun pada foto dengan pencahayaan buruk, sudut ekstrem, atau kain berlipat yang berbeda dari data latih.