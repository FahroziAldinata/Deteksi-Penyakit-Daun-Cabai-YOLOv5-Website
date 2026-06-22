# 🌶️ Deteksi Penyakit Daun Cabai

Aplikasi web untuk mendeteksi penyakit pada daun cabai secara otomatis menggunakan model **YOLOv8 (Ultralytics)**, dibangun dengan **Flask** untuk backend dan **Bootstrap 5** untuk tampilan antarmuka.

Pengguna dapat mengunggah satu gambar atau satu folder berisi banyak gambar daun cabai, lalu sistem akan memproses dan menampilkan hasil deteksi (bounding box, label, dan confidence score) secara langsung di halaman web, lengkap dengan opsi unduh hasil dalam format ZIP.

---

## ✨ Fitur Utama

- **Unggah Gambar Tunggal** maupun **Unggah Folder** (banyak gambar sekaligus).
- **Deteksi otomatis** penyakit daun cabai menggunakan model YOLOv8 yang telah dilatih (`best.pt`).
- **Isolasi per-permintaan (per-request)** — setiap proses deteksi mendapatkan folder unik (`request_id`), sehingga hasil antar pengguna tidak akan tertimpa atau tercampur.
- **Halaman hasil** menampilkan gambar yang telah diberi bounding box hasil deteksi.
- **Evaluasi akurasi** otomatis jika gambar yang diunggah memiliki ground truth pada dataset uji internal.
- **Unduh hasil deteksi** dalam bentuk file ZIP.
- **Validasi keamanan dasar**: hanya menerima ekstensi gambar (`png`, `jpg`, `jpeg`, `webp`), validasi `request_id` untuk mencegah path traversal, dan pembatasan ukuran unggahan maksimum 20 MB.

---

## 🛠️ Teknologi yang Digunakan

| Komponen        | Teknologi                              |
|------------------|------------------------------------------|
| Backend          | Flask (Python)                          |
| Model AI         | YOLOv5 — [Ultralytics](https://github.com/ultralytics/ultralytics) |
| Frontend         | HTML, CSS, Bootstrap 5, Bootstrap Icons |
| Template Desain  | TemplateMo 583 Festava Live             |
| JavaScript       | jQuery, Bootstrap Bundle, Sticky JS     |

---

## 📁 Struktur Proyek

```
├── app.py                          # Entry point aplikasi Flask
├── best.pt                         # Model YOLOv5 hasil training
├── requirements.txt                 # Daftar dependensi Python
├── dataset/
│   └── test/labels/                 # Label ground truth untuk evaluasi akurasi
├── templates/
│   ├── index.html                   # Halaman utama / landing page
│   ├── Deteksi.html                 # Halaman form unggah gambar
│   └── Hasil.html                   # Halaman hasil deteksi
└── static/
    ├── css/                          # Bootstrap, Bootstrap Icons, custom CSS
    ├── js/                           # jQuery, Bootstrap JS, custom scripts
    ├── images/                       # Aset gambar template
    ├── uploads/                      # Folder sementara hasil unggahan (per request)
    └── runs/detect/                  # Folder output hasil deteksi YOLO (per request)
```

---

## ⚙️ Cara Instalasi & Menjalankan

### 1. Clone repository
```bash
git clone <url-repository-anda>
cd <nama-folder-proyek>
```

### 2. Buat virtual environment (opsional, disarankan)
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
```

### 3. Install dependensi
```bash
pip install -r requirements.txt
```

### 4. Pastikan file model tersedia
Letakkan file model hasil training (`best.pt`) pada direktori utama proyek (sejajar dengan `app.py`).

### 5. Jalankan aplikasi
```bash
python app.py
```

Secara default, mode debug bersifat **non-aktif**. Untuk mode pengembangan (debug), set environment variable berikut sebelum menjalankan:

```bash
# Windows
set FLASK_DEBUG=true

# Linux/Mac
export FLASK_DEBUG=true
```

### 6. Akses aplikasi
Buka browser dan kunjungi:
```
http://127.0.0.1:5000
```

---

## 🚀 Cara Penggunaan

1. Buka halaman utama, klik tombol **"Mulai Deteksi"**.
2. Pada halaman input, unggah **gambar tunggal** atau **folder berisi banyak gambar** daun cabai (format yang didukung: `.png`, `.jpg`, `.jpeg`, `.webp`).
3. Klik tombol **"Deteksi"**.
4. Tunggu proses deteksi selesai (akan muncul indikator loading).
5. Hasil deteksi akan ditampilkan di halaman berikutnya beserta confidence score dan (jika tersedia) evaluasi akurasi model.
6. Hasil deteksi dapat diunduh secara lengkap dalam format **ZIP**.

---

## 📌 Catatan Teknis

- Maksimum ukuran total unggahan per permintaan adalah **20 MB**.
- Hasil deteksi dan unggahan disimpan dalam folder unik berbasis UUID untuk setiap permintaan, sehingga aman digunakan oleh banyak pengguna secara bersamaan.
- Evaluasi akurasi hanya akan ditampilkan apabila gambar yang diunggah memiliki data label (ground truth) yang cocok pada folder `dataset/test/labels`.

---

## 👨‍💻 Tim Pengembang

1. **Fahrozi Aldinata**
2. **Cindy Aisyah Putri**
3. **Trisma Muliani**
4. **Akmal Ardiansyah**

*Kelompok 7 — Jaringan Saraf Tiruan (JST)*

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan tugas akademik. Template antarmuka berbasis **TemplateMo 583 Festava Live** (https://templatemo.com/tm-583-festava-live).

© 2025 Deteksi Penyakit Daun Cabai
