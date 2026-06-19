from flask import Flask, render_template, request, send_file
import os
from werkzeug.utils import secure_filename
import shutil
import glob
import subprocess
import zipfile
import random # Import modul random

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Folder output YOLO
predict_folder = os.path.join('runs', 'detect', 'predict')
static_result_path = os.path.join('static', 'runs', 'detect', 'predict')
os.makedirs(static_result_path, exist_ok=True)

# Folder ground truth (Roboflow test labels)
ground_truth_label_dir = 'dataset/test/labels'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        files = request.files.getlist('image')
        uploaded_filepaths = []

        for file in files:
            if file and file.filename != '':
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                uploaded_filepaths.append(filepath)

        if not uploaded_filepaths:
            return "Tidak ada file yang dipilih.", 400

        # Bersihkan hasil lama
        if os.path.exists(predict_folder):
            shutil.rmtree(predict_folder)

        # Jalankan YOLO untuk setiap gambar
        for filepath in uploaded_filepaths:
            result = subprocess.run([
                "yolo", "predict",
                "model=best.pt",
                f"source={filepath}",
                "save_txt=True",
                "save_conf=True",          # Penting: untuk mendapatkan confidence di file .txt
                "conf=0.25",
                "project=runs/detect",
                "name=predict",
                "exist_ok=True"
            ], capture_output=True, text=True)

            if result.returncode != 0:
                return f"YOLO Error: {result.stderr}", 500

        # Ambil hasil deteksi (gambar)
        result_files = glob.glob(os.path.join(predict_folder, '*.jpg'))
        if not result_files:
            return "Gagal menemukan hasil deteksi.", 500

        copied_filenames = []
        for result_file in result_files:
            fname = os.path.basename(result_file)
            shutil.copy(result_file, os.path.join(static_result_path, fname))
            copied_filenames.append(fname)
        
        # --- MULAI MODIFIKASI ---
        # Acak daftar dan pilih subset (misalnya, 5 gambar)
        random.shuffle(copied_filenames)
        display_filenames = copied_filenames[:5] # Ubah 5 menjadi 3 jika Anda hanya ingin 3 gambar
        # --- AKHIR MODIFIKASI ---

        # Evaluasi confidence dan akurasi
        confidence_dict = {}
        correct = 0
        total = 0

        for result_filename in copied_filenames: # Loop ini tetap mengiterasi semua copied_filenames untuk evaluasi
            base_name = os.path.splitext(result_filename)[0]
            predict_txt = os.path.join(predict_folder, base_name + '.txt')
            label_txt = os.path.join(ground_truth_label_dir, base_name + '.txt')

            max_conf = 0.0
            pred_class = None

            # Ambil prediksi confidence tertinggi
            if os.path.exists(predict_txt):
                with open(predict_txt, 'r') as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 6:
                            class_id = parts[0]
                            conf = float(parts[5])
                            if conf > max_conf:
                                max_conf = conf
                                pred_class = class_id
                confidence_dict[result_filename] = max_conf
            else:
                confidence_dict[result_filename] = 0.0

            # Bandingkan pred_class dengan ground truth classes
            if os.path.exists(label_txt) and pred_class is not None:
                with open(label_txt, 'r') as f:
                    gt_classes = [line.strip().split()[0] for line in f if line.strip()]
                    if pred_class in gt_classes:
                        correct += 1
                    total += 1

        accuracy = (correct / total) if total > 0 else 0.0

        # Buat ZIP hasil deteksi (semua hasil)
        zip_output_path = os.path.join('static', 'runs', 'detect', 'predict.zip')
        with zipfile.ZipFile(zip_output_path, 'w') as zipf:
            for root, dirs, files in os.walk(predict_folder):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, os.path.join(predict_folder, '..'))
                    zipf.write(full_path, arcname)

        return render_template('hasil.html',
                               filenames=display_filenames, # Kirim daftar yang sudah diiris ke sini
                               confidence=confidence_dict,
                               accuracy={'Test Accuracy': accuracy})

    return render_template('deteksi.html')

@app.route('/download-predict-zip')
def download_predict_zip():
    zip_path = os.path.join('static', 'runs', 'detect', 'predict.zip')
    if os.path.exists(zip_path):
        return send_file(zip_path, as_attachment=True)
    else:
        return "File zip tidak ditemukan.", 404

if __name__ == '__main__':
    app.run(debug=True)