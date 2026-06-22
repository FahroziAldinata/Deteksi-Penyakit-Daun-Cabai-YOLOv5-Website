import os
import uuid
import glob
import zipfile
import random

from flask import Flask, render_template, request, send_file
from werkzeug.utils import secure_filename
from ultralytics import YOLO

# ---------------------------------------------------------
# PATH SETUP (portable, relatif terhadap lokasi app.py)
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__)

# ---------------------------------------------------------
# [FASE 1] KONFIGURASI PRODUKSI
# ---------------------------------------------------------
# Debug mode dikontrol lewat environment variable, BUKAN hardcode True.
# Default-nya False (aman). Untuk development lokal, jalankan dengan:
#   set FLASK_DEBUG=true   (Windows)  /  export FLASK_DEBUG=true (Linux/Mac)
# sebelum "python app.py". Untuk produksi, JANGAN di-set sama sekali.
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'

# Batas ukuran total request (semua file yang diupload digabung).
# Mencegah orang upload file raksasa untuk menguras disk/CPU server.
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20 MB

UPLOAD_FOLDER = os.path.join(BASE_DIR, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

STATIC_RESULTS_ROOT = os.path.join(BASE_DIR, 'static', 'runs', 'detect')
os.makedirs(STATIC_RESULTS_ROOT, exist_ok=True)

ground_truth_label_dir = os.path.join(BASE_DIR, 'dataset', 'test', 'labels')

MODEL_PATH = os.path.join(BASE_DIR, 'best.pt')

# Hanya tipe file gambar yang diizinkan diupload.
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}


def allowed_file(filename):
    return (
        '.' in filename
        and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def is_valid_request_id(value):
    # request_id selalu hex uuid4 (32 karakter 0-9a-f).
    # Validasi ini mencegah path traversal lewat URL (mis. "../../../etc").
    return len(value) == 32 and all(c in '0123456789abcdef' for c in value)


# Load model sekali saat app start, bukan setiap request.
model = YOLO(MODEL_PATH)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/deteksi', methods=['GET', 'POST'])
def deteksi():
    if request.method == 'POST':
        files = request.files.getlist('image')

        # ---------------------------------------------------------
        # [FASE 1] ISOLASI PER-REQUEST
        # Setiap request dapat folder unik (request_id) untuk upload
        # maupun hasil deteksi. Ini menggantikan pola lama yang pakai
        # SATU folder global ("runs/detect/predict") untuk semua user
        # -- pola lama itu rawan: kalau dua orang upload bersamaan,
        # request kedua bisa menghapus/menimpa hasil milik request
        # pertama (race condition), bahkan bisa bikin user A melihat
        # hasil deteksi gambar milik user B.
        # ---------------------------------------------------------
        request_id = uuid.uuid4().hex
        upload_dir = os.path.join(app.config['UPLOAD_FOLDER'], request_id)
        os.makedirs(upload_dir, exist_ok=True)

        uploaded_filepaths = []
        for file in files:
            if not file or file.filename == '':
                continue
            if not allowed_file(file.filename):
                # Lewati file yang bukan gambar (mis. .exe, .php, .html)
                continue
            filename = secure_filename(file.filename)
            filepath = os.path.join(upload_dir, filename)
            file.save(filepath)
            uploaded_filepaths.append(filepath)

        if not uploaded_filepaths:
            return "Tidak ada file gambar yang valid untuk diproses (format diterima: png, jpg, jpeg, webp).", 400

        # Output YOLO langsung disimpan di dalam static/, di folder
        # khusus request_id ini -- tidak perlu lagi proses copy manual
        # dari runs/detect/predict ke static/runs/detect/predict seperti
        # versi sebelumnya.
        result_project_dir = os.path.join(STATIC_RESULTS_ROOT, request_id)

        try:
            results = model.predict(
                source=uploaded_filepaths,
                conf=0.25,
                save=True,
                save_txt=True,
                save_conf=True,
                project=result_project_dir,
                name='predict',
                exist_ok=True
            )
        except Exception as e:
            return f"YOLO Error: {e}", 500

        result_dir = os.path.join(result_project_dir, 'predict')
        result_files = glob.glob(os.path.join(result_dir, '*.jpg'))
        if not result_files:
            return "Gagal menemukan hasil deteksi.", 500

        all_filenames = [os.path.basename(f) for f in result_files]

        # Acak & pilih subset untuk ditampilkan (misal 5 gambar)
        random.shuffle(all_filenames)
        display_filenames = all_filenames[:5]

        # ---------------------------------------------------------
        # EVALUASI CONFIDENCE & AKURASI (dari objek `results` di memori,
        # bukan baca ulang file .txt dari disk)
        # ---------------------------------------------------------
        confidence_dict = {}
        correct = 0
        total = 0

        for result in results:
            result_filename = os.path.basename(result.path)
            base_name = os.path.splitext(result_filename)[0]

            max_conf = 0.0
            pred_class = None

            if result.boxes is not None and len(result.boxes) > 0:
                confs = result.boxes.conf.tolist()
                classes = result.boxes.cls.tolist()
                best_idx = max(range(len(confs)), key=lambda i: confs[i])
                max_conf = confs[best_idx]
                pred_class = str(int(classes[best_idx]))

            confidence_dict[result_filename] = max_conf

            label_txt = os.path.join(ground_truth_label_dir, base_name + '.txt')
            if os.path.exists(label_txt) and pred_class is not None:
                with open(label_txt, 'r') as f:
                    gt_classes = [line.strip().split()[0] for line in f if line.strip()]
                    if pred_class in gt_classes:
                        correct += 1
                    total += 1

        # Akurasi hanya berarti kalau ada ground truth yang cocok
        # (mis. gambar dari dataset uji internal kamu). Untuk gambar
        # sembarang dari user publik biasanya tidak ada ground truth,
        # jadi total == 0 dan accuracy jadi None (ditandai, bukan 0%
        # yang menyesatkan).
        accuracy = (correct / total) if total > 0 else None

        return render_template(
            'hasil.html',
            request_id=request_id,
            filenames=display_filenames,
            confidence=confidence_dict,
            accuracy={'Test Accuracy': accuracy} if accuracy is not None else None
        )

    return render_template('deteksi.html')


@app.route('/download-predict-zip/<request_id>')
def download_predict_zip(request_id):
    if not is_valid_request_id(request_id):
        return "ID tidak valid.", 400

    result_dir = os.path.join(STATIC_RESULTS_ROOT, request_id, 'predict')
    if not os.path.isdir(result_dir):
        return "Hasil deteksi tidak ditemukan atau sudah kedaluwarsa.", 404

    zip_output_path = os.path.join(STATIC_RESULTS_ROOT, request_id, 'predict.zip')
    if not os.path.exists(zip_output_path):
        with zipfile.ZipFile(zip_output_path, 'w') as zipf:
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    arcname = os.path.relpath(full_path, result_dir)
                    zipf.write(full_path, arcname)

    return send_file(zip_output_path, as_attachment=True, download_name='hasil_deteksi.zip')


@app.errorhandler(413)
def file_too_large(e):
    return "Ukuran file terlalu besar. Maksimal 20MB per request.", 413


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'])