from pathlib import Path
import os

# WAJIB ditaruh sebelum import cv2 / tensorflow / torch
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import base64
import time
import traceback
import threading

import cv2
import numpy as np
from flask import Flask, redirect, render_template, render_template_string, request, url_for

try:
    cv2.setNumThreads(1)
except Exception:
    pass


BASE_DIR = Path(__file__).resolve().parent

SERVING_ROOT = BASE_DIR / "static" / "serving"
SERVING_ROOT.mkdir(parents=True, exist_ok=True)

ORIGINAL_IMAGE_FILENAME = "original_image.jpg"
DETECTED_IMAGE_FILENAME = "detected_image.jpg"

ORIGINAL_IMAGE_ROOT = SERVING_ROOT / ORIGINAL_IMAGE_FILENAME
DETECTED_IMAGE_ROOT = SERVING_ROOT / DETECTED_IMAGE_FILENAME

app = Flask(__name__)

# Batasi ukuran upload agar Render tidak kehabisan memori
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024

_braille_classifier = None
_classifier_lock = threading.Lock()


def log_time(label, start_time):
    duration = time.time() - start_time
    print(f"[TIME] {label}: {duration:.2f} detik", flush=True)


def resize_image_if_needed(image, max_side=1200):
    """
    Mengecilkan gambar agar proses YOLO/CNN tidak terlalu berat.
    """
    if image is None or image.size == 0:
        return image

    height, width = image.shape[:2]
    current_max_side = max(height, width)

    if current_max_side <= max_side:
        return image

    scale = max_side / current_max_side
    new_width = int(width * scale)
    new_height = int(height * scale)

    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def get_braille_classifier():
    """
    Load model satu kali saja.
    Jangan load model di dalam route secara berulang.
    """
    global _braille_classifier

    if _braille_classifier is None:
        with _classifier_lock:
            if _braille_classifier is None:
                start = time.time()
                print("[INFO] Mulai load BrailleClassifier...", flush=True)

                try:
                    import tensorflow as tf
                    tf.config.threading.set_intra_op_parallelism_threads(1)
                    tf.config.threading.set_inter_op_parallelism_threads(1)
                except Exception as exc:
                    print(f"[WARNING] TensorFlow thread config gagal: {exc}", flush=True)

                try:
                    import torch
                    torch.set_num_threads(1)
                except Exception as exc:
                    print(f"[WARNING] Torch thread config gagal: {exc}", flush=True)

                from control.classify import BrailleClassifier

                _braille_classifier = BrailleClassifier(
                    model_path=str(BASE_DIR / "weights" / "cnn_v1.hdf5"),
                    json_path=str(BASE_DIR / "utils" / "class_labels.json"),
                    symbols_path=str(BASE_DIR / "utils" / "braille_symbols.json"),
                    numbers_path=str(BASE_DIR / "utils" / "braille_numbers.json"),
                    yolo_weight=str(BASE_DIR / "weights" / "yolov8_braille.pt"),
                )

                log_time("Load BrailleClassifier selesai", start)

    return _braille_classifier


def order_document_points(points):
    """
    Mengurutkan 4 titik menjadi kiri-atas, kanan-atas, kanan-bawah, kiri-bawah.
    """
    rect = np.zeros((4, 2), dtype="float32")
    points = points.astype("float32")

    point_sum = points.sum(axis=1)
    rect[0] = points[np.argmin(point_sum)]
    rect[2] = points[np.argmax(point_sum)]

    point_diff = np.diff(points, axis=1)
    rect[1] = points[np.argmin(point_diff)]
    rect[3] = points[np.argmax(point_diff)]

    return rect


def four_point_transform(image, points):
    """
    Meluruskan area dokumen berdasarkan 4 titik sudut.
    """
    rect = order_document_points(points)
    top_left, top_right, bottom_right, bottom_left = rect

    width_a = np.linalg.norm(bottom_right - bottom_left)
    width_b = np.linalg.norm(top_right - top_left)
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(top_right - bottom_right)
    height_b = np.linalg.norm(top_left - bottom_left)
    max_height = int(max(height_a, height_b))

    if max_width < 80 or max_height < 80:
        return image

    destination = np.array(
        [
            [0, 0],
            [max_width - 1, 0],
            [max_width - 1, max_height - 1],
            [0, max_height - 1],
        ],
        dtype="float32",
    )

    matrix = cv2.getPerspectiveTransform(rect, destination)
    corrected = cv2.warpPerspective(image, matrix, (max_width, max_height))

    return resize_image_if_needed(corrected, max_side=1200)


def auto_straighten_document(image):
    """
    Mencari empat sudut kertas lalu melakukan koreksi perspektif.
    Jika sudut kertas tidak ditemukan, gambar dikembalikan apa adanya.
    """
    if image is None or image.size == 0:
        return image

    original = image.copy()
    height, width = image.shape[:2]
    max_side = max(height, width)

    # Dibuat lebih ringan dari 900 agar Render tidak terlalu berat
    scale = 700.0 / max_side if max_side > 700 else 1.0

    if scale != 1.0:
        resized = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA
        )
    else:
        resized = image.copy()

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    edges = cv2.dilate(edges, kernel, iterations=1)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=1)

    contours, _ = cv2.findContours(
        edges,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return resize_image_if_needed(original, max_side=1200)

    image_area = resized.shape[0] * resized.shape[1]
    best_quad = None
    best_area = 0

    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:8]:
        area = cv2.contourArea(contour)

        if area < image_area * 0.08:
            continue

        perimeter = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.025 * perimeter, True)

        if len(approx) == 4:
            quad = approx.reshape(4, 2).astype("float32")
        else:
            rect = cv2.minAreaRect(contour)
            quad = cv2.boxPoints(rect).astype("float32")

        x, y, w, h = cv2.boundingRect(quad.astype("int32"))

        if w < resized.shape[1] * 0.25 or h < resized.shape[0] * 0.25:
            continue

        aspect_ratio = max(w / max(h, 1), h / max(w, 1))

        if aspect_ratio > 5.0:
            continue

        if area > best_area:
            best_area = area
            best_quad = quad

    if best_quad is None:
        return resize_image_if_needed(original, max_side=1200)

    if scale != 1.0:
        best_quad = best_quad / scale

    corrected = four_point_transform(original, best_quad)

    corrected_area = corrected.shape[0] * corrected.shape[1]
    original_area = original.shape[0] * original.shape[1]

    if corrected_area < original_area * 0.12:
        return resize_image_if_needed(original, max_side=1200)

    return resize_image_if_needed(corrected, max_side=1200)


def save_image_with_optional_straightening(image_bytes, output_path, enable_straightening=True):
    """
    Menyimpan gambar hasil upload/kamera, lalu meluruskannya bila memungkinkan.
    """
    start = time.time()

    np_buffer = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(np_buffer, cv2.IMREAD_COLOR)

    if image is None:
        output_path.write_bytes(image_bytes)
        return False

    if enable_straightening:
        image = auto_straighten_document(image)
    else:
        image = resize_image_if_needed(image, max_side=1200)

    cv2.imwrite(
        str(output_path),
        image,
        [int(cv2.IMWRITE_JPEG_QUALITY), 85]
    )

    log_time("Simpan gambar selesai", start)
    return True


@app.route("/")
def index():
    return redirect(url_for("predict"))


@app.route("/healthz")
def healthz():
    return "OK", 200


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "GET":
        return render_template("predict.html")

    start = time.time()

    uploaded_file = request.files.get("file")
    captured_image = (request.form.get("captured_image") or "").strip()
    enable_straightening = request.form.get("enable_straightening", "1") == "1"

    if ORIGINAL_IMAGE_ROOT.exists():
        ORIGINAL_IMAGE_ROOT.unlink()

    if DETECTED_IMAGE_ROOT.exists():
        DETECTED_IMAGE_ROOT.unlink()

    if captured_image:
        try:
            if "," in captured_image:
                captured_image = captured_image.split(",", 1)[1]

            image_bytes = base64.b64decode(captured_image)
            save_image_with_optional_straightening(
                image_bytes,
                ORIGINAL_IMAGE_ROOT,
                enable_straightening
            )

            log_time("POST kamera /predict selesai", start)
            return redirect(url_for("result"))

        except Exception:
            print(traceback.format_exc(), flush=True)
            return redirect(url_for("predict"))

    if uploaded_file is None or uploaded_file.filename == "":
        return redirect(url_for("predict"))

    image_bytes = uploaded_file.read()

    save_image_with_optional_straightening(
        image_bytes,
        ORIGINAL_IMAGE_ROOT,
        enable_straightening
    )

    log_time("POST upload /predict selesai", start)
    return redirect(url_for("result"))


@app.route("/result")
def result():
    if not ORIGINAL_IMAGE_ROOT.exists():
        return redirect(url_for("predict"))

    start = time.time()

    try:
        classifier = get_braille_classifier()

        recognition_start = time.time()
        recognition_result = classifier.recognize_braille(str(ORIGINAL_IMAGE_ROOT))
        log_time("Recognize Braille selesai", recognition_start)

        if len(recognition_result) == 6:
            (
                predicted_image,
                character_result,
                syllable_result,
                speech_text,
                character_cells,
                syllable_cells,
            ) = recognition_result
        else:
            (
                predicted_image,
                syllable_result,
                speech_text,
                syllable_cells,
            ) = recognition_result

            character_result = syllable_result
            character_cells = syllable_cells

        if predicted_image is None:
            raise RuntimeError(
                "Gambar tidak dapat diproses. Coba gunakan foto Braille yang lebih jelas."
            )

        predicted_image = resize_image_if_needed(predicted_image, max_side=1200)

        cv2.imwrite(
            str(DETECTED_IMAGE_ROOT),
            predicted_image,
            [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        )

        log_time("GET /result selesai total", start)

        return render_template(
            "result.html",
            cache_buster=int(time.time()),
            predicted_result=syllable_result,
            character_result=character_result,
            syllable_result=syllable_result,
            speech_text=speech_text,
            detected_cells=syllable_cells,
            character_cells=character_cells,
            syllable_cells=syllable_cells,
        )

    except Exception as exc:
        print(traceback.format_exc(), flush=True)

        return render_template_string(
            """
            {% extends "base.html" %}
            {% block content %}
            <div style="max-width:900px;margin:40px auto;padding:24px;background:#fff;border-radius:12px;">
                <h2>Aplikasi berhasil jalan, tetapi proses pengenalan Braille gagal.</h2>
                <p><b>Error:</b> {{ error }}</p>
                <p>
                    Biasanya ini terjadi karena model terlalu berat, file model tidak ditemukan,
                    dependency machine learning belum cocok, atau gambar tidak terbaca.
                </p>
                <pre style="white-space:pre-wrap;background:#f4f4f4;padding:16px;border-radius:8px;">{{ details }}</pre>
                <a href="{{ url_for('predict') }}">Kembali upload gambar</a>
            </div>
            {% endblock %}
            """,
            error=str(exc),
            details=traceback.format_exc(),
        ), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 3001))
    print(f"Aplikasi berjalan di port: {port}", flush=True)
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
