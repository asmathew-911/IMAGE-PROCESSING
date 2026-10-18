"""
Flask API server for the Vehicle Number Plate Recognition frontend.
Run with:  python app.py
Then open http://localhost:5000 in your browser.
"""
import os
import io
import base64
import tempfile
import traceback
import uuid
import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory

# ── import the core pipeline from main.py ──────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from main import (
    prepare, detect_plate,
    ocr_easyocr, ocr_tesseract, segment_chars, recognize, build_templates,
    fix_indian,
    OCR_MODE, USE_OCR, INDIAN_FORMAT, TESSERACT_PATH
)

app = Flask(__name__, static_folder="frontend", static_url_path="")

# ── initialise OCR engine once at startup ──────────────────────────────────
_reader = None
_templates = None
_ocr_mode = None


def _init_ocr():
    global _reader, _templates, _ocr_mode
    if _ocr_mode is not None:
        return

    if not USE_OCR:
        _ocr_mode = None
        return

    mode = OCR_MODE
    if mode == "easyocr":
        try:
            import easyocr
            print("[ANPR] Loading EasyOCR ...")
            _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
            _ocr_mode = "easyocr"
            print("[ANPR] EasyOCR ready.")
            return
        except Exception as e:
            print(f"[ANPR] EasyOCR unavailable ({e}), trying Tesseract ...")
            mode = "tesseract"

    if mode == "tesseract":
        try:
            import pytesseract
            if os.path.exists(TESSERACT_PATH):
                pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH
            pytesseract.get_tesseract_version()
            _ocr_mode = "tesseract"
            print("[ANPR] Tesseract ready.")
            return
        except Exception as e:
            print(f"[ANPR] Tesseract unavailable ({e}), using template matching ...")
            mode = "template"

    if mode == "template":
        try:
            _templates = build_templates()
            _ocr_mode = "template"
            print("[ANPR] Template matching ready.")
        except Exception as e:
            print(f"[ANPR] OCR completely disabled: {e}")
            _ocr_mode = None


_init_ocr()


# ── helpers ────────────────────────────────────────────────────────────────
def _encode_img(bgr_img):
    """Encode a BGR OpenCV image to a base64 PNG data-URI."""
    ok, buf = cv2.imencode(".png", bgr_img)
    if not ok:
        return None
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()


def _encode_gray(gray_img):
    return _encode_img(cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR))


def _draw_boxes(img, res):
    """Return a copy of img with detection boxes drawn on it."""
    out = img.copy()
    if res.get("box_a") is not None:
        cv2.drawContours(out, [res["box_a"]], -1, (0, 0, 255), 2)   # Method A - red
    if res.get("box") is not None:
        cv2.drawContours(out, [res["box"]], -1, (0, 255, 0), 3)      # Method B - green
    return out


# ── routes ─────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    data = file.read()
    nparr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        img = prepare(img)
        res = detect_plate(img)

        # ── collect processing stages ──────────────────────────────────────
        stages = [
            {"label": "Original",                    "img": _encode_img(img)},
            {"label": "Grayscale",                   "img": _encode_gray(res["gray"])},
            {"label": "CLAHE",                       "img": _encode_gray(res["clahe"])},
            {"label": "Denoised (Bilateral Filter)", "img": _encode_gray(res["blur"])},
            {"label": "Edges (Canny)",               "img": _encode_gray(res["edges"])},
            {"label": "Threshold (Otsu)",            "img": _encode_gray(res["thresh"])},
        ]
        if res.get("closed") is not None:
            stages.append({"label": "Morphology (Closed)", "img": _encode_gray(res["closed"])})

        annotated = _draw_boxes(img, res)
        stages.append({"label": "Detection Result  (Red=A, Green=B)", "img": _encode_img(annotated)})

        # ── OCR ────────────────────────────────────────────────────────────
        text = ""
        plate_uri = None
        detected = res.get("plate") is not None

        if detected:
            plate_uri = _encode_img(res["plate"])
            try:
                if _ocr_mode == "easyocr":
                    text = ocr_easyocr(res["plate"], _reader, res.get("tight"))
                elif _ocr_mode == "tesseract":
                    text = ocr_tesseract(res["plate"])
                elif _ocr_mode == "template" and _templates:
                    text = recognize(segment_chars(res["plate"]), _templates)
            except Exception as e:
                text = ""
                print("[ANPR] OCR error:", e)

            if INDIAN_FORMAT and text:
                text = fix_indian(text)

            stages.append({"label": f'Plate Crop  -  "{text}"', "img": plate_uri})

        return jsonify({
            "detected": detected,
            "plate_text": text,
            "ocr_engine": _ocr_mode or "none",
            "plate_image": plate_uri,
            "stages": stages,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("=" * 55)
    print("  ANPR Web Server  ->  http://localhost:5000")
    print("=" * 55)
    app.run(debug=True, host="0.0.0.0", port=5000)
