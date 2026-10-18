"""
Flask API server for the Vehicle Number Plate Recognition frontend.
Run with:  python app.py
Then open http://localhost:5000 in your browser.
"""
import os
import base64
import traceback
import cv2
import numpy as np

from flask import Flask, request, jsonify, send_from_directory

# ── import the core pipeline from main.py ──────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))

from main import (
    prepare, detect_plate,
    segment_chars, recognize, build_templates,
    fix_indian,
    USE_OCR, INDIAN_FORMAT
)

app = Flask(__name__, static_folder="frontend", static_url_path="")

# ── initialise classical template OCR once at startup ─────────────────────
_templates = None
_ocr_active = False

if USE_OCR:
    try:
        _templates = build_templates()
        _ocr_active = True
        print("[ANPR] Template matching OCR ready.")
    except Exception as _e:
        print(f"[ANPR] OCR disabled (font problem): {_e}")


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
                if _ocr_active and _templates:
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
            "ocr_engine": "template" if _ocr_active else "none",
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
