"""
Flask API server for the ANPR frontend.
Run:  python app.py   ->  http://localhost:5000
"""

import os, base64, traceback
import cv2, numpy as np
from flask import Flask, request, jsonify, send_from_directory

import sys
sys.path.insert(0, os.path.dirname(__file__))

from main import (
    prepare, detect_plate, ocr_plate, build_templates, _precompute_hu,
    fix_indian, USE_OCR, INDIAN_FORMAT
)

app = Flask(__name__, static_folder="frontend", static_url_path="")

_templates     = None
_tmpl_hu_cache = None
_ocr_active    = False

if USE_OCR:
    try:
        _templates     = build_templates()
        _tmpl_hu_cache = _precompute_hu(_templates)
        _ocr_active    = True
        print("[ANPR] Template matching OCR ready.")
    except Exception as _e:
        print(f"[ANPR] OCR disabled: {_e}")


def _enc(bgr):
    ok, buf = cv2.imencode(".png", bgr)
    return ("data:image/png;base64," + base64.b64encode(buf.tobytes()).decode()) if ok else None

def _enc_gray(g):
    return _enc(cv2.cvtColor(g, cv2.COLOR_GRAY2BGR))

def _draw_boxes(img, res):
    out = img.copy()
    if res.get("box_a") is not None:
        cv2.drawContours(out, [res["box_a"]], -1, (0, 0, 255), 2)
    if res.get("box") is not None:
        cv2.drawContours(out, [res["box"]], -1, (0, 255, 0), 3)
    return out


@app.route("/")
def index():
    return send_from_directory("frontend", "index.html")


@app.route("/process", methods=["POST"])
def process():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    data = request.files["image"].read()
    arr  = np.frombuffer(data, np.uint8)
    img  = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Could not decode image"}), 400

    try:
        img = prepare(img)
        res = detect_plate(img)

        stages = [
            {"label": "Original",                    "img": _enc(img)},
            {"label": "Grayscale",                   "img": _enc_gray(res["gray"])},
            {"label": "Gamma Corrected",             "img": _enc_gray(res["gamma"])},
            {"label": "CLAHE",                       "img": _enc_gray(res["clahe"])},
            {"label": "Denoised (Bilateral Filter)", "img": _enc_gray(res["blur"])},
            {"label": "Edges (Canny)",               "img": _enc_gray(res["edges"])},
            {"label": "Threshold (Otsu)",            "img": _enc_gray(res["thresh"])},
        ]
        if res.get("closed") is not None:
            stages.append({"label": "Morphology (Closed)",
                           "img": _enc_gray(res["closed"])})
        stages.append({"label": "Detection (Red=A, Green=B)",
                       "img": _enc(_draw_boxes(img, res))})

        text      = ""
        plate_uri = None
        detected  = res.get("plate") is not None

        if detected:
            plate_uri = _enc(res["plate"])
            if _ocr_active and _templates:
                try:
                    text, bw_dbg = ocr_plate(res["plate"], _templates, _tmpl_hu_cache)
                    if (not text or len(text.replace("?", "")) < 2) \
                            and res.get("tight") is not None:
                        t2, bw2 = ocr_plate(res["tight"], _templates, _tmpl_hu_cache)
                        if len(t2.replace("?", "")) > len(text.replace("?", "")):
                            text, bw_dbg = t2, bw2
                    if bw_dbg is not None:
                        stages.append({"label": "OCR Binary (normalised plate)",
                                       "img": _enc_gray(bw_dbg)})
                except Exception as e:
                    print("[ANPR] OCR error:", e)

            if INDIAN_FORMAT and text:
                text = fix_indian(text)

            stages.append({"label": f'Plate Crop - "{text}"',
                           "img": plate_uri})

        return jsonify({
            "detected":    detected,
            "plate_text":  text,
            "ocr_engine":  "template" if _ocr_active else "none",
            "plate_image": plate_uri,
            "stages":      stages,
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("=" * 55)
    print(f"  ANPR  ->  http://localhost:{port}")
    print("=" * 55)
    app.run(debug=False, host="0.0.0.0", port=port)
