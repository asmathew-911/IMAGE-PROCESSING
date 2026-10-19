"""
Vehicle Number Plate Detection & OCR
Fully classical image-processing pipeline (no AI / ML).

Detection
  Method A - morphological edges + contour filtering
  Method B - adaptive-threshold char-blob grouping
  Both results are fused; Method B wins when available.

OCR (fully classical)
  1. Normalise the plate crop  (deskew, rescale, contrast-stretch)
  2. Multi-strategy binarisation vote (Otsu at 5 blur scales + adaptive)
  3. Row-profile crop to isolate character band
  4. Vertical projection profile -> character column boundaries
  5. cv2.matchTemplate + Hu-moment shape descriptor hybrid matcher
  6. Confidence-gated output; low-confidence chars marked "?"

Put images in  images/  and run:  python main.py
Results saved to  output/
"""

import os
import glob
import string
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- config
IMAGE_DIR     = "images"
OUTPUT_DIR    = "output"
MAX_WIDTH     = 1280
INDIAN_FORMAT = False
USE_OCR       = True
FONT_PATH     = "C:/Windows/Fonts/arialbd.ttf"

PLATE_W, PLATE_H = 520, 110
CHAR_W,  CHAR_H  = 40, 60
CONFIDENCE_THRESH = 0.25

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ================================================================ PREPROCESSING
def prepare(img):
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        s = MAX_WIDTH / w
        img = cv2.resize(img, (MAX_WIDTH, int(h * s)), interpolation=cv2.INTER_AREA)
    return img


def enhance(gray):
    lut     = np.array([min(255, int((i / 255.0) ** 0.5 * 255))
                        for i in range(256)], dtype=np.uint8)
    gamma   = cv2.LUT(gray, lut)
    eq      = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gamma)
    denoise = cv2.bilateralFilter(eq, 9, 75, 75)
    return gamma, eq, denoise


# ================================================================ GEOMETRY
def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1); d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def warp_region(image, box):
    tl, tr, br, bl = order_points(box)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < 5 or h < 5:
        return None
    dst = np.array([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(order_points(box), dst)
    return cv2.warpPerspective(image, M, (w, h))


def deskew_plate(plate_gray):
    edges = cv2.Canny(plate_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=plate_gray.shape[1] // 4,
                            maxLineGap=10)
    if lines is None:
        return plate_gray
    angles = []
    for l in lines:
        pts = np.array(l).reshape(-1)
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        if abs(x2 - x1) > abs(y2 - y1):
            angles.append(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
    if not angles:
        return plate_gray
    angle = np.median(angles)
    if abs(angle) < 0.5 or abs(angle) > 15:
        return plate_gray
    h, w = plate_gray.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(plate_gray, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ================================================================ METHOD A - MORPHOLOGY
def detect_morph(img, gray_enhanced):
    edges = cv2.Canny(gray_enhanced, 20, 180)
    _, thresh = cv2.threshold(gray_enhanced, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_area = img.shape[0] * img.shape[1]
    out = dict(edges=edges, thresh=thresh, closed=None, box=None, plate=None)

    for kw, kh in [(17, 3), (25, 5), (9, 3), (35, 7), (45, 9)]:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kw, kh))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        closed = cv2.dilate(closed, None, iterations=1)
        if out["closed"] is None:
            out["closed"] = closed

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for c in contours:
            rect = cv2.minAreaRect(c)
            w, h = rect[1]
            if w < h: w, h = h, w
            if h < 6: continue
            aspect = w / h
            area   = w * h
            sol    = cv2.contourArea(c) / (area + 1e-6)
            if not (1.5 < aspect < 8.0 and 400 < area < 0.08 * img_area
                    and sol > 0.4):
                continue
            box = cv2.boxPoints(rect)
            roi = warp_region(gray_enhanced, box)
            if roi is None: continue
            _, roi_bw = cv2.threshold(roi, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ratio = cv2.countNonZero(roi_bw) / float(roi_bw.size)
            ratio = min(ratio, 1 - ratio)
            if not (0.06 < ratio < 0.55): continue
            e       = warp_region(edges, box)
            density = cv2.countNonZero(e) / float(e.size) if e is not None else 0
            candidates.append((density, box))

        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            box = candidates[0][1]
            out["closed"] = closed
            out["box"]    = box.astype(int)
            out["plate"]  = warp_region(img, box)
            break
    return out


# ================================================================ METHOD B - CHAR BLOBS
def _find_char_blobs(gray, polarity):
    H, W = gray.shape
    bs   = int(max(15, (min(H, W) // 18) | 1))
    bw   = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 polarity, bs, 6)
    n, _, st, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (0.010 * H <= h <= 0.20 * H): continue
        if not (0.10 <= w / h <= 1.4):        continue
        if a < 0.20 * w * h:                  continue
        blobs.append((x, y, w, h))
    return blobs


def _group_blobs(blobs):
    blobs = sorted(blobs, key=lambda b: b[0])
    n     = len(blobs)
    used  = [False] * n
    groups = []
    for i in range(n):
        if used[i]: continue
        chain = [i]; used[i] = True; changed = True
        while changed:
            changed = False
            last = blobs[chain[-1]]
            for j in range(n):
                if used[j]: continue
                b   = blobs[j]
                hr  = b[3] / last[3]
                dy  = abs((b[1] + b[3]/2) - (last[1] + last[3]/2))
                gap = b[0] - (last[0] + last[2])
                if (0.65 < hr < 1.55 and dy < 0.4 * last[3]
                        and -0.4 * last[3] <= gap < 1.8 * last[3]):
                    chain.append(j); used[j] = True; changed = True; break
        groups.append([blobs[k] for k in chain])
    return groups


def detect_chars(img, gray_enhanced):
    H, W = img.shape[:2]
    for scale in (1, 2):
        g = (gray_enhanced if scale == 1
             else cv2.resize(gray_enhanced, None, fx=2, fy=2,
                             interpolation=cv2.INTER_CUBIC))
        g = cv2.GaussianBlur(g, (3, 3), 0)
        best, best_score = None, 0
        for pol in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            for grp in _group_blobs(_find_char_blobs(g, pol)):
                if len(grp) < 4: continue
                x0 = min(b[0] for b in grp); x1 = max(b[0]+b[2] for b in grp)
                y0 = min(b[1] for b in grp); y1 = max(b[1]+b[3] for b in grp)
                hs  = np.array([b[3] for b in grp], dtype=float)
                consistency = 1 - hs.std() / (hs.mean() + 1e-6)
                aspect = (x1 - x0) / max(1, (y1 - y0))
                if not (1.4 < aspect < 10): continue
                score = min(len(grp), 10) + 4 * consistency
                if score > best_score:
                    best_score, best = score, (x0, y0, x1, y1, len(grp))
        if best:
            x0, y0, x1, y1, nc = [v / scale for v in best]
            bx0, by0, bx1, by1 = x0, y0, x1, y1
            h   = y1 - y0
            x0i = int(max(0, x0 - 0.6*h));  x1i = int(min(W, x1 + 0.6*h))
            y0i = int(max(0, y0 - 0.35*h)); y1i = int(min(H, y1 + 0.35*h))
            box   = np.array([[x0i,y0i],[x1i,y0i],[x1i,y1i],[x0i,y1i]])
            tx0 = int(max(0, bx0 - 0.1*h)); tx1 = int(min(W, bx1 + 0.1*h))
            ty0 = int(max(0, by0 - 0.1*h)); ty1 = int(min(H, by1 + 0.1*h))
            tight = img[ty0:ty1, tx0:tx1]
            return box, img[y0i:y1i, x0i:x1i], int(nc), tight
    return None, None, 0, None


# ================================================================ COMBINED DETECTOR
def detect_plate(img):
    gray             = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma, eq, den   = enhance(gray)
    res_a            = detect_morph(img, den)
    box_b, pl_b, nc, tight_b = detect_chars(img, eq)
    return {
        "gray": gray, "gamma": gamma, "clahe": eq, "blur": den,
        "edges": res_a["edges"], "thresh": res_a["thresh"],
        "closed": res_a["closed"],
        "box_a": res_a["box"],   "plate_a": res_a["plate"],
        "box":   box_b  if box_b  is not None else res_a["box"],
        "plate": pl_b   if pl_b   is not None else res_a["plate"],
        "tight": tight_b, "chars": nc,
    }


# ================================================================ OCR TEMPLATES
def build_templates():
    """
    Render A-Z and 0-9 in multiple Windows system fonts, then create
    morphological variants (erode / dilate / blur).
    Returns dict[char] -> list[np.ndarray uint8 CHAR_W x CHAR_H].
    """
    from PIL import Image, ImageDraw, ImageFont

    font_paths = [
        FONT_PATH, "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/arialn.ttf", "C:/Windows/Fonts/verdana.ttf",
        "C:/Windows/Fonts/courbd.ttf", "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/impact.ttf", "C:/Windows/Fonts/consola.ttf",
    ]
    fonts = []
    for fp in font_paths:
        try:
            fonts.append(ImageFont.truetype(fp, 80))
        except Exception:
            pass
    if not fonts:
        fonts = [ImageFont.load_default()]

    k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    k_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))

    templates = {}
    for ch in string.ascii_uppercase + string.digits:
        variants = []
        for font in fonts:
            im  = Image.new("L", (130, 140), 0)
            ImageDraw.Draw(im).text((15, 10), ch, fill=255, font=font)
            arr = np.array(im)
            ys, xs = np.where(arr > 20)
            if len(ys) == 0:
                continue
            arr  = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
            base = cv2.resize(arr, (CHAR_W, CHAR_H))
            variants.append(base)
            variants.append(cv2.erode(base,  k_e, iterations=1))
            variants.append(cv2.dilate(base, k_d, iterations=1))
            variants.append(cv2.GaussianBlur(base, (3, 3), 1))
            variants.append(cv2.dilate(base, k_d, iterations=2))
        if variants:
            templates[ch] = variants
    return templates


# Precompute Hu moments for templates (done once at build time)
def _hu(bw):
    m  = cv2.moments(bw)
    hu = cv2.HuMoments(m).flatten()
    with np.errstate(divide="ignore", invalid="ignore"):
        hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)
    return hu


def _precompute_hu(templates):
    """Return dict[char] -> list[hu_vector] matching the template variants."""
    result = {}
    for ch, variants in templates.items():
        result[ch] = [_hu((v > 50).astype(np.uint8) * 255) for v in variants]
    return result


# ================================================================ NORMALISE PLATE
def normalise_plate(plate_bgr):
    """
    BGR plate crop -> clean binary image at PLATE_W x PLATE_H.

    Pipeline:
      1. Grayscale + trim frame
      2. Deskew (Hough angle correction)
      3. Resize to PLATE_W x PLATE_H
      4. Percentile contrast stretch (2nd-98th)
      5. Unsharp mask sharpening
      6. Multi-strategy binarisation:
           - Otsu at blur sigma 0,1,2,3,4 (both polarities)
           - Adaptive at block 19 and 31 (both polarities)
         -> pick whichever produces the most character-shaped blobs
      7. Morphological open (remove specks) + close (fill gaps)
      8. Row-profile crop to isolate character band
    """
    g = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    h0, w0 = g.shape
    g = g[int(0.07*h0): int(0.93*h0), int(0.03*w0): int(0.97*w0)]
    if g.size == 0:
        return None

    g = deskew_plate(g)
    g = cv2.resize(g, (PLATE_W, PLATE_H), interpolation=cv2.INTER_CUBIC)

    lo, hi = np.percentile(g, 2), np.percentile(g, 98)
    if hi > lo:
        g = np.clip((g.astype(np.float32) - lo) / (hi - lo) * 255,
                    0, 255).astype(np.uint8)

    blur_s = cv2.GaussianBlur(g, (0, 0), 1.5)
    g = cv2.addWeighted(g, 1.8, blur_s, -0.8, 0)
    g = np.clip(g, 0, 255).astype(np.uint8)

    # -- score function: count character-like blobs --
    def _score(bw):
        bw2 = bw.copy()
        bw2[:4, :] = 0; bw2[-4:, :] = 0; bw2[:, :4] = 0; bw2[:, -4:] = 0
        cnts, _ = cv2.findContours(bw2, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        good = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if (PLATE_H * 0.28 < h < PLATE_H * 0.95
                    and 0.12 < w / h < 1.15
                    and cv2.contourArea(c) > 0.15 * w * h):
                good += 1
        return good

    cands = []
    # Otsu at multiple blur levels
    for sigma in (0, 1, 2, 3, 4):
        blurred = g if sigma == 0 else cv2.GaussianBlur(g, (sigma*2+1, sigma*2+1), sigma)
        _, bwn = cv2.threshold(blurred, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bwi = cv2.bitwise_not(bwn)
        cands.append((_score(bwn), bwn))
        cands.append((_score(bwi), bwi))

    # Adaptive at two block sizes
    for bs in (19, 31, 51):
        for pol in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            bwa = cv2.adaptiveThreshold(g, 255,
                                        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        pol, bs, 8)
            cands.append((_score(bwa), bwa))

    cands.sort(key=lambda t: t[0], reverse=True)
    best_bw = cands[0][1]

    # morphological cleanup
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    kc = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    best_bw = cv2.morphologyEx(best_bw, cv2.MORPH_OPEN,  ko, iterations=1)
    best_bw = cv2.morphologyEx(best_bw, cv2.MORPH_CLOSE, kc, iterations=1)

    # row-profile crop: keep only the character band
    rsum = best_bw.sum(axis=1).astype(np.float32)
    rsm  = np.convolve(rsum, np.ones(5) / 5, mode="same")
    thr  = 0.20 * rsm.max() if rsm.max() > 0 else 1
    ink  = np.where(rsm > thr)[0]
    if len(ink) >= 4:
        r0 = max(0, ink[0] - 2)
        r1 = min(best_bw.shape[0], ink[-1] + 3)
        best_bw = best_bw[r0:r1, :]

    return best_bw


# ================================================================ CHAR SEGMENTATION
def segment_chars_projection(bw):
    """Vertical projection profile -> column boundaries -> char cells."""
    row_sum = bw.sum(axis=1).astype(np.float32)
    rsm     = np.convolve(row_sum, np.ones(5) / 5, mode="same")
    ink_rows = np.where(rsm > 0.15 * rsm.max())[0]
    if len(ink_rows) == 0:
        return []
    band = bw[ink_rows[0]: ink_rows[-1] + 1, :]
    if band.shape[0] < 5:
        return []

    col_sum = band.sum(axis=0).astype(np.float32)
    csm     = np.convolve(col_sum, np.ones(3) / 3, mode="same")
    ink_t   = 0.08 * csm.max()

    in_ch, segs, start = False, [], 0
    for c in range(len(csm)):
        if csm[c] > ink_t:
            if not in_ch: start = c; in_ch = True
        else:
            if in_ch: segs.append((start, c)); in_ch = False
    if in_ch:
        segs.append((start, len(csm)))

    # merge narrow splits
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 4:
            merged[-1] = (merged[-1][0], s[1])
        elif s[1] - s[0] >= 4:
            merged.append(s)

    h_band = band.shape[0]
    valid = [(c0, c1) for c0, c1 in merged
             if c1 - c0 >= 3 and 0.10 < (c1 - c0) / h_band < 1.3]
    if not valid:
        return []
    return [cv2.resize(band[:, c0:c1], (CHAR_W, CHAR_H),
                       interpolation=cv2.INTER_AREA) for c0, c1 in valid]


def segment_chars_contour(bw):
    """Fallback: connected-component bounding boxes."""
    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    H = bw.shape[0]
    boxes = [(cv2.boundingRect(c)) for c in cnts]
    boxes = [(x, y, w, h) for x, y, w, h in boxes
             if H * 0.20 < h < H * 0.95 and w > 4
             and 0.12 < w / h < 1.2 and cv2.contourArea(
                 cnts[boxes.index((x,y,w,h))] if (x,y,w,h) in boxes else cnts[0]
             ) > 0.15 * w * h]
    # Simpler approach - refilter properly
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if (H * 0.20 < h < H * 0.95 and w > 4
                and 0.12 < w / h < 1.2
                and cv2.contourArea(c) > 0.15 * w * h):
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    if not boxes:
        return []
    return [cv2.resize(bw[y:y+h, x:x+w], (CHAR_W, CHAR_H))
            for x, y, w, h in boxes]


# ================================================================ CHARACTER MATCHING
def _match_one(cell, templates, tmpl_hu_cache):
    """
    Hybrid matcher: template matching (pixel) + Hu moments (shape).
    60% template-match score + 40% Hu-moment similarity.
    """
    cell_bw = (cell > 50).astype(np.uint8) * 255
    cell_hu = _hu(cell_bw)
    cell_f  = cell.astype(np.float32)

    best_ch, best_sc = "?", -1.0
    for ch, variants in templates.items():
        hu_list = tmpl_hu_cache.get(ch, [])
        for idx, tmpl in enumerate(variants):
            tm_sc = cv2.matchTemplate(cell_f, tmpl.astype(np.float32),
                                      cv2.TM_CCOEFF_NORMED)[0][0]
            if idx < len(hu_list):
                hu_dist = np.linalg.norm(cell_hu - hu_list[idx])
                hu_sim  = 1.0 / (1.0 + hu_dist / 3.0)
            else:
                hu_sim = 0.3
            combined = 0.60 * tm_sc + 0.40 * hu_sim
            if combined > best_sc:
                best_sc, best_ch = combined, ch
    return best_ch, best_sc


def recognize(chars, templates, tmpl_hu_cache):
    text = ""
    for cell in chars:
        ch, sc = _match_one(cell, templates, tmpl_hu_cache)
        text += ch if sc >= CONFIDENCE_THRESH else "?"
    return text


# ================================================================ OCR ENTRY POINT
def ocr_plate(plate_bgr, templates, tmpl_hu_cache=None):
    """
    Full OCR: normalise -> segment -> match -> text.
    Returns (text, bw_debug).
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return "", None
    if tmpl_hu_cache is None:
        tmpl_hu_cache = _precompute_hu(templates)

    bw = normalise_plate(plate_bgr)
    if bw is None:
        return "", None

    # projection first, contour fallback
    chars = segment_chars_projection(bw)
    if len(chars) < 3:
        chars_c = segment_chars_contour(bw)
        if len(chars_c) >= len(chars):
            chars = chars_c

    if not chars:
        return "", bw

    text = recognize(chars, templates, tmpl_hu_cache)
    text = text.strip("?")
    return text, bw


# ================================================================ POST-CORRECTION
_DIGIT_LIKE  = {"O": "0", "I": "1", "J": "1", "B": "8",
                "S": "5", "Z": "2", "G": "6", "Q": "0"}
_LETTER_LIKE = {"0": "O", "1": "I", "8": "B", "5": "S",
                "2": "Z", "6": "G"}


def fix_indian(text):
    if len(text) != 10:
        return text
    out = ""
    for i, ch in enumerate(text):
        if i in (2, 3, 6, 7, 8, 9):
            out += _DIGIT_LIKE.get(ch, ch)
        else:
            out += _LETTER_LIKE.get(ch, ch)
    return out


# ================================================================ OUTPUT
def save_figure(name, img, res, text, bw_debug=None):
    result = img.copy()
    if res.get("box_a") is not None:
        cv2.drawContours(result, [res["box_a"]], -1, (0, 0, 255), 2)
    if res.get("box") is not None:
        cv2.drawContours(result, [res["box"]], -1, (0, 255, 0), 3)

    stages = [
        ("Original",             cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        ("Grayscale",            res["gray"]),
        ("Gamma Corrected",      res["gamma"]),
        ("CLAHE",                res["clahe"]),
        ("Denoised (Bilateral)", res["blur"]),
        ("Edges (Canny)",        res["edges"]),
        ("Threshold (Otsu)",     res["thresh"]),
        ("Morphology (Closed)",  res["closed"]),
        ("Detection R=A G=B",    cv2.cvtColor(result, cv2.COLOR_BGR2RGB)),
    ]
    if res["plate"] is not None:
        stages.append(("Plate Crop",
                       cv2.cvtColor(res["plate"], cv2.COLOR_BGR2RGB)))
    if bw_debug is not None:
        stages.append((f"OCR Binary -> {text or '?'}", bw_debug))

    cols = 3
    rows = (len(stages) + cols - 1) // cols
    plt.figure(figsize=(15, 4 * rows))
    for i, (title, im) in enumerate(stages):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
        plt.title(title, fontsize=9)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_stages.png"), dpi=100)
    plt.close()
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_detected.png"), result)
    if res["plate"] is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_plate.png"), res["plate"])


# ================================================================ MAIN
def main():
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        paths += glob.glob(os.path.join(IMAGE_DIR, ext))
    paths.sort()
    if not paths:
        print("No images found in", IMAGE_DIR); return

    templates    = None
    tmpl_hu_cache = None
    if USE_OCR:
        try:
            templates     = build_templates()
            tmpl_hu_cache = _precompute_hu(templates)
            print("OCR: template matching ready")
        except Exception as e:
            print("OCR disabled:", e)

    print(f"{'Image':30} {'Det':5} OCR text")
    print("-" * 60)
    found = 0
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img  = cv2.imread(p)
        if img is None:
            print(f"{name:30} could not read"); continue

        img = prepare(img)
        res = detect_plate(img)

        text, bw_dbg = "", None
        if res["plate"] is not None:
            found += 1
            if templates:
                try:
                    text, bw_dbg = ocr_plate(res["plate"], templates, tmpl_hu_cache)
                    if (not text or len(text.replace("?", "")) < 2) \
                            and res.get("tight") is not None:
                        t2, bw2 = ocr_plate(res["tight"], templates, tmpl_hu_cache)
                        if len(t2.replace("?", "")) > len(text.replace("?", "")):
                            text, bw_dbg = t2, bw2
                except Exception as e:
                    print(f"  OCR error: {e}")
            if INDIAN_FORMAT and text:
                text = fix_indian(text)

        save_figure(name, img, res, text, bw_dbg)
        status = "YES" if res["plate"] is not None else "NO"
        print(f"{name:30} {status:5} {text}")

    print("-" * 60)
    print(f"Detected {found}/{len(paths)}. Results in output/")


if __name__ == "__main__":
    main()