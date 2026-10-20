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
INDIAN_FORMAT = True
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
    Render A-Z and 0-9 using:
    1. Built-in OpenCV Hershey vector fonts (100% OS-independent, renders on any OS/Cloud)
    2. Matplotlib bundled TrueType fonts (DejaVuSans shipped inside matplotlib wheel)
    3. System TrueType fonts if available (Windows / Linux)
    Creates morphological variants (erode / dilate / blur).
    Returns dict[char] -> list[np.ndarray uint8 CHAR_W x CHAR_H].
    """
    from PIL import Image, ImageDraw, ImageFont

    templates = {ch: [] for ch in string.ascii_uppercase + string.digits}
    k_e = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    k_d = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))

    # 1. OpenCV built-in Hershey fonts (vector fonts compiled into OpenCV)
    cv_font_configs = [
        (cv2.FONT_HERSHEY_SIMPLEX, 2.1, 4),
        (cv2.FONT_HERSHEY_DUPLEX,  2.1, 4),
        (cv2.FONT_HERSHEY_COMPLEX, 2.1, 4),
        (cv2.FONT_HERSHEY_TRIPLEX, 2.1, 4),
    ]
    for face, scale, thick in cv_font_configs:
        for ch in templates.keys():
            canvas = np.zeros((140, 120), dtype=np.uint8)
            (tw, th), _ = cv2.getTextSize(ch, face, scale, thick)
            x = max(5, (120 - tw) // 2)
            y = max(th + 5, (140 + th) // 2)
            cv2.putText(canvas, ch, (x, y), face, scale, 255, thick, cv2.LINE_AA)
            ys, xs = np.where(canvas > 30)
            if len(ys) == 0:
                continue
            crop = canvas[ys.min():ys.max()+1, xs.min():xs.max()+1]
            base = cv2.resize(crop, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
            templates[ch].append(base)
            templates[ch].append(cv2.erode(base, k_e, iterations=1))
            templates[ch].append(cv2.dilate(base, k_d, iterations=1))

    # 2. Matplotlib bundled fonts (cross-platform, reliable on Render / Linux)
    mpl_ttf = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
    ttfs = []
    if os.path.exists(mpl_ttf):
        for name in ["DejaVuSans-Bold.ttf", "DejaVuSans.ttf", "DejaVuSansMono-Bold.ttf"]:
            p = os.path.join(mpl_ttf, name)
            if os.path.exists(p):
                ttfs.append(p)

    # 3. System font paths (Windows & Linux)
    for p in [FONT_PATH, "C:/Windows/Fonts/arialbd.ttf", "C:/Windows/Fonts/arial.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"]:
        if os.path.exists(p) and p not in ttfs:
            ttfs.append(p)

    for ttf in ttfs:
        try:
            fnt = ImageFont.truetype(ttf, 75)
            for ch in templates.keys():
                im = Image.new("L", (130, 140), 0)
                ImageDraw.Draw(im).text((15, 10), ch, fill=255, font=fnt)
                arr = np.array(im)
                ys, xs = np.where(arr > 30)
                if len(ys) == 0:
                    continue
                crop = arr[ys.min():ys.max()+1, xs.min():xs.max()+1]
                base = cv2.resize(crop, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
                templates[ch].append(base)
                templates[ch].append(cv2.erode(base, k_e, iterations=1))
                templates[ch].append(cv2.dilate(base, k_d, iterations=1))
        except Exception:
            pass

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

    Classical CV Pipeline:
      1. Trim outer perimeter margins (removes vehicle bumper/bracket noise)
      2. Deskew (Hough transform angle estimation & affine rotation)
      3. Resize to canonical PLATE_W x PLATE_H (520 x 110)
      4. CLAHE local contrast enhancement & unsharp masking
      5. Multi-candidate binarisation (Otsu & Adaptive Gaussian)
      6. Polarity normalisation (ensures white characters on black background)
      7. Border margin suppression & morphological opening
    """
    if plate_bgr is None or plate_bgr.size == 0:
        return None
    h0, w0 = plate_bgr.shape[:2]
    # Trim outer margin to discard mounting frames and vehicle bumper borders
    trim = plate_bgr[int(0.06 * h0): int(0.94 * h0), int(0.02 * w0): int(0.98 * w0)]
    if trim.size == 0:
        trim = plate_bgr

    g = cv2.cvtColor(trim, cv2.COLOR_BGR2GRAY)
    g = deskew_plate(g)
    g = cv2.resize(g, (PLATE_W, PLATE_H), interpolation=cv2.INTER_CUBIC)

    # Contrast enhancement & sharpening
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    g = clahe.apply(g)
    blur_s = cv2.GaussianBlur(g, (0, 0), 1.5)
    g = cv2.addWeighted(g, 1.8, blur_s, -0.8, 0)
    g = np.clip(g, 0, 255).astype(np.uint8)

    cands = []
    # 1. Otsu thresholding across multiple Gaussian smoothing levels
    for sigma in (0, 1, 2):
        bg = g if sigma == 0 else cv2.GaussianBlur(g, (sigma * 2 + 1, sigma * 2 + 1), sigma)
        _, bwn = cv2.threshold(bg, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Check border polarity: if borders are predominantly white, plate has white bg -> invert
        b_mean = (np.mean(bwn[:4, :]) + np.mean(bwn[-4:, :]) + np.mean(bwn[:, :4]) + np.mean(bwn[:, -4:])) / 4.0
        if b_mean > 120:
            bwn = cv2.bitwise_not(bwn)
        cands.append(bwn)

    # 2. Adaptive Gaussian thresholding across multiple window sizes
    for bs in (19, 31, 45):
        for pol in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            bwa = cv2.adaptiveThreshold(g, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, pol, bs, 7)
            b_mean = (np.mean(bwa[:4, :]) + np.mean(bwa[-4:, :]) + np.mean(bwa[:, :4]) + np.mean(bwa[:, -4:])) / 4.0
            if b_mean > 120:
                bwa = cv2.bitwise_not(bwa)
            cands.append(bwa)

    # Rank candidates by number of character-like connected components
    best_bw, best_score = cands[0], -1
    for bw in cands:
        bw_clean = bw.copy()
        bw_clean[:4, :] = 0; bw_clean[-4:, :] = 0; bw_clean[:, :4] = 0; bw_clean[:, -4:] = 0
        cnts, _ = cv2.findContours(bw_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        good = 0
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            if (PLATE_H * 0.25 < h < PLATE_H * 0.95 and 6 < w < PLATE_H * 1.1 and 0.12 < w / h < 1.2):
                good += 1
        if good > best_score:
            best_score = good
            best_bw = bw_clean

    # Clear border margin to disconnect any bezel frame remnants
    best_bw[:3, :] = 0; best_bw[-3:, :] = 0; best_bw[:, :3] = 0; best_bw[:, -3:] = 0

    # Morphological opening (removes 1-2 pixel isolated salt-and-pepper noise)
    ko = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    best_bw = cv2.morphologyEx(best_bw, cv2.MORPH_OPEN, ko, iterations=1)
    return best_bw


# ================================================================ CHAR SEGMENTATION
def segment_chars_projection(bw):
    """Vertical projection profile -> column boundaries -> char cells."""
    row_sum = bw.sum(axis=1).astype(np.float32)
    rsm     = np.convolve(row_sum, np.ones(5) / 5, mode="same")
    ink_rows = np.where(rsm > 0.12 * rsm.max())[0] if rsm.max() > 0 else []
    if len(ink_rows) == 0:
        return []
    band = bw[ink_rows[0]: ink_rows[-1] + 1, :]
    if band.shape[0] < 5:
        return []

    col_sum = band.sum(axis=0).astype(np.float32)
    csm     = np.convolve(col_sum, np.ones(3) / 3, mode="same")
    ink_t   = 0.08 * csm.max() if csm.max() > 0 else 1

    in_ch, segs, start = False, [], 0
    for c in range(len(csm)):
        if csm[c] > ink_t:
            if not in_ch: start = c; in_ch = True
        else:
            if in_ch: segs.append((start, c)); in_ch = False
    if in_ch:
        segs.append((start, len(csm)))

    # Merge narrow splits
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 4:
            merged[-1] = (merged[-1][0], s[1])
        elif s[1] - s[0] >= 4:
            merged.append(s)

    h_band = band.shape[0]
    valid = [(c0, c1) for c0, c1 in merged
             if c1 - c0 >= 4 and 0.10 < (c1 - c0) / float(h_band) < 1.35]
    if not valid:
        return []

    res = []
    for c0, c1 in valid:
        patch = band[:, c0:c1]
        aspect = (c1 - c0) / float(h_band)
        pad_x = max(1, int(0.08 * (c1 - c0)))
        pad_y = max(1, int(0.08 * h_band))
        padded = cv2.copyMakeBorder(patch, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=0)
        cell = cv2.resize(padded, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
        res.append((cell, aspect))
    return res


def segment_chars_contour(bw):
    """
    Connected-component character segmentation with hierarchy filtering:
    - RETR_CCOMP retrieves external letters and enclosed holes.
    - Suppresses duplicate/nested inner contours.
    - Mask isolation separates adjacent touches and lines.
    - Preserves aspect ratio for geometric matching.
    """
    H, W = bw.shape
    cnts, hier = cv2.findContours(bw.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return []

    valid = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        aspect = w / float(h)
        area = cv2.contourArea(c)
        if (0.22 * H < h < 0.96 * H and
            5 < w < 0.30 * W and
            0.10 < aspect < 1.30 and
            area > 0.08 * w * h):
            valid.append((x, y, w, h, c, aspect))

    if not valid:
        return []

    # Sort by area descending for non-maximum suppression of nested holes
    valid.sort(key=lambda b: b[2] * b[3], reverse=True)
    kept = []
    for b in valid:
        bx, by, bw_, bh_, bc, basp = b
        overlap = False
        for kx, ky, kw, kh, _, _ in kept:
            if bx >= kx - 2 and by >= ky - 2 and (bx + bw_) <= (kx + kw + 2) and (by + bh_) <= (ky + kh + 2):
                overlap = True
                break
        if not overlap:
            kept.append(b)

    # Sort strictly left to right
    kept.sort(key=lambda b: b[0])

    char_cells = []
    for x, y, w, h, c, aspect in kept:
        cell_mask = np.zeros(bw.shape, dtype=np.uint8)
        cv2.drawContours(cell_mask, [c], -1, 255, -1)
        isolated = cv2.bitwise_and(bw, bw, mask=cell_mask)[y:y+h, x:x+w]

        pad_y = max(1, int(0.08 * h))
        pad_x = max(1, int(0.08 * w))
        isolated = cv2.copyMakeBorder(isolated, pad_y, pad_y, pad_x, pad_x,
                                      cv2.BORDER_CONSTANT, value=0)
        cell = cv2.resize(isolated, (CHAR_W, CHAR_H), interpolation=cv2.INTER_AREA)
        char_cells.append((cell, aspect))

    return char_cells


# ================================================================ CHARACTER MATCHING
def _match_one(cell_info, templates, tmpl_hu_cache):
    """
    Hybrid classical matcher:
      1. Normalized Cross-Correlation (TM_CCOEFF_NORMED)
      2. Jaccard Index (IoU) overlap to strictly penalize missing/extra strokes
      3. Hu Moments shape descriptors
      4. Geometric aspect ratio penalty (e.g. 'I'/'1' cannot be wide)
    """
    if isinstance(cell_info, tuple):
        cell, box_aspect = cell_info
    else:
        cell, box_aspect = cell_info, 0.50

    cell_bin = (cell > 50).astype(np.uint8)
    if cell_bin.sum() < 30:
        return "?", 0.0

    cell_hu = _hu(cell_bin * 255)
    cell_f  = cell.astype(np.float32)

    best_ch, best_sc = "?", -1.0
    for ch, variants in templates.items():
        hu_list = tmpl_hu_cache.get(ch, [])
        for idx, tmpl in enumerate(variants):
            # 1. Pearson correlation
            res = cv2.matchTemplate(cell_f, tmpl.astype(np.float32), cv2.TM_CCOEFF_NORMED)[0][0]
            tm_sc = float(res) if not np.isnan(res) else 0.0

            # 2. IoU / Jaccard pixel overlap
            tmpl_bin = (tmpl > 50).astype(np.uint8)
            inter = np.logical_and(cell_bin, tmpl_bin).sum()
            union = np.logical_or(cell_bin, tmpl_bin).sum()
            iou = inter / float(union) if union > 0 else 0.0

            # 3. Hu moment shape similarity
            if idx < len(hu_list):
                hu_dist = np.linalg.norm(cell_hu - hu_list[idx])
                hu_sim  = 1.0 / (1.0 + hu_dist / 3.0)
            else:
                hu_sim = 0.3

            # Combined score
            combined = 0.45 * max(0.0, tm_sc) + 0.40 * iou + 0.15 * hu_sim

            # Aspect ratio geometric filter
            if ch in ("I", "1") and box_aspect > 0.48:
                combined *= 0.40
            elif ch in ("W", "M", "Q", "O", "0") and box_aspect < 0.28:
                combined *= 0.40

            if combined > best_sc:
                best_sc, best_ch = combined, ch

    return best_ch, best_sc


INDIAN_PATTERN = ["L", "L", "D", "D", "L", "L", "D", "D", "D", "D"]


def recognize(chars, templates, tmpl_hu_cache):
    if not chars:
        return ""

    # If Indian format is active and we have candidate characters:
    if INDIAN_FORMAT and len(chars) >= 10:
        if len(chars) == 10:
            text = ""
            for i, cell in enumerate(chars):
                exp = INDIAN_PATTERN[i]
                allowed = string.ascii_uppercase if exp == "L" else string.digits
                sub_tmpls = {k: v for k, v in templates.items() if k in allowed}
                ch, sc = _match_one(cell, sub_tmpls, tmpl_hu_cache)
                text += ch if sc >= 0.18 else "?"
            return text
        else:
            # More than 10 detected (rivets/artifacts): select best 10 aligned subset
            best_text, best_sc = "", -1
            for offset in range(len(chars) - 9):
                cand = chars[offset: offset + 10]
                cand_text = ""
                total_sc = 0.0
                for i, cell in enumerate(cand):
                    exp = INDIAN_PATTERN[i]
                    allowed = string.ascii_uppercase if exp == "L" else string.digits
                    sub_tmpls = {k: v for k, v in templates.items() if k in allowed}
                    ch, sc = _match_one(cell, sub_tmpls, tmpl_hu_cache)
                    cand_text += ch if sc >= 0.18 else "?"
                    total_sc += sc
                if total_sc > best_sc:
                    best_sc = total_sc
                    best_text = cand_text
            return best_text

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

    # Dual-strategy character segmentation: pick strategy yielding cleaner character set
    chars_p = segment_chars_projection(bw)
    chars_c = segment_chars_contour(bw)
    chars = chars_c if len(chars_c) >= len(chars_p) else chars_p

    if not chars:
        return "", bw

    text = recognize(chars, templates, tmpl_hu_cache)
    text = text.strip("?")
    return text, bw


# ================================================================ POST-CORRECTION
_DIGIT_MAP = {
    "O": "0", "D": "0", "Q": "0", "U": "0",
    "I": "1", "J": "1", "L": "1", "T": "7",
    "Z": "2",
    "E": "3",
    "A": "4",
    "S": "5",
    "G": "6", "C": "0",
    "B": "8", "R": "8",
    "P": "9", "Y": "7", "V": "7"
}

_LETTER_MAP = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "3": "E",
    "4": "A",
    "5": "S",
    "6": "G",
    "7": "T",
    "8": "B",
    "9": "P"
}


def fix_indian(text):
    """
    Enforces standard Indian license plate format:
    2 Letters (State), 2 Numbers (RTO), 2 Letters (Series), 4 Numbers (Unique ID)
    Format: LL NN LL NNNN (e.g., MH 20 CS 1941)
    """
    if not text:
        return text

    # Extract all alphanumeric characters
    cleaned = [c for c in text.upper() if c.isalnum() or c == "?"]
    if not cleaned:
        return text

    pattern = ["L", "L", "N", "N", "L", "L", "N", "N", "N", "N"]
    chars_to_fix = cleaned[:10]
    out = []
    for i, ch in enumerate(chars_to_fix):
        exp = pattern[i] if i < len(pattern) else "N"
        if ch == "?":
            out.append(ch)
        elif exp == "L":
            # Must be letter
            out.append(_LETTER_MAP.get(ch, "A") if ch.isdigit() else ch)
        else:
            # Must be number
            out.append(_DIGIT_MAP.get(ch, "0") if ch.isalpha() else ch)

    res = "".join(out)
    # Format with standard spaces: LL NN LL NNNN
    if len(res) == 10:
        return f"{res[0:2]} {res[2:4]} {res[4:6]} {res[6:10]}"
    elif len(res) >= 6:
        parts = []
        if len(res) >= 2: parts.append(res[0:2])
        if len(res) >= 4: parts.append(res[2:4])
        if len(res) >= 6: parts.append(res[4:6])
        if len(res) > 6:  parts.append(res[6:])
        return " ".join(parts)
    return res


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