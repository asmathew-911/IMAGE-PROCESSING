"""
Vehicle Number Plate Detection - classical image processing only (no AI/ML)
Put your images in the  images/  folder, run:  python main.py
Results are saved in  output/
"""
import os
import glob
import string
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")            # save figures to files (no popup windows)
import matplotlib.pyplot as plt

IMAGE_DIR = "images"
OUTPUT_DIR = "output"
INDIAN_FORMAT = False            # True only for Indian plates like MH02AB1234
USE_OCR = True                   # set False to only detect plates
OCR_MODE = "template"            # pure classical: segment chars + cv2.matchTemplate
MAX_WIDTH = 1280                 # only shrink if image is wider than this
PAD_X, PAD_Y = 0.6, 0.35         # extra margin around the character row when cropping the plate
FONT_PATH = "C:/Windows/Fonts/arialbd.ttf"   # font for template rendering (Pillow, no ML)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ---------------------------------------------------------------- helpers
def order_points(pts):
    pts = pts.reshape(4, 2).astype("float32")
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).ravel()
    return np.array([pts[np.argmin(s)], pts[np.argmin(d)],
                     pts[np.argmax(s)], pts[np.argmax(d)]], dtype="float32")


def warp_region(image, box):
    """Straighten a tilted quadrilateral into a flat rectangle."""
    tl, tr, br, bl = order_points(box)
    w = int(max(np.linalg.norm(br - bl), np.linalg.norm(tr - tl)))
    h = int(max(np.linalg.norm(tr - br), np.linalg.norm(tl - bl)))
    if w < 5 or h < 5:
        return None
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(order_points(box), dst)
    return cv2.warpPerspective(image, M, (w, h))


def prepare(img):
    """Resize (only if big) and black out timestamp strips at top/bottom."""
    h, w = img.shape[:2]
    if w > MAX_WIDTH:
        img = cv2.resize(img, (MAX_WIDTH, int(h * MAX_WIDTH / w)))
    h = img.shape[0]
    img = img.copy()
    fill = np.median(img.reshape(-1, 3), axis=0)   # fill with median colour (no fake edge)
    img[: int(0.10 * h), :] = fill       # top overlay text (timestamps)
    img[-int(0.07 * h):, :] = fill       # bottom overlay text
    return img


# ------------------------------------------------------------- detection
def detect_morph(img):
    """METHOD A: edges + morphology + contours (baseline, shown in red)."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    blur = cv2.bilateralFilter(equalized, 11, 17, 17)
    edges = cv2.Canny(blur, 30, 200)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    img_area = img.shape[0] * img.shape[1]
    out = dict(gray=gray, clahe=equalized, blur=blur, edges=edges,
               thresh=thresh, closed=None, box=None, plate=None)

    # try a few kernel sizes (plate size varies between images)
    for kw, kh in [(17, 3), (25, 5), (9, 3), (35, 7)]:
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
            if w < h:
                w, h = h, w
            if h < 8:
                continue
            aspect, area = w / h, w * h
            solidity = cv2.contourArea(c) / area
            if not (1.8 < aspect < 6.5 and 600 < area < 0.05 * img_area
                    and solidity > 0.5):
                continue
            box = cv2.boxPoints(rect)
            roi = warp_region(blur, box)
            if roi is None:
                continue
            # local Otsu inside the candidate: a plate has some dark characters
            # on a light background (or the reverse)
            _, roi_bw = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            ratio = cv2.countNonZero(roi_bw) / float(roi_bw.size)
            ratio = min(ratio, 1 - ratio)       # fraction of the minority colour
            if 0.08 < ratio < 0.5:
                # score: dense edges inside = text-like
                e = warp_region(edges, box)
                density = cv2.countNonZero(e) / float(e.size) if e is not None else 0
                candidates.append((density, box))

        if candidates:
            candidates.sort(key=lambda t: t[0], reverse=True)
            box = candidates[0][1]
            out["closed"] = closed
            out["box"] = box.astype(int)
            out["plate"] = warp_region(img, box)
            break
    return out


# ------------------------------------------------- METHOD B: character grouping
# Idea: a plate is a ROW of 4+ similar-sized, evenly spaced character blobs.
def find_char_blobs(gray, polarity):
    H, W = gray.shape
    bs = int(max(15, (min(H, W) // 20) | 1))
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               polarity, bs, 6)
    n, lab, st, cen = cv2.connectedComponentsWithStats(bw, connectivity=8)
    blobs = []
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if not (0.012 * H <= h <= 0.16 * H):
            continue
        if not (0.12 <= w / h <= 1.2):
            continue
        if a < 0.25 * w * h:
            continue
        blobs.append((x, y, w, h))
    return blobs

def group_blobs(blobs):
    """Chain blobs that sit in one row, have similar height and are close."""
    blobs = sorted(blobs, key=lambda b: b[0])
    n = len(blobs)
    used = [False] * n
    groups = []
    for i in range(n):
        if used[i]:
            continue
        chain = [i]
        used[i] = True
        changed = True
        while changed:
            changed = False
            last = blobs[chain[-1]]
            for j in range(n):
                if used[j]:
                    continue
                b = blobs[j]
                hr = b[3] / last[3]
                dy = abs((b[1] + b[3] / 2) - (last[1] + last[3] / 2))
                gap = b[0] - (last[0] + last[2])
                if 0.7 < hr < 1.43 and dy < 0.35 * last[3] and -0.3 * last[3] <= gap < 1.6 * last[3]:
                    chain.append(j); used[j] = True; changed = True
                    break
        groups.append([blobs[k] for k in chain])
    return groups


def detect_chars(img):
    """METHOD B (final): returns box + crop, or None. Retries on 2x image for tiny plates."""
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    for scale in (1, 2):
        g = clahe if scale == 1 else cv2.resize(clahe, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        best, best_score = None, 0
        for pol in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
            for grp in group_blobs(find_char_blobs(g, pol)):
                if len(grp) < 4:
                    continue
                x0 = min(b[0] for b in grp); x1 = max(b[0] + b[2] for b in grp)
                y0 = min(b[1] for b in grp); y1 = max(b[1] + b[3] for b in grp)
                hs = np.array([b[3] for b in grp])
                consistency = 1 - hs.std() / hs.mean()
                aspect = (x1 - x0) / max(1, (y1 - y0))
                if not (1.5 < aspect < 9):
                    continue
                score = min(len(grp), 10) + 3 * consistency
                if score > best_score:
                    best_score, best = score, (x0, y0, x1, y1, len(grp))
        if best:
            x0, y0, x1, y1, n = [v / scale for v in best]
            bx0, by0, bx1, by1 = x0, y0, x1, y1      # raw character-row box
            h = y1 - y0
            padx, pady = PAD_X * h, PAD_Y * h
            x0 = int(max(0, x0 - padx)); x1 = int(min(W, x1 + padx))
            y0 = int(max(0, y0 - pady)); y1 = int(min(H, y1 + pady))
            box = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            # tight crop (little margin) - recognisers often like this better
            tx0 = int(max(0, bx0 - 0.15 * h)); tx1 = int(min(W, bx1 + 0.15 * h))
            ty0 = int(max(0, by0 - 0.12 * h)); ty1 = int(min(H, by1 + 0.12 * h))
            tight = img[ty0:ty1, tx0:tx1]
            return box, img[y0:y1, x0:x1], int(n), tight
    return None, None, 0, None


def detect_plate(img):
    res = detect_morph(img)                 # Method A (stages + baseline box)
    res["box_a"], res["plate_a"] = res["box"], res["plate"]
    box, plate, n, tight = detect_chars(img)   # Method B (final answer)
    res["box"], res["plate"], res["chars"], res["tight"] = box, plate, n, tight
    return res


# ------------------------------------------------------------------- OCR
def segment_chars(plate_img):
    g = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    h0, w0 = g.shape
    g = g[int(0.08 * h0):int(0.92 * h0), int(0.03 * w0):int(0.97 * w0)]   # trim plate frame
    g = cv2.resize(g, (400, 100))
    g = cv2.GaussianBlur(g, (3, 3), 0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    bw[:4, :] = 0; bw[-4:, :] = 0; bw[:, :4] = 0; bw[:, -4:] = 0

    cnts, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if 22 < h < 95 and 3 < w < 100:
            boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: b[0])
    chars = [cv2.resize(bw[y:y + h, x:x + w], (40, 60)) for x, y, w, h in boxes]
    return chars


def build_templates():
    from PIL import Image, ImageDraw, ImageFont
    font = ImageFont.truetype(FONT_PATH, 80)
    templates = {}
    for ch in string.ascii_uppercase + string.digits:
        im = Image.new("L", (100, 120), 0)
        ImageDraw.Draw(im).text((10, 10), ch, fill=255, font=font)
        arr = np.array(im)
        ys, xs = np.where(arr > 0)
        arr = arr[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
        templates[ch] = cv2.resize(arr, (40, 60))
    return templates


def recognize(chars, templates):
    text = ""
    for ch in chars:
        best, best_score = "?", -1
        for label, t in templates.items():
            score = cv2.matchTemplate(ch, t, cv2.TM_CCOEFF_NORMED)[0][0]
            if score > best_score:
                best, best_score = label, score
        text += best
    return text




def fix_indian(text):
    """Indian format: 2 letters, 2 digits, 2 letters, 4 digits (10 chars)."""
    if len(text) != 10:
        return text
    to_digit = {"O": "0", "I": "1", "J": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}
    to_letter = {v: k for k, v in {"O": "0", "I": "1", "B": "8", "S": "5", "Z": "2", "G": "6"}.items()}
    out = ""
    for i, ch in enumerate(text):
        if i in (2, 3, 6, 7, 8, 9):
            out += to_digit.get(ch, ch)
        else:
            out += to_letter.get(ch, ch)
    return out


# ------------------------------------------------------------ save figure
def save_figure(name, img, res, text):
    result = img.copy()
    if res.get("box_a") is not None:
        cv2.drawContours(result, [res["box_a"]], -1, (0, 0, 255), 2)   # Method A = red
    if res["box"] is not None:
        cv2.drawContours(result, [res["box"]], -1, (0, 255, 0), 3)     # Method B = green

    stages = [("Original", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
              ("Grayscale", res["gray"]), ("CLAHE", res["clahe"]),
              ("Denoised", res["blur"]), ("Edges (Canny)", res["edges"]),
              ("Threshold", res["thresh"]), ("Morphology", res["closed"]),
              ("Red = A, Green = B (final)", cv2.cvtColor(result, cv2.COLOR_BGR2RGB))]
    if res["plate"] is not None:
        stages.append((f"Plate: {text}", cv2.cvtColor(res["plate"], cv2.COLOR_BGR2RGB)))

    plt.figure(figsize=(15, 9))
    for i, (title, im) in enumerate(stages):
        plt.subplot(3, 3, i + 1)
        plt.imshow(im, cmap="gray" if im.ndim == 2 else None)
        plt.title(title)
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{name}_stages.png"), dpi=100)
    plt.close()
    cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_detected.png"), result)
    if res["plate"] is not None:
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"{name}_plate.png"), res["plate"])


# ------------------------------------------------------------------ main
def main():
    paths = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
        paths += glob.glob(os.path.join(IMAGE_DIR, ext))
    paths.sort()
    if not paths:
        print("No images found in", IMAGE_DIR)
        return

    # ---- classical-only OCR: cv2.matchTemplate against rendered font templates
    templates = None
    mode = None
    if USE_OCR:
        try:
            templates = build_templates()
            mode = "template"
        except Exception as e:
            print("OCR disabled (font problem):", e)

    print(f"OCR mode: {mode}")
    print(f"{'Image':30} {'Detected':10} OCR text")
    print("-" * 55)
    found = 0
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        if img is None:
            print(f"{name:30} could not read file")
            continue
        img = prepare(img)
        res = detect_plate(img)

        text = ""
        if res["plate"] is not None:
            found += 1
            try:
                if mode == "template" and templates:
                    text = recognize(segment_chars(res["plate"]), templates)
            except Exception as e:
                text = ""
                print("  OCR error:", e)
            if INDIAN_FORMAT and text:
                text = fix_indian(text)
        save_figure(name, img, res, text)
        print(f"{name:30} {'YES' if res['plate'] is not None else 'NO':10} {text}")

    print("-" * 55)
    print(f"Detected {found} of {len(paths)} images. Check the output/ folder.")


if __name__ == "__main__":
    main()