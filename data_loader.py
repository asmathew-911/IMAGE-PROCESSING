"""
data_loader.py  –  Load images + ground-truth boxes from both datasets.

Dataset 4 (primary):  YOLO format  → class x_center y_center width height  (normalised)
Dataset 1 (supplement): Pascal VOC XML → xmin ymin xmax ymax  + plate text
"""

import os
import glob
import xml.etree.ElementTree as ET
import cv2
import numpy as np


# ---------------------------------------------------------------- YOLO loader
def parse_yolo_label(label_path, img_w, img_h):
    """
    Read a YOLO .txt file and return a list of bounding boxes.
    Each box is (xmin, ymin, xmax, ymax) in pixel coordinates.
    """
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            # class_id, x_center, y_center, w, h  (all normalised 0-1)
            _, xc, yc, w, h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            xc, yc, w, h = xc * img_w, yc * img_h, w * img_w, h * img_h
            xmin = int(xc - w / 2)
            ymin = int(yc - h / 2)
            xmax = int(xc + w / 2)
            ymax = int(yc + h / 2)
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes


def load_yolo_dataset(dataset_dir, split="test", max_images=None):
    """
    Loads images and ground truth from a YOLO-format dataset.
    Expected structure:
        dataset_dir/
            <split>/
                images/   *.jpg
                labels/   *.txt

    Returns list of dicts:  {"path", "image", "gt_boxes": [(xmin,ymin,xmax,ymax), ...]}
    """
    img_dir = os.path.join(dataset_dir, split, "images")
    lbl_dir = os.path.join(dataset_dir, split, "labels")

    if not os.path.isdir(img_dir):
        # try flat structure: dataset_dir/images/split/
        img_dir = os.path.join(dataset_dir, "images", split)
        lbl_dir = os.path.join(dataset_dir, "labels", split)

    if not os.path.isdir(img_dir):
        # try nested subdirectories (e.g., dataset_dir/*/split/images)
        subdirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
        for sub in subdirs:
            cand_img = os.path.join(dataset_dir, sub, split, "images")
            cand_lbl = os.path.join(dataset_dir, sub, split, "labels")
            if os.path.isdir(cand_img):
                img_dir, lbl_dir = cand_img, cand_lbl
                break
            cand_img = os.path.join(dataset_dir, sub, "images", split)
            cand_lbl = os.path.join(dataset_dir, sub, "labels", split)
            if os.path.isdir(cand_img):
                img_dir, lbl_dir = cand_img, cand_lbl
                break

    if not os.path.isdir(img_dir):
        print(f"[WARN] Image directory not found: {img_dir}")
        return []

    paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
    if max_images:
        paths = paths[:max_images]

    data = []
    for p in paths:
        ext = os.path.splitext(p)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            continue
        img = cv2.imread(p)
        if img is None:
            continue
        h, w = img.shape[:2]
        name = os.path.splitext(os.path.basename(p))[0]
        lbl = os.path.join(lbl_dir, name + ".txt")
        boxes = parse_yolo_label(lbl, w, h)
        data.append({"path": p, "name": name, "image": img, "gt_boxes": boxes})
    return data


# ---------------------------------------------------------- Pascal VOC loader
def parse_voc_xml(xml_path):
    """
    Read a Pascal VOC XML annotation file.
    Returns list of dicts: {"box": (xmin,ymin,xmax,ymax), "text": plate_text}
    """
    results = []
    if not os.path.exists(xml_path):
        return results
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        for obj in root.findall("object"):
            name = obj.find("name")
            plate_text = name.text.strip() if name is not None else ""
            bbox = obj.find("bndbox")
            if bbox is None:
                continue
            xmin = int(float(bbox.find("xmin").text))
            ymin = int(float(bbox.find("ymin").text))
            xmax = int(float(bbox.find("xmax").text))
            ymax = int(float(bbox.find("ymax").text))
            results.append({"box": (xmin, ymin, xmax, ymax), "text": plate_text})
    except ET.ParseError:
        pass
    return results


def load_voc_dataset(dataset_dir, max_images=None):
    """
    Loads images and annotations from Dataset 1 (Pascal VOC XML).
    Searches recursively for image+XML pairs.

    Returns list of dicts:
        {"path", "image", "gt_boxes": [(xmin,ymin,xmax,ymax),...], "plate_texts": [str,...]}
    """
    # find all XML files recursively
    xml_files = sorted(glob.glob(os.path.join(dataset_dir, "**", "*.xml"), recursive=True))
    if max_images:
        xml_files = xml_files[:max_images]

    data = []
    for xf in xml_files:
        annots = parse_voc_xml(xf)
        if not annots:
            continue
        # find matching image (same name, different extension)
        base = os.path.splitext(xf)[0]
        img_path = None
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
            candidate = base + ext
            if os.path.exists(candidate):
                img_path = candidate
                break
        if img_path is None:
            # try in same directory
            directory = os.path.dirname(xf)
            name = os.path.splitext(os.path.basename(xf))[0]
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp"):
                candidate = os.path.join(directory, name + ext)
                if os.path.exists(candidate):
                    img_path = candidate
                    break
        if img_path is None:
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        boxes = [a["box"] for a in annots]
        texts = [a["text"] for a in annots]
        name = os.path.splitext(os.path.basename(img_path))[0]
        data.append({
            "path": img_path,
            "name": name,
            "image": img,
            "gt_boxes": boxes,
            "plate_texts": texts,
        })
    return data


# ------------------------------------------------------------------ metrics
def compute_iou(box_a, box_b):
    """
    Compute Intersection over Union between two boxes.
    Each box is (xmin, ymin, xmax, ymax).
    """
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter = max(0, xb - xa) * max(0, yb - ya)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter

    if union == 0:
        return 0.0
    return inter / union


def detection_box_from_result(res):
    """
    Convert the detection result's box (4 corner points) into
    (xmin, ymin, xmax, ymax) format for IoU comparison.
    """
    box = res.get("box")
    if box is None:
        return None
    pts = box.reshape(-1, 2) if hasattr(box, "reshape") else np.array(box).reshape(-1, 2)
    xmin, ymin = pts.min(axis=0)
    xmax, ymax = pts.max(axis=0)
    return (int(xmin), int(ymin), int(xmax), int(ymax))


if __name__ == "__main__":
    # Quick self-test
    print("=== Data Loader Self-Test ===")

    primary = "datasets/primary"
    supplement = "datasets/supplement"

    if os.path.isdir(primary):
        for split in ("train", "valid", "test"):
            d = load_yolo_dataset(primary, split, max_images=3)
            print(f"  YOLO [{split}]: loaded {len(d)} samples")
            for s in d:
                print(f"    {s['name']}  boxes={len(s['gt_boxes'])}  shape={s['image'].shape}")
    else:
        print(f"  Primary dataset not found at {primary}")

    if os.path.isdir(supplement):
        d = load_voc_dataset(supplement, max_images=3)
        print(f"  VOC: loaded {len(d)} samples")
        for s in d:
            print(f"    {s['name']}  boxes={len(s['gt_boxes'])}  texts={s['plate_texts']}")
    else:
        print(f"  Supplement dataset not found at {supplement}")
