"""
evaluate.py  –  Batch evaluation of the number plate detection pipeline.

Usage:
    python evaluate.py                         # run on all available data
    python evaluate.py --dataset primary       # only Dataset 4 (YOLO)
    python evaluate.py --dataset supplement    # only Dataset 1 (XML)
    python evaluate.py --max 50               # limit to 50 images
    python evaluate.py --save-failures         # save images where detection failed
"""

import os
import sys
import time
import argparse
import json
import csv
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# import the detection pipeline from main.py
from main import prepare, detect_plate
from data_loader import (
    load_yolo_dataset, load_voc_dataset,
    compute_iou, detection_box_from_result,
)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


# ----------------------------------------------------------- evaluation core
def evaluate_single(image, gt_boxes):
    """
    Run detection on one image, compare with ground-truth boxes.
    Returns dict with metrics for this image.
    """
    t0 = time.time()
    img = prepare(image.copy())
    res = detect_plate(img)
    elapsed = time.time() - t0

    detected = res["box"] is not None
    det_box = detection_box_from_result(res) if detected else None

    best_iou = 0.0
    matched_gt = None
    if det_box and gt_boxes:
        for gt in gt_boxes:
            iou = compute_iou(det_box, gt)
            if iou > best_iou:
                best_iou = iou
                matched_gt = gt

    # A detection is "correct" if IoU >= 0.3 with any ground-truth box
    # (0.3 is lenient because classical methods aren't pixel-perfect)
    correct = best_iou >= 0.3

    return {
        "detected": detected,
        "correct": correct,
        "iou": best_iou,
        "det_box": det_box,
        "matched_gt": matched_gt,
        "time": elapsed,
        "result": res,
    }


def draw_comparison(image, det_box, gt_boxes, iou, name, save_path):
    """Draw detection (green) and ground truth (blue) boxes on image."""
    vis = image.copy()
    for gt in gt_boxes:
        cv2.rectangle(vis, (gt[0], gt[1]), (gt[2], gt[3]), (255, 0, 0), 2)  # blue = GT
    if det_box:
        cv2.rectangle(vis, (det_box[0], det_box[1]), (det_box[2], det_box[3]), (0, 255, 0), 3)  # green = detection
    # Put IoU text
    label = f"IoU={iou:.2f}" if det_box else "NO DETECTION"
    cv2.putText(vis, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.imwrite(save_path, vis)


# ------------------------------------------------------------- batch runner
def run_evaluation(dataset_name, data, save_failures=False, save_samples=10):
    """
    Run evaluation on a list of data samples.
    Returns summary dict with aggregate metrics.
    """
    print(f"\n{'='*60}")
    print(f"  Evaluating: {dataset_name}  ({len(data)} images)")
    print(f"{'='*60}")
    print(f"{'#':>5}  {'Name':30}  {'Det':>4}  {'IoU':>6}  {'Time':>7}")
    print("-" * 60)

    total = len(data)
    detected_count = 0
    correct_count = 0
    ious = []
    times = []
    failures = []
    all_results = []

    # create output dirs
    ds_dir = os.path.join(RESULTS_DIR, dataset_name)
    os.makedirs(os.path.join(ds_dir, "samples"), exist_ok=True)
    if save_failures:
        os.makedirs(os.path.join(ds_dir, "failures"), exist_ok=True)

    saved_count = 0
    for i, sample in enumerate(data):
        name = sample["name"]
        img = sample["image"]
        gt_boxes = sample["gt_boxes"]

        metrics = evaluate_single(img, gt_boxes)

        detected_count += int(metrics["detected"])
        correct_count += int(metrics["correct"])
        if metrics["detected"]:
            ious.append(metrics["iou"])
        times.append(metrics["time"])

        status = "OK" if metrics["correct"] else ("FAIL" if metrics["detected"] else "MISS")
        disp_name = (name[:27] + "...") if len(name) > 30 else name
        print(f"{i+1:>5}  {disp_name:30}  {status:>5}  {metrics['iou']:>6.3f}  {metrics['time']:>6.2f}s")

        # Save sample comparisons (first N and random failures)
        record = {
            "name": name,
            "detected": metrics["detected"],
            "correct": metrics["correct"],
            "iou": round(metrics["iou"], 4),
            "time": round(metrics["time"], 3),
        }
        all_results.append(record)

        if saved_count < save_samples:
            draw_comparison(
                img, metrics["det_box"], gt_boxes, metrics["iou"], name,
                os.path.join(ds_dir, "samples", f"{name}_compare.jpg"),
            )
            saved_count += 1

        if save_failures and not metrics["correct"]:
            draw_comparison(
                img, metrics["det_box"], gt_boxes, metrics["iou"], name,
                os.path.join(ds_dir, "failures", f"{name}_fail.jpg"),
            )
            failures.append(name)

    # --- Aggregate metrics ---
    det_rate = detected_count / total if total > 0 else 0
    correct_rate = correct_count / total if total > 0 else 0
    avg_iou = np.mean(ious) if ious else 0
    median_iou = np.median(ious) if ious else 0
    avg_time = np.mean(times) if times else 0

    # Precision = correct / detected (of those detected, how many are right)
    precision = correct_count / detected_count if detected_count > 0 else 0
    # Recall = correct / total GT (how many GTs did we find correctly)
    recall = correct_count / total if total > 0 else 0

    summary = {
        "dataset": dataset_name,
        "total_images": total,
        "detected": detected_count,
        "correct (IoU>=0.3)": correct_count,
        "detection_rate": round(det_rate * 100, 1),
        "accuracy (correct/total)": round(correct_rate * 100, 1),
        "precision": round(precision * 100, 1),
        "recall": round(recall * 100, 1),
        "avg_iou": round(avg_iou, 4),
        "median_iou": round(median_iou, 4),
        "avg_time_sec": round(avg_time, 3),
        "total_time_sec": round(sum(times), 1),
    }

    print("\n" + "=" * 60)
    print(f"  RESULTS — {dataset_name}")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k:30}  {v}")
    print("=" * 60)

    # --- Save results ---
    # JSON summary
    with open(os.path.join(ds_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Per-image CSV
    csv_path = os.path.join(ds_dir, "per_image.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "detected", "correct", "iou", "time"])
        writer.writeheader()
        writer.writerows(all_results)

    # --- IoU distribution histogram ---
    if ious:
        plt.figure(figsize=(8, 5))
        plt.hist(ious, bins=20, range=(0, 1), color="#4CAF50", edgecolor="white", alpha=0.85)
        plt.axvline(avg_iou, color="red", linestyle="--", label=f"Mean IoU = {avg_iou:.3f}")
        plt.axvline(0.3, color="orange", linestyle=":", label="Threshold = 0.3")
        plt.xlabel("IoU")
        plt.ylabel("Count")
        plt.title(f"IoU Distribution — {dataset_name}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(ds_dir, "iou_distribution.png"), dpi=120)
        plt.close()

    # --- Detection rate bar chart ---
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Detected", "Correct\n(IoU≥0.3)", "Missed"]
    values = [detected_count, correct_count, total - detected_count]
    colors = ["#2196F3", "#4CAF50", "#F44336"]
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.6)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontweight="bold")
    ax.set_ylabel("Images")
    ax.set_title(f"Detection Summary — {dataset_name}  (n={total})")
    plt.tight_layout()
    plt.savefig(os.path.join(ds_dir, "detection_summary.png"), dpi=120)
    plt.close()

    print(f"\n  Results saved to {ds_dir}/")
    return summary


# -------------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser(description="Evaluate number plate detection pipeline")
    parser.add_argument("--dataset", choices=["primary", "supplement", "all"], default="all",
                        help="Which dataset to evaluate on")
    parser.add_argument("--max", type=int, default=None,
                        help="Max images to process (for quick testing)")
    parser.add_argument("--save-failures", action="store_true",
                        help="Save images where detection failed")
    parser.add_argument("--split", default="test",
                        help="Which split to use for YOLO dataset (default: test)")
    args = parser.parse_args()

    summaries = []

    # --- Dataset 4 (Primary — YOLO) ---
    if args.dataset in ("primary", "all"):
        primary_dir = "datasets/primary"
        if os.path.isdir(primary_dir):
            data = load_yolo_dataset(primary_dir, split=args.split, max_images=args.max)
            if data:
                s = run_evaluation("primary_dataset4", data,
                                   save_failures=args.save_failures)
                summaries.append(s)
            else:
                print(f"\n[!] No images found in {primary_dir}/{args.split}/")
                print("    Expected structure: datasets/primary/<split>/images/ + labels/")
        else:
            print(f"\n[!] Primary dataset not found at '{primary_dir}'")
            print("    Download from: https://www.kaggle.com/datasets/sunrajbishnolia/license-plate-detection")
            print("    Extract into: datasets/primary/")

    # --- Dataset 1 (Supplement — VOC XML) ---
    if args.dataset in ("supplement", "all"):
        supplement_dir = "datasets/supplement"
        if os.path.isdir(supplement_dir):
            data = load_voc_dataset(supplement_dir, max_images=args.max)
            if data:
                s = run_evaluation("supplement_dataset1", data,
                                   save_failures=args.save_failures)
                summaries.append(s)
            else:
                print(f"\n[!] No annotated images found in {supplement_dir}/")
                print("    Check that XML + image pairs exist in subdirectories.")
        else:
            print(f"\n[!] Supplement dataset not found at '{supplement_dir}'")
            print("    Download from: https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset")
            print("    Extract into: datasets/supplement/")

    # --- Combined summary ---
    if len(summaries) > 1:
        print("\n" + "=" * 60)
        print("  COMBINED RESULTS")
        print("=" * 60)
        for s in summaries:
            print(f"\n  [{s['dataset']}]")
            for k, v in s.items():
                if k != "dataset":
                    print(f"    {k:30}  {v}")

    # Save combined summary
    if summaries:
        with open(os.path.join(RESULTS_DIR, "combined_summary.json"), "w") as f:
            json.dump(summaries, f, indent=2)
        print(f"\n  Combined results saved to {RESULTS_DIR}/combined_summary.json")

    if not summaries:
        print("\n" + "=" * 60)
        print("  NO DATASETS FOUND — Please download first!")
        print("=" * 60)
        print("""
  Steps to set up:

  1. DATASET 4 (Primary — 15K images):
     → Go to: https://www.kaggle.com/datasets/sunrajbishnolia/license-plate-detection
     → Download & extract into: datasets/primary/
     → Should have: datasets/primary/train/images/ + labels/
                    datasets/primary/valid/images/ + labels/
                    datasets/primary/test/images/  + labels/

  2. DATASET 1 (Supplement — 1.5K images):
     → Go to: https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset
     → Download & extract into: datasets/supplement/
     → Should have XML + image pairs in subdirectories

  Then re-run:  python evaluate.py
        """)


if __name__ == "__main__":
    main()
