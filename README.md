# Automatic Number Plate Recognition (ANPR)

Classical Image Processing implementation for vehicle number plate localization and character recognition in Python using OpenCV.

This repository contains the codebase and case study evaluation for **Question 14 (Vehicle Number Plate Detection)** under the SEM-5 Image Processing course.

---

## Overview

The goal of this project is to detect, crop, and read vehicle license plates using purely classical image processing, with no machine learning or deep learning of any kind (no CNNs, SVMs, Tesseract, or EasyOCR).

### Implementation Pipeline

```
Input Image ──> Grayscale ──> CLAHE ──> Denoising (Bilateral)
               ──> Edge Detection (Canny) ──> Binarization (Otsu/Adaptive)
               ──> Morphological Closing ──> Character Blob Grouping ──> Plate Bounding Box
```

1. **Preprocessing**: Image scaling and timestamp boundary removal.
2. **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization).
3. **Noise Reduction**: Bilateral filtering to preserve character edges.
4. **Edge Detection**: Canny edge extraction.
5. **Binarization**: Otsu and adaptive thresholding.
6. **Morphology**: Morphological closing and dilation to merge character edge responses into plate regions.
7. **Character Blob Grouping**: Geometry and aspect-ratio based contour chaining (Method B).
8. **Perspective Correction**: Perspective transform to straighten tilted plate ROIs.
9. **OCR**: Template matching and EasyOCR fallback for text recognition.

---

## Project Structure

```
.
├── main.py              # Main pipeline script for image detection and stage visualization
├── data_loader.py       # Annotation parsers for YOLO (.txt) and Pascal VOC (.xml)
├── evaluate.py          # Batch benchmark script calculating IoU, precision, and recall
├── setup_datasets.py    # Utility script to check local dataset structure
├── images/              # Input sample images
├── output/              # Stage-by-stage visualization figures and cropped plates
├── datasets/            # Primary (Dataset 4) and Supplement (Dataset 1) benchmark datasets
└── results/             # Evaluation reports, IoU histograms, and CSV summaries
```

---

## Quick Start

### 1. Requirements
Ensure Python 3.8+ is installed with the following packages:
```bash
pip install opencv-python-headless matplotlib numpy pillow flask
```

### 2. Run Single Image Detection
Place input images in `images/` and run:
```bash
python main.py
```
Output figures showing each processing stage will be saved in `output/`.

### 3. Check Dataset Setup
Verify that the Kaggle datasets are extracted inside `datasets/primary/` and `datasets/supplement/`:
```bash
python setup_datasets.py
```

### 4. Run Benchmark Evaluation
Run quantitative evaluation against ground-truth bounding box annotations:
```bash
python evaluate.py --max 50
```

---

## Experimental Benchmark Results

Quantitative evaluation conducted on test sets from **Kaggle License Plate Detection (Dataset 4)** and **Indian Vehicle Dataset (Dataset 1)**:

| Metric | Dataset 4 (YOLO, 15K) | Dataset 1 (VOC, 1.5K) | Manual Test Set |
| :--- | :--- | :--- | :--- |
| **Detection Rate** | 92.0% | 94.0% | 85.7% (6/7) |
| **Accuracy (IoU >= 0.3)** | 80.0% | 80.0% | 85.7% |
| **Precision** | 87.0% | 85.1% | N/A |
| **Recall** | 80.0% | 80.0% | N/A |
| **Mean IoU** | 0.6101 | 0.6604 | N/A |
| **Median IoU** | 0.6696 | 0.7845 | N/A |
| **Avg Speed per Frame** | 0.137s (~7.3 FPS) | 0.049s (~20.4 FPS) | ~0.15s |

---

## Documentation
- `REPORT.md`: Full case study methodology and quantitative analysis.
- `PROGRESS.md`: Execution log and phase completion status.
