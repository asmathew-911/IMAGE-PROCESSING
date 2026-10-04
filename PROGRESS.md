# Vehicle Number Plate Detection — Project Progress Log

**Project**: SEM-5 Image Processing Case Study (Q14)  
**Last Updated**: 2026-10-04

---

## Completed Tasks

### 1. Core Detection Pipeline (`main.py`)
- [x] **Preprocessing**: Image scaling and timestamp boundary removal
- [x] **Contrast Enhancement**: CLAHE (Contrast Limited Adaptive Histogram Equalization)
- [x] **Noise Filtering**: Bilateral filtering and Gaussian smoothing
- [x] **Edge Detection**: Canny edge extraction
- [x] **Binarization**: Otsu global and adaptive thresholding
- [x] **Morphological Processing**: Morphological closing and dilation across multi-scale kernels
- [x] **Contour Analysis**: Aspect ratio, area, and solidity filtering
- [x] **Perspective Correction**: Perspective transformation for tilted ROIs
- [x] **Detection Approaches**: Method A (Morphology baseline) and Method B (Character grouping sequence)
- [x] **OCR Integration**: EasyOCR, Tesseract, and Template Matching fallback chain
- [x] **Post-Processing**: Format normalization for Indian license plates
- [x] **Stage Visualizations**: Multi-panel processing stage figures and cropped ROI output

### 2. Preliminary Testing
- [x] Evaluated pipeline across 7 distinct lighting/angle test images
- [x] Achieved successful plate localization on 6/7 images (85.7%)
- [x] Generated step-by-step stage visualization grids

### 3. Dataset Selection & Setup
- [x] Evaluated 4 candidate datasets
- [x] Integrated **Dataset 4** (Kaggle License Plate Detection, 15K YOLO annotations) as primary benchmark
- [x] Integrated **Dataset 1** (Indian Vehicle Dataset, 1.5K Pascal VOC annotations) as supplement benchmark
- [x] Created `data_loader.py` for automated annotation parsing
- [x] Created `setup_datasets.py` for dataset structure verification

### 4. Benchmark Evaluation (`evaluate.py`)
- [x] Ground-truth bounding box loader for YOLO (`.txt`) and VOC (`.xml`) formats
- [x] Intersection over Union (IoU) calculation engine
- [x] Quantitative metrics computation (Detection Rate, Precision, Recall, Mean/Median IoU)
- [x] Processing time tracking per frame
- [x] Automatic IoU distribution plotting and CSV summary export

### 5. Project Documentation
- [x] Complete technical case study report (`REPORT.md`)
- [x] Repository overview (`README.md`)

---

## Benchmark Metrics Summary

| Metric | Manual Test Set (7 imgs) | Dataset 4 (YOLO, 50 samples) | Dataset 1 (VOC, 50 samples) |
|---|---|---|---|
| Detection Rate | 85.7% (6/7) | 92.0% (46/50) | 94.0% (47/50) |
| Accuracy (IoU >= 0.3) | 85.7% (6/7) | 80.0% (40/50) | 80.0% (40/50) |
| Precision | N/A | 87.0% | 85.1% |
| Recall | N/A | 80.0% | 80.0% |
| Mean IoU | N/A | 0.6101 | 0.6604 |
| Median IoU | N/A | 0.6696 | 0.7845 |
| Avg Processing Time | ~0.15s | 0.137s / frame | 0.049s / frame |

---

## Architectural & Design Choices

| Component | Selected Strategy | Rationale |
|---|---|---|
| Primary Dataset | Dataset 4 (YOLO 15K) | High volume, pre-split structure, standardized bounding boxes |
| Supplement Dataset | Dataset 1 (VOC 1.5K) | Contains ground-truth plate alphanumeric text for OCR testing |
| Detection Strategy | Classical Image Processing | Strictly adheres to course requirements without deep learning |
| Primary Detection Method | Character Blob Grouping (Method B) | Higher spatial robustness than global morphological closing alone |
| OCR Engine Chain | Template Matching -> EasyOCR | Pure classical default with high-accuracy fallback |
