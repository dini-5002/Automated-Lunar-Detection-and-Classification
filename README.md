# Automated Lunar Crater Detection and Classification

**AA 310 — Satellite Imaging Project**  
**Dhriti Jha · Nandini Kumari**  
**April 2026**

A computer-vision pipeline for detecting and classifying lunar craters from **Chandrayaan-2 OHRC imagery** using YOLO. The project also explores metadata-driven crater depth estimation from shadow geometry.

---

## Overview

Lunar crater detection is challenging because crater appearance varies with illumination, erosion, resolution, and surface conditions. This project develops an automated detection pipeline while reducing the amount of manual annotation required through an iterative labeling workflow.

The final system:

- Detects lunar craters in Chandrayaan-2 OHRC imagery
- Classifies detections into **small, medium, and large** craters
- Uses overlapping image tiling to handle large OHRC images
- Builds and refines a labeled dataset through iterative annotation
- Trains a **YOLO11s** object-detection model
- Evaluates performance on an unseen test set
- Uses OHRC metadata and shadow geometry to estimate crater depth
- Produces structured JSON output containing crater detections and derived measurements

---

## Pipeline

```text
Chandrayaan-2 OHRC Imagery
            │
            ▼
   Image Filtering & Tiling
            │
            ▼
      Manual Annotation
            │
            ▼
    Initial Model Training
            │
            ▼
      Pseudo-labeling
            │
            ▼
     Label Refinement
            │
            ▼
      ~850 Image Dataset
            │
            ▼
        YOLO11s Training
            │
            ▼
      Crater Detection
       ┌────┴────┐
       ▼         ▼
   Classification   Metadata
                    │
                    ▼
             Shadow-based
             Depth Estimation
                    │
                    ▼
          JSON + Visualization
```

---

## Dataset

The imagery was obtained from the **Chandrayaan-2 PRADAN portal** and processed into a dataset of approximately **850 images**.

### Dataset preparation

- Large OHRC images were divided into **512 × 512 tiles**
- Tiles were generated with **25% overlap**
- The preprocessing pipeline generated **898 tiles**, followed by filtering of 48 tiles
- **850 images** were retained for the final labeled dataset
- Craters were annotated in three classes:
  - `crater_small`
  - `crater_medium`
  - `crater_large`

The repository does **not** contain the full dataset because of its size.

> **Dataset:** [Download the full dataset](https://drive.google.com/file/d/1q1BQzvSmGIQ7hM9HMENl6IKvIu8igt3U/view)

---

## Iterative Annotation

Because manually labeling a large lunar-image dataset is time-consuming, the project used an iterative annotation workflow.

1. An initial subset of images was manually labeled.
2. A YOLO model was trained on the available annotations.
3. The trained model was used to generate predictions/pseudo-labels on additional imagery.
4. Predictions were reviewed and labels were refined.
5. Additional labeled images were incorporated into the dataset.
6. The process was repeated to obtain the final dataset of approximately 850 images.

The intermediate notebooks contain the dataset-merging, annotation, splitting, and pre-annotation model workflows.

---

## Preprocessing

The preprocessing notebook performs the following operations:

### Tiling

Original OHRC images are divided into overlapping tiles:

- Tile size: **512 × 512**
- Overlap: **25%**
- Stride: `512 × (1 - 0.25) = 384 pixels`

### Image processing

The pipeline also includes:

- Resizing tiles to **640 × 640**
- Contrast/brightness augmentation
- OpenCV-based image processing

The final training pipeline uses 640 × 640 input resolution.

---

## Model

The final detector is **YOLO11s** trained using the Ultralytics framework.

### Training configuration

| Parameter | Value |
|---|---:|
| Model | YOLO11s |
| Epochs | 150 |
| Image size | 640 × 640 |
| Batch size | 16 |
| Initial learning rate | 0.001 |
| Patience | 30 |
| Mosaic augmentation | Enabled |
| Copy-paste augmentation | 0.3 |
| Horizontal flip | 0.5 |
| Vertical flip | 0.5 |
| Rotation | ±45° |

The training notebook saves the best checkpoint as:

```text
best_final.pt
```

---

## Dataset Split

The final dataset contains **850 images**:

| Split | Images | Percentage |
|---|---:|---:|
| Training | 594 | 70% |
| Validation | 128 | 15% |
| Test | 128 | 15% |
| **Total** | **850** | **100%** |

The split was performed with class-aware grouping to preserve representation of crater categories across the splits.

The test set contains **128 unseen images**.

---

## Results

### Test Performance

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Small | 0.559 | 0.476 | 0.541 | 0.282 |
| Medium | 0.546 | 0.544 | 0.578 | 0.387 |
| Large | 0.708 | 0.714 | 0.786 | 0.552 |
| **Overall** | **0.604** | **0.578** | **0.635** | **0.407** |

The final model achieved:

- **mAP@50: 0.635**
- **mAP@50–95: 0.407**
- **Precision: 0.604**
- **Recall: 0.578**

Large craters achieved the strongest detection performance, while **small crater detection remained the primary challenge**.

### Validation Performance

| Class | Precision | Recall | mAP@50 | mAP@50–95 |
|---|---:|---:|---:|---:|
| Small | 0.581 | 0.586 | 0.625 | 0.338 |
| Medium | 0.542 | 0.605 | 0.630 | 0.440 |
| Large | 0.695 | 0.767 | 0.821 | 0.527 |
| **Overall** | **0.606** | **0.653** | **0.692** | **0.435** |

---

## Inference

The inference notebook loads the trained YOLO11s checkpoint and runs detection on the 128-image test set.

Example inference configuration:

```python
CONF = 0.25
IOU = 0.45
IMGSZ = 640
```

The pipeline outputs bounding boxes, confidence scores, and crater classes.

Example:

```text
crater_small   0.84
crater_medium  0.71
crater_large   0.83
```

The inference notebook also generates visualization grids containing detected craters.

---

## Depth Estimation

In addition to object detection, the project includes an experimental physics-based depth-estimation pipeline using OHRC metadata.

The approach uses:

- Pixel resolution
- Sun elevation
- Sun azimuth
- Crater shadow geometry

### Processing steps

1. Parse PDS XML metadata.
2. Extract pixel resolution, Sun elevation, and Sun azimuth.
3. Detect craters using YOLO.
4. Extract the detected crater region.
5. Rotate the crater region according to Sun azimuth so that the shadow direction is aligned.
6. Detect the darker shadow region using image intensity thresholding.
7. Estimate shadow length.
8. Convert pixel distance to metres using pixel resolution.
9. Estimate crater depth using:

```text
d = S × tan(θ)
```

where:

- `d` = estimated crater depth
- `S` = shadow length in metres
- `θ` = Sun elevation angle

Crater diameter is estimated from the bounding-box dimensions and pixel resolution.

The pipeline also computes:

```text
d / D
```

where `D` is estimated crater diameter.

---

## End-to-End Output

The full pipeline combines YOLO detections with metadata-driven measurements and writes structured JSON output.

Example:

```json
{
  "image": "example.png",
  "craters": [
    {
      "crater_id": 0,
      "bbox": [183, 203, 271, 287],
      "label": "crater_large",
      "class_id": 2,
      "confidence": 0.8397,
      "depth_m": 1.9388,
      "depth_class": "deep",
      "diameter_m": 20.24,
      "d_by_D": 0.0958
    }
  ]
}
```

The inference pipeline also creates a visualization comparing the original image with the detected crater bounding boxes.

---

## Repository Contents

```text
Automated-Lunar-Detection-and-Classification/
│
├── README.md
│
├── preprocessing.ipynb
│   └── OHRC image tiling and preprocessing
│
├── pre-annotation model training and inference.ipynb
│   └── Dataset merging, splitting and intermediate model inference
│
├── final-training-chandrayan-crater-detection.ipynb
│   └── Final dataset preparation, YOLO11s training and evaluation
│
├── inference-chandrayan-tiled-test.ipynb
│   └── Test-set inference and visualization
│
├── full_pipeline.ipynb
│   └── YOLO detection + metadata parsing + depth estimation + JSON output
│
└── ...
```

---

## Notebooks

### `preprocessing.ipynb`

Contains the initial OHRC image preprocessing workflow:

- 512 × 512 overlapping tiling
- Image inspection
- Resizing to 640 × 640
- Contrast/brightness processing
- Intermediate dataset preparation

### `pre-annotation model training and inference.ipynb`

Contains the intermediate annotation/model workflow:

- Merging labeled datasets
- Creating train/validation/test splits
- YOLO dataset configuration
- Loading an intermediate YOLO11 model
- Generating predictions for additional imagery

### `final-training-chandrayan-crater-detection.ipynb`

Contains the final training workflow:

- Merging the final labeled datasets
- Class-aware train/validation/test split
- YOLO11s training
- Final checkpoint creation
- Test-set evaluation
- mAP and per-class metrics

### `inference-chandrayan-tiled-test.ipynb`

Contains final-model inference on the held-out test images and prediction visualization.

### `full_pipeline.ipynb`

Contains the experimental end-to-end inference pipeline:

```text
OHRC image + PDS metadata
        ↓
Random 640 × 640 crop
        ↓
YOLO11s detection
        ↓
Crater classification
        ↓
Shadow-based depth estimation
        ↓
Diameter estimation
        ↓
JSON output + visualization
```

---

## Limitations

The current system has several limitations:

- Small craters are substantially harder to detect than large craters.
- The final model achieves **0.635 mAP@50** on the held-out test set.
- Processing the original high-resolution OHRC images directly is computationally expensive, so inference is performed on tiles/crops.
- The depth-estimation component is an experimental physics-based approximation based on detected shadow regions and metadata.
- The full dataset and trained model weights are not stored directly in this repository because of their size.

---

## Future Work

Potential improvements include:

- Improving small-crater detection
- Increasing the amount and diversity of manually verified training data
- More robust pseudo-label filtering
- Segmentation-based crater boundary refinement
- More robust shadow detection for depth estimation
- Full-image tiled inference with coordinate stitching
- Improved depth-estimation validation against reference measurements

---

## Authors

**Nandini Kumari**  
Model training · Manual labeling · Report writing

**Dhriti Jha**  
Research · Model training · Manual labeling · Depth estimation · Presentation

---

## Acknowledgements

- Chandrayaan-2 OHRC imagery obtained through the **PRADAN portal**
- Ultralytics YOLO framework
- AA 310 — Satellite Imaging Project
