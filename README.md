# Depth Pro and YOLO Integration

---

## **Overview**
The repository integrates Depth Pro with YOLO to detect objects in an image and estimate their depth values. Depth Pro provides accurate metric depth predictions, while YOLO detects objects and their bounding boxes. Together, they enable applications such as 3D object localization and scene understanding.

---
<p align="center">
  <img src="https://github.com/user-attachments/assets/8cd8b9d5-8c6e-42f0-9802-21c215a1a4dd" alt="Screenshot 1" width="45%">
  <img src="https://github.com/user-attachments/assets/f445b942-c8f1-408d-b218-7321a0765966" alt="Screenshot 2" width="45%">
</p>

---


## How It Works

1. **YOLO Object Detection**:
   - YOLO detects objects in an image and returns bounding box coordinates for each detected object.

2. **Depth Pro Depth Estimation**:
   - Depth Pro processes the same image to produce a depth map.
   - For each detected object, the depth value at the center of its bounding box is extracted from the depth map.

3. **Integration**:
   - Bounding boxes and depth values are combined to display detection results with depth annotations on the input image.

## Getting Started

### Prerequisites

Ensure you have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) installed.

### Cloning the Repository

Clone the official Depth Pro repository from GitHub:

```bash
git clone https://github.com/apple/ml-depth-pro.git
cd ml-depth-pro
```

### Setting Up the Environment

Create and activate a virtual environment using Conda:

```bash
conda create -n depth-pro -y python=3.9
conda activate depth-pro
```

Install the Depth Pro package in editable mode:

```bash
pip install -e .
```

### Downloading Pretrained Checkpoints

Run the following script to download pretrained checkpoints:

```bash
source get_pretrained_models.sh
```

This will download files to the `checkpoints` directory.

## Running the Program

### Running the Integration Script

To run the YOLO and Depth Pro integration script, execute the `check_dist.py` file from the root directory of the Depth Pro repository:

```bash
python check_dist.py
```

This will:
1. Load an input image.
2. Detect objects using YOLO.
3. Estimate depth using Depth Pro.
4. Display the results with bounding boxes and depth annotations.

## **Citation**
Research paper that acted as a beaken light for this project:

```bibtex
@article{Bochkovskii2024:arxiv,

  title      = {Depth Pro: Sharp Monocular Metric Depth in Less Than a Second},
  year       = {2024},
  url        = {https://arxiv.org/abs/2410.02073},
}
```

---

## **Acknowledgments**
This project builds upon **Depth Pro** and **Ultralytics YOLOv8**. Thanks to the developers for open-sourcing their work!
