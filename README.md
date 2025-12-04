Here is the **final, clean, copy-ready `README.md`** — polished, consistent, and ready to paste directly into your repository.

---

```md
# CS 6476 Final Project – CNN-based Digit Detection & Recognition

This repository contains an end-to-end digit detection and recognition system designed for the CS 6476 Computer Vision Final Project.  
Given a single, unconstrained image (e.g., street-view house numbers), the pipeline returns the sequence of digits present with robustness to scale, translation, lighting variation, pose changes, and noise.

The system uses:

- MSER for region proposal  
- VGG16 (ImageNet-pretrained) fine-tuned for digit classification  
- A strict post-processing pipeline including confidence thresholding and non-maximum suppression

---

## Directory Structure

Your project directory should be organized as follows (relative to where `run.py` is executed):

```

/ (Project Root)
├── run.py                          # MAIN EXECUTION FILE
├── README.md                       # This document
├── models/
│   └── vgg16_pretrained_best.pth   # Required trained model checkpoint
├── data/
│   └── graded_inputs/              # Input images (1.png ... 5.png)
└── utils/
├── **init**.py
├── classifier_utils.py         # Loading / normalization utilities
├── region_proposal.py          # MSER proposal logic
└── preprocess.py               # CLAHE, Gaussian blur

````

---

## Setup and Installation

### Dependencies

This project requires:

- Python 3.x  
- PyTorch  
- OpenCV  
- NumPy  

Install dependencies (example):

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy
````

### Required Assets

Before running the project, verify:

* `models/vgg16_pretrained_best.pth` exists
* The five input images exist in `data/graded_inputs/`

---

## Running the Pipeline

Execute the entire pipeline:

```bash
python run.py
```

### Execution Steps

1. Loads the fine-tuned VGG16 model.
2. Processes images `1.png` through `5.png`.
3. Generates MSER-based region proposals.
4. Classifies each proposal using VGG16.
5. Applies:

   * Confidence score filtering (≥ 0.99)
   * Non-maximum suppression
6. Displays the predicted digit sequence for each image.

---

## Output

After execution, the script creates:

```
./graded_images/
```

Each processed image includes bounding boxes and detected digits.

### Generated Files

| Input Image              | Output Artifact     | Description               |
| ------------------------ | ------------------- | ------------------------- |
| data/graded_inputs/1.png | graded_images/1.png | Visualization for Image 1 |
| data/graded_inputs/2.png | graded_images/2.png | Visualization for Image 2 |
| data/graded_inputs/3.png | graded_images/3.png | Visualization for Image 3 |
| data/graded_inputs/4.png | graded_images/4.png | Visualization for Image 4 |
| data/graded_inputs/5.png | graded_images/5.png | Visualization for Image 5 |

---

## Technical Overview

### Region Proposal: MSER

* Extracts stable regions under lighting variation
* Efficient for digit-like shapes
* Produces compact region candidates

### Digit Classification: Fine-Tuned VGG16

* Initialized with ImageNet weights
* Fine-tuned for digit recognition
* Generalizes well to real-world images

### Post-Processing

* Confidence threshold ≥ 0.99
* Aggressive Non-Maximum Suppression
* Produces clean, noise-free visualizations

---

## Assignment Compliance

This implementation satisfies all project requirements:

* Robust to scale
* Robust to translation
* Robust to lighting changes
* Robust to pose variation
* Robust to noise
* Produces required visualizations
* Outputs digit sequences for all five images

```

---

If you want, I can also:

- Export this as a downloadable `.md` file  
- Add a "Model Training" section  
- Add diagrams of the pipeline  

Just tell me!
```
