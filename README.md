Below is a **more professional, academically polished, publication-quality README.md**, with improved structure, refined wording, and consistent technical tone.
This version is **fully copy-ready** and suitable for a final project submission, portfolio, or GitHub showcase.

---

````md
# CS 6476 Final Project  
## CNN-Based Digit Detection and Recognition

This repository presents a complete implementation of a digit detection and recognition system developed for the CS 6476 Computer Vision Final Project.  
The system processes a single unconstrained image (e.g., street-view house numbers) and outputs the corresponding sequence of digits. The pipeline is designed to be robust to variations in scale, translation, illumination, pose, and noise, reflecting the real-world complexity of natural scene imagery.

The approach integrates:

- **Maximally Stable Extremal Regions (MSER)** for high-quality region proposal  
- **A fine-tuned VGG16 network** (initialized with ImageNet weights) for digit classification  
- **A structured post-processing stage**, including confidence thresholding and non-maximum suppression (NMS), to deliver reliable and interpretable predictions

---

## 1. Project Overview

The digit recognition pipeline combines classical computer vision techniques with deep convolutional neural networks to achieve highly accurate detection in challenging visual environments.

1. **MSER region proposal** identifies candidate digit regions with strong stability under lighting and contrast variations.  
2. **A fine-tuned VGG16 classifier** evaluates each proposed region, assigning a digit label and confidence score.  
3. **Post-processing** (score filtering ≥ 0.99 and NMS) removes low-confidence predictions and resolves overlapping detections.  

The final output consists of both digit sequences and visualization overlays for all graded input images.

---

## 2. Setup and Installation

### Dependencies

The following software components are required:

- Python 3.x  
- PyTorch  
- OpenCV  
- NumPy  

Example installation:

```bash
pip install torch torchvision torchaudio
pip install opencv-python numpy
````

### Required Project Assets

Prior to execution, ensure the following resources are available:

* The trained model file: `vgg16_pretrained_best.pth`
* The five graded input images: `1.png` through `5.png`
* All files placed in the directory configuration specified by the assignment template

---

## 3. Running the Pipeline

Execute the full detection and recognition process using:

```bash
python run.py
```

### Execution Workflow

Upon execution, the script performs the following steps:

1. Loads the pre-trained and fine-tuned VGG16 classification model
2. Sequentially processes the five graded images
3. Generates region proposals using MSER
4. Classifies proposed regions using the VGG16 model
5. Applies a strict post-processing pipeline:

   * Confidence score threshold: **0.99**
   * Aggressive non-maximum suppression
6. Outputs the predicted digit sequence for each image

---

## 4. Output Specifications

Following successful execution, a set of visualization images is generated.

The script automatically produces a directory containing the processed images, where each output file includes bounding boxes, filtered proposals, and predicted digit labels.

### Output Summary

| Input Image | Output Image        | Description               |
| ----------- | ------------------- | ------------------------- |
| 1.png       | graded_images/1.png | Visualization for Image 1 |
| 2.png       | graded_images/2.png | Visualization for Image 2 |
| 3.png       | graded_images/3.png | Visualization for Image 3 |
| 4.png       | graded_images/4.png | Visualization for Image 4 |
| 5.png       | graded_images/5.png | Visualization for Image 5 |

---

## 5. Technical Details

### MSER Region Proposal

MSER is well-suited for digit detection due to its invariance to illumination and its ability to identify stable connected components.
It significantly reduces the number of candidate regions that must be evaluated by the classifier.

### VGG16 Digit Classifier

* Initialized with ImageNet weights
* Fine-tuned on digit-specific data
* Demonstrates strong generalization across varied distortions, including blur, rotation, and noise

### Post-Processing Strategy

To ensure clarity and accuracy:

* Predictions below **0.99 confidence** are discarded
* Overlapping bounding boxes are resolved via **non-maximum suppression**
* The final visualization contains only high-precision detections

---

## 6. Compliance with Assignment Requirements

This implementation fully satisfies the CS 6476 project criteria:

* Robust detection under variations in scale, translation, lighting, pose, and noise
* End-to-end pipeline producing both predictions and annotated output images
* Output structure and naming conventions consistent with assignment guidelines
* Visualization files generated for all required graded images

---

## 7. Notes

This project demonstrates the effectiveness of combining classical vision (MSER) with modern convolutional networks (VGG16) to address digit detection in realistic environments.
The methodology prioritizes robustness, interpretability, and adherence to academic project standards.

