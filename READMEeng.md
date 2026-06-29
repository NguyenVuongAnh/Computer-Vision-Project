# Diabetes Detection in Fundus Images using Lesion Segmentation and Explainable AI

## Overview

This project proposes an explainable framework for diabetic retinopathy (DR) detection from retinal fundus images. Instead of directly predicting the disease grade from the image, the system first identifies clinically meaningful retinal lesions and then uses these lesion-related features to classify the severity of diabetic retinopathy.

The framework consists of three main stages:

1. Lesion Segmentation
2. Lesion-Based Feature Extraction
3. DR Grade Classification

A Streamlit web application is developed to visualize segmentation results, lesion statistics, disease prediction, and model explanations.

---

## Motivation

Most deep learning models for diabetic retinopathy classification behave as black boxes. Although they may achieve high accuracy, clinicians often cannot understand why a prediction was made.

To improve interpretability, this project uses retinal lesions as intermediate concepts:

* Microaneurysms (MA)
* Hemorrhages (HE)
* Hard Exudates (EX)
* Soft Exudates (SE)

These lesions are important clinical indicators used by ophthalmologists for diabetic retinopathy diagnosis.

---

## System Architecture

```text
Fundus Image
      │
      ▼
Lesion Segmentation Network
      │
      ├── MA Mask
      ├── HE Mask
      ├── EX Mask
      └── SE Mask
      │
      ▼
Feature Extraction
      │
      ▼
DR Grade Classifier
      │
      ▼
Disease Severity Prediction
```

---

## Dataset

### IDRiD Dataset

The Indian Diabetic Retinopathy Image Dataset (IDRiD) provides:

#### Original Images

* Color retinal fundus images

#### Segmentation Ground Truth

* Microaneurysms (MA)
* Hemorrhages (HE)
* Hard Exudates (EX)
* Soft Exudates (SE)

#### Disease Grading Labels

* Grade 0: No DR
* Grade 1: Mild DR
* Grade 2: Moderate DR
* Grade 3: Severe DR
* Grade 4: Proliferative DR

---

## Methodology

### 1. Image Preprocessing

Preprocessing techniques include:

* Image resizing
* Normalization
* CLAHE (Contrast Limited Adaptive Histogram Equalization)
* Data augmentation

---

### 2. Lesion Segmentation

A segmentation model is trained to detect four lesion types:

| Lesion         | Abbreviation |
| -------------- | ------------ |
| Microaneurysms | MA           |
| Hemorrhages    | HE           |
| Hard Exudates  | EX           |
| Soft Exudates  | SE           |

Output:

```text
(H, W, 4)
```

where each channel represents one lesion mask.

Loss Function:

```text
Dice Loss + Binary Cross Entropy Loss
```

Evaluation Metrics:

* Dice Score
* IoU
* Precision
* Recall

---

### 3. Feature Extraction

The predicted lesion masks are converted into quantitative clinical features:

* Lesion area
* Lesion count
* Lesion density
* Lesion distribution

Example feature vector:

```text
[
 MA_area,
 MA_count,
 HE_area,
 HE_count,
 EX_area,
 EX_count,
 SE_area,
 SE_count
]
```

---

### 4. DR Classification

The extracted lesion features are used to predict diabetic retinopathy severity.

Output Classes:

| Grade | Description      |
| ----- | ---------------- |
| 0     | No DR            |
| 1     | Mild DR          |
| 2     | Moderate DR      |
| 3     | Severe DR        |
| 4     | Proliferative DR |

Classification Models:

* Logistic Regression
* Random Forest
* XGBoost
* Multi-Layer Perceptron (MLP)

Evaluation Metrics:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

---

## Explainability

Unlike conventional black-box classification models, this framework provides lesion-based explanations.

Example:

```text
Prediction: Severe DR

Detected Lesions:
- MA: 32 regions
- HE: 14 regions
- EX: 7 regions
- SE: 2 regions
```

This allows clinicians to understand which retinal abnormalities contributed to the final diagnosis.

---

## Streamlit Application

The web interface provides:

### Upload Image

Users can upload a retinal fundus image.

### Lesion Visualization

Display:

* Original image
* MA mask
* HE mask
* EX mask
* SE mask

### Disease Prediction

Display:

* Predicted DR grade
* Confidence score

### Explanation Dashboard

Display:

* Lesion statistics
* Lesion contribution chart
* Clinical interpretation

---

## Project Structure

```text
project/
│
├── app.py
│
├── models/
│   ├── segmentation_model.pt
│   └── classifier_model.pt
│
├── datasets/
│
├── notebooks/
│
├── utils/
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── feature_extraction.py
│   └── classification.py
│
├── assets/
│
├── requirements.txt
│
└── README.md
```

---

## Installation

Clone repository:

```bash
git clone <repository-url>
cd project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run Streamlit application:

```bash
streamlit run app.py
```

---

## Expected Outcomes

* Accurate lesion segmentation.
* Reliable DR severity prediction.
* Clinically interpretable explanations.
* User-friendly web application for demonstration and research purposes.

---

## Future Work

* Multi-task learning for joint segmentation and classification.
* Concept Bottleneck Networks.
* Attention-based explainability.
* Integration with larger retinal datasets.
* Real-time clinical deployment.

---

## Author

Student Project – Artificial Intelligence

Topic:
Diabetes Detection in Fundus Images Based on Retinal Lesion Segmentation and Explainable AI
