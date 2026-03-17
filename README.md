# Meesho-Cloth Flagger


# 🚀 E-commerce Attribute Prediction using ML

## 📌 Overview

This project builds a **multi-label image classification system** to automatically predict product attributes (color, pattern, sleeve type) from images.

Goal: **reduce mismatch between product images and supplier data**, improving customer trust and lowering returns. 

---

## 🧠 Approach

* **Baseline:** Logistic Regression, Random Forest (HOG + color features)
* **CNN Pipeline:** ResNet50 / EfficientNet + XGBoost
* **Transformer Pipeline:** PVT-v2
* **Final Model:** **Ensemble (CNN + Transformer)**

---

## 📊 Results

* **Micro F1:** 90.0%
* **Macro F1:** 87.3%

Ensemble model performs best, especially on **complex & rare attributes**. 

---

## 📦 Dataset

* ~26K fashion images
* Attributes: Color (10), Pattern (8), Sleeve (6)
* Preprocessing: resize, normalization, augmentation

---

## ⚙️ Tech Stack

* Python
* PyTorch / TensorFlow
* XGBoost, LightGBM
* OpenCV, Scikit-learn

---

## 💻 Features

* Multi-label prediction
* Image-based attribute detection
* Scalable ML pipeline
