# 🌋 Volcano Detection on Venus using Machine Learning Models 🌑

Welcome to the **Volcano Detection on Venus** project! This repository contains the code and workflows for detecting volcanic structures on Venus using data collected from NASA's Magellan mission. This project aims to identify and classify volcanic features from radar images of Venus, contributing to the understanding of planetary geology.

---

## 📖 Overview

Venus is covered in volcanic landforms, many of which are yet to be studied in detail. With data from the Magellan mission, this project uses advanced image processing techniques and machine learning to automatically detect volcanic features from radar images of Venus’ surface.

### Objectives
- 🔍 **Automatic Detection**: Identify potential volcanoes from radar images.
- 🛰️ **Data Processing**: Preprocess large datasets from Magellan.
- 🧠 **Machine Learning**: Train models to classify volcanic structures.

---

## 📊 Data

The data for this project comes from the **Magellan mission**, which provided radar imaging of the Venusian surface. The processed version of the dataset can be accessed from [Kaggle](https://www.kaggle.com/). 

### Dataset Structure:
- **Radar Images**: `.png` or `.jpg` files representing Venus' surface.
- **Annotations**: `.txt` files describing volcanic structures.
  
Each image has a corresponding annotation file for detailed analysis.

---

## 🔬 Methodology

The workflow follows a multi-step approach combining classical image processing techniques with deep learning:

1. **Data Preprocessing**: Normalize image data and clean noise from radar images.
2. **Feature Extraction**: Use convolutional neural networks (CNNs) to extract visual features.
3. **Model Training**: Train models to detect volcanoes based on labeled examples.
4. **Evaluation**: Evaluate model performance using accuracy, precision, and recall metrics.

---

## 🛠️ Workflow
