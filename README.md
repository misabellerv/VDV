# 🌋 Volcano Detection on Venus using Machine Learning Models 🌑

Welcome to the **Volcano Detection on Venus (VDV)** project! This repository contains the code and workflows for classifying the presence of volcanic structures on Venus using data collected from NASA's Magellan mission. This project aims to identify and classify volcanic features from radar images of Venus, contributing to the understanding of planetary geology.

The work uses classic machine learning algorithms to address a classification problem regarding the presence of volcanoes on the surface of Venus, using images generated by the Magellan spacecraft. Preprocessing techniques are discussed, such as the use of Gaussian filters and Wavelets. Additionally, intrinsic augmentation of the images is performed to balance the classes, avoiding imbalance in classifier performance. The models used were SVM, Decision Trees, XGBoost, and KNN. A final result of 95% recall on the test set was achieved, demonstrating the effectiveness of the techniques used.

---

## 📖 Overview

The Magellan mission, launched by NASA on May 4, 1989, was one of the most significant planetary exploration initiatives of its time. Its primary objective was to map the surface of Venus using Synthetic Aperture Radar (SAR) due to the impossibility of direct visual observation caused by the dense cloud layer enveloping the planet. Magellan mapped approximately 98% of Venus’s surface with high resolution, revealing complex geological formations such as mountains, volcanic plains, impact craters, and extensive tectonic deformation regions. This data was crucial for advancing the understanding of
Venus’s geology and evolution, as well as providing valuable insights into the formation and dynamics of rocky planets.

With the advent of artificial intelligence (AI) and computer vision, space exploration
is undergoing a technological revolution. AI is capable of processing large volumes of data, such as those generated by the Magellan mission, significantly faster and more efficiently than the techniques available at the time. This enables detailed, real-time analysis of the geological and atmospheric characteristics of planets, allowing the iden- tification of patterns that might go unnoticed by traditional methods. In addition to data analysis, AI plays a critical role in the autonomy of space missions. In extreme environments, such as the surface of Venus or Mars, where conditions are challenging and communication with Earth can be limited or delayed, the ability of spacecraft to make autonomous decisions is essential. AI can, for example, identify safe landing areas, avoiding dangerous terrain, and adapt the mission in response to unforeseen changes in environmental conditions. Complementarily, computer vision allows probes and rovers to navigate the terrain accurately, collecting essential data and avoiding obstacles. The current work focuses on the use of classical machine learning algorithms to perform the supervised classification task. The presence or absence of a volcano is classified, thus optimizing the probe’s work for faster inference applications and helping to avoid potential damage in the event of landings in areas considered hazardous, which could harm probes or rovers that may enter the planet’s surface.

## 📊 Data

The data for this project comes from the **Magellan mission**, which provided radar imaging of the Venusian surface. The processed version of the dataset can be accessed from [Kaggle]([https://www.kaggle.com/](https://www.kaggle.
com/datasets/fmena14/volcanoesvenus)). The original images from the radar, with a resolution of 75
meters per pixel, were cropped into smaller sections and converted to grayscale, with
a resolution of 120 × 120 pixels. This processing facilitates a better interpretation of each
image by reducing the amount of encoded information and increasing the volume of
data available for machine learning models.

---

## 🔬 Methodology

The workflow follows a multi-step approach combining classical image processing techniques with deep learning:

1. **Data Preprocessing**: Normalize image data and clean noise from radar images.
2. **Feature Extraction**: Use convolutional neural networks (CNNs) to extract visual features.
3. **Model Training**: Train models to detect volcanoes based on labeled examples.
4. **Evaluation**: Evaluate model performance using accuracy, precision, and recall metrics.

---

## 🛠️ Workflow
