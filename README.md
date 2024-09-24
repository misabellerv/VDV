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

The data for this project comes from the **Magellan mission**, which provided radar imaging of the Venusian surface. The processed version of the dataset can be accessed from Kaggle (https://www.kaggle.com/datasets/fmena14/volcanoesvenus). The original images from the radar, with a resolution of 75 meters per pixel, were cropped into smaller sections and converted to grayscale, with a resolution of 120 × 120 pixels. This processing facilitates a better interpretation of each image by reducing the amount of encoded information and increasing the volume of data available for machine learning models.

More details of data analysis can be found at (to be defined...).

---

## 🔬 Methodology
### 1. Data Augmentation

Given the imbalance in the dataset, with 6000 non-volcanic images and 1000 volcanic images in the training set, we applied intrinsic data augmentation to address this issue. The goal was to balance the dataset and improve the performance of machine learning models. We created 5 augmented copies of the training set containing volcanic images, applying the following transformations:

- **CLAHE (Contrast Limited Adaptive Histogram Equalization)**
- **Random Horizontal Flip**
- **Random Vertical Flip**
- **Random Diagonal Flip (D1)**
- **Random Anti-Diagonal Flip (D2)**

For each augmented dataset, the corresponding labels were generated (`1` for volcanic and `0` for non-volcanic). The augmented volcanic images were combined with the original dataset, resulting in a balanced dataset of 6000 volcanic images and 6000 non-volcanic images. The final dataset was randomly shuffled to avoid bias.

### 2. Preprocessing and Feature Extraction

We tested two main approaches for feature extraction after normalizing the images:

- **Gaussian Blur** + HOG (Histogram of Oriented Gradients)
- **Wavelet Denoising** + HOG

The images were first normalized and then passed through either Gaussian blur or wavelet denoising for noise reduction. The key features were extracted using the HOG algorithm, and we conducted two separate tests to evaluate which preprocessing method provided better model performance in terms of optimization and evaluation metrics.

### 3. Training and Validation

After preprocessing, the dataset was split into training and validation sets. We applied **k-fold cross-validation** to ensure robust model performance across different data splits. A **GridSearchCV** was performed to search for the best combination of hyperparameters for the model. 

Once the best model and hyperparameters were identified, the final step involved training the model on the full training set and then predicting on the test set using the **optimal model** found through **GridSearchCV**.

---

## 🛠️ Workflow

Below is a diagram representing the workflow of the project:

![load data](https://github.com/user-attachments/assets/9e6dbfba-9349-4c74-a8db-d6961ba82c6f)


## :heavy_check_mark: Running the Project

To run the project, follow these steps:

1. **Clone the repository**:

   ```bash
   git clone https://github.com/misabellerv/VDV.git
   cd VDV
   ```
2. **Install Dependencies**:

   - **With pip**:
     ```bash
     pip install -r requirements.txt
     ```

   - **With Conda**:
     ```bash
     conda env create -f environment.yml
     ```
3. **Activate Environment**:
   -  **With Conda**:
      ```bash
      conda activate <env_name>
      ```
4. ⚙️ **Set Configurations**:

All project configurations are embedded in a single JSON file located at `configs/configs.json`. Please, check (to be defined) to more detailed instructions.

5. **Training and Inference on Test Set**:
After setting configurations, you can train models, predict metric results and generate confusion matrixes for the test set using the following command:
```bash
python main.py
```

## :shipit: **Additional Information**

To more information on how the code works, you can check (to be defined...).
