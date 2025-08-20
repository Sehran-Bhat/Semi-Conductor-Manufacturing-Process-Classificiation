# Semi-Conductor-Manufacturing-Process-Classificiation

# 📊 Semiconductor Manufacturing Process Classification

## 🔹 Project Overview

This project focuses on building a **machine learning classifier** to predict the **Pass/Fail yield** in a semiconductor manufacturing process. Modern semiconductor production involves monitoring hundreds of sensor signals, many of which may be irrelevant or noisy. The goal of this project is to identify the most significant features impacting yield quality and develop models that can accurately classify production outcomes.

---

## 🔹 Dataset

* **File:** `sensor-data.csv`
* **Shape:** 1567 rows × 592 columns
* **Features:** 591 sensor signal measurements
* **Target:**

  * `-1` → Pass
  * `1` → Fail

---

## 🔹 Project Objectives

1. **Data Exploration & Cleaning**

   * Handle missing values and irrelevant attributes
   * Apply feature selection to identify key signals

2. **Exploratory Data Analysis (EDA)**

   * Statistical analysis
   * Univariate, bivariate, and multivariate visualizations

3. **Preprocessing**

   * Train-test split
   * Standardization/Normalization
   * Address class imbalance (SMOTE)

4. **Model Development**

   * Train multiple classifiers (Random Forest, SVM, Naïve Bayes, etc.)
   * Perform **cross-validation** and **GridSearchCV** for hyperparameter tuning
   * Compare performance across models

5. **Results & Model Selection**

   * Evaluate using **classification reports** and accuracy metrics
   * Select the best-performing model
   * Save the model for future inference

---

## 🔹 Tools & Technologies

* **Languages:** Python
* **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
* **Techniques:** EDA, Feature Selection, SMOTE, Hyperparameter Tuning (GridSearchCV)

---

## 🔹 Results

* Developed multiple machine learning models for classification.
* Identified key features influencing semiconductor yield.
* Selected the best-performing model based on train-test accuracy and classification metrics.

---

## 🔹 Future Improvements

* Apply **deep learning methods** for feature extraction.
* Use **advanced dimensionality reduction** (PCA, t-SNE) for high-dimensional data.
* Deploy model as a web app for real-time predictions.

---

## 🔹 How to Run

1. Clone the repository:

   ```bash
   git clone <repo_url>
   cd semiconductor-yield-classification
   ```
2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```
3. Open the Jupyter Notebook and run step by step:

   ```bash
   jupyter notebook SEMICONDCUTOR_MANUFACTURING_PROCESS.ipynb
   ```

---


