## ALS Detection Using Neural Networks and Support Vector Machines

This project implements a machine learning pipeline to detect Amyotrophic Lateral Sclerosis (ALS) from CSV-formatted biosignal data. It uses a shallow neural network (MLPClassifier) and Support Vector Machines (SVC) to distinguish between ALS and normal samples, achieving high accuracy.

---

### Table of Contents
- [Project Overview](#project-overview)
- [Approach](#approach)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Data Format](#data-format)
- [Results](#results)

---

## Project Overview
This repository contains code and data for classifying ALS vs. normal samples using biosignal features. The model is trained on provided CSV files and evaluated for accuracy, sensitivity, and specificity.

## Approach
- **Data Preparation:**
  - Data is split into ALS (`A01.csv`-`A11.csv`) and normal (`N01.csv`-`N11.csv`) samples in the `ALSDetection_data/` folder.
  - Each file contains biosignal features as comma-separated values.
  - For training, one pair (ALS and normal) is randomly left out for testing (leave-one-out cross-validation).
- **Model:**
	- Two approaches are provided:
		- A shallow neural network (`MLPClassifier` from scikit-learn) with early stopping and L2 regularization to prevent overfitting.
		- A support vector machine (SVM) classifier (`SVC` from scikit-learn) with regularization and RBF kernel to help prevent overfitting.
	- Data is standardized using `StandardScaler` in both approaches.
- **Evaluation:**
  - Model performance is measured using accuracy, sensitivity, and specificity.
  - Confusion matrix is printed for detailed analysis.

## Setup & Installation
1. **Clone the repository:**
	```powershell
	git clone https://github.com/aryanjverma/als-detection.git
	cd als-detection
	```
2. **Install dependencies:**
	Ensure you have Python 3.8+ installed. Then run:
	```powershell
	pip install -r requirements.txt
	```

## How to Run
1. **Test the neural network model:**
	```powershell
	python test.py
	```
	This will process the data, load the neural network, and print the confusion matrix, sensitivity, specificity, and overall accuracy.

2. **Test the SVM model:**
	```powershell
	python svm.py
	```
	This will train and save the SVM model to `svm.joblib`.
	Then, to evaluate the SVM model:
	```powershell
	python svm_test.py
	```
	This will print the confusion matrix, sensitivity, specificity, and overall accuracy for the SVM approach.

3. **Predict on your own data (neural network):**
	Ensure `model.joblib` and `svm.joblib` exists.
	```powershell
	python predict.py
	```
	The script will prompt you to enter the file location of the data you want to analyze, for example:
	```powershell
	Enter the file you want to test (must be in correct format): ALSDetection_data\N04.csv
	```
    Then, when prompted, choose either the SVM or NN.
	The model will then predict if the person has ALS or not.
## Data Format
- All data is stored in the `ALSDetection_data/` directory.
- Files are named `A01.csv`-`A11.csv` (ALS) and `N01.csv`-`N11.csv` (Normal).
- Each file contains rows of comma-separated biosignal features.

## Results
- **Neural Network (MLPClassifier):**
	- Accuracy: ~94.6%
	- Sensitivity: ~98.5%
	- Specificity: ~84.9%
- **SVM:**
	- Accuracy: ~96.5%
	- Sensitivity: ~99.8%
	- Specificity: ~88.2%

## Notes
- The approach uses leave-one-out cross-validation for robust evaluation.
- SVM was tested but did not perform as well as the neural network.
- Make sure that samples are in the same format as those in the csv's in ALSDetection_data.

---

