# Human Activity Recognition Using Machine Learning

A machine learning project that recognizes six human physical activities using smartphone sensor measurements. This project compares multiple classification algorithms and presents the results through an interactive Streamlit dashboard.

## Project Overview

Human Activity Recognition (HAR) uses sensor data to identify physical activities performed by people. This project uses the UCI Human Activity Recognition Using Smartphones dataset to classify sensor feature records into six activity categories.

### Activities Recognized

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

## Dataset

**Source:** [UCI Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

- Training records: 7,352
- Testing records: 2,947
- Features per record: 561
- Activity classes: 6

The dataset contains engineered features derived from smartphone accelerometer and gyroscope signals.

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib
- Streamlit
- Joblib
- Git and GitHub

## Machine Learning Models

Three classification algorithms were evaluated using the dataset's predefined training and test sets:

| Model | Test Accuracy | Macro F1 Score |
|---|---:|---:|
| Linear SVM | 96.71% | 96.74% |
| Logistic Regression | 96.10% | 96.09% |
| Random Forest | 92.57% | 92.38% |

**Best-performing model:** Linear Support Vector Machine (Linear SVM), based on the measured test accuracy and macro F1 score.

## Model Evaluation

The Linear SVM achieved:

- Accuracy: 96.71%
- Macro F1 Score: 96.74%
- Weighted F1 Score: 96.69%

The evaluation also identified differences in performance between activities. Sitting had the lowest recall at 87.17%, indicating that some sitting records were incorrectly classified, particularly as standing.

## Interactive Dashboard

The Streamlit application includes:

- Project overview and supported activities
- CSV-based activity prediction using 561 engineered features
- Model comparison results
- Linear SVM confusion matrix
- Dataset information

The prediction feature expects numeric CSV data with exactly 561 columns in the original dataset's feature order. It does not directly process raw smartphone sensor streams.

## Project Structure

```text
human-activity-recognition/
├── UCI HAR Dataset/
├── models/
│   ├── linear_svm_model.joblib
│   └── random_forest_model.joblib
├── results/
│   ├── confusion_matrix.png
│   ├── linear_svm_confusion_matrix.png
│   ├── model_comparison.csv
│   └── model_comparison.png
├── app.py
├── check_dataset.py
├── compare_models.py
├── evaluate_model.py
├── explore_activities.py
├── load_data.py
├── plot_model_comparison.py
├── train_model.py
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/ImashaCheni2003/human-activity-recognition.git
cd human-activity-recognition
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install pandas numpy matplotlib scikit-learn streamlit joblib
```

### 4. Run the dashboard

```bash
streamlit run app.py
```

Ensure the dataset and trained model files are available in the locations expected by the application.

## Future Improvements

- Evaluate additional machine learning algorithms.
- Investigate sitting-versus-standing misclassifications.
- Add per-activity performance visualizations.
- Improve prediction input validation and user experience.
- Explore real-time smartphone sensor integration.

## Acknowledgements

Thanks to the University of California, Irvine (UCI) Machine Learning Repository for providing the Human Activity Recognition Using Smartphones dataset.
