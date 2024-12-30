# EEG Data Analysis for Attention Classification

This repository contains code for analyzing EEG data to classify attention levels in a multi-class classification problem. The goal of this analysis is to use different EEG features, including time-domain and frequency-domain features, to predict different paradigms associated with attention.

## Overview

This project processes EEG data from multiple subjects (24 subjects in total) and extracts features relevant to classification, such as power spectral densities across various frequency bands. The resulting data is then used to train multiple classifiers to predict the mental state of attention under various tasks.

The main tasks are:
- Preprocessing EEG data (bandpass filtering)
- Extracting statistical and spectral features
- Training classifiers on the extracted features
- Evaluating classifier performance using various metrics (confusion matrix, classification report)

## Features and Tasks

The tasks in the dataset are related to different attention paradigms, such as:
- `baseline_eyesclosed`
- `baseline_eyesopen`
- `dual-task_paradigm`
- `oddball_paradigm`
- `stroop_task`
- `task-switching_paradigm`

The model uses these tasks to classify EEG signals into categories. The focus is on analyzing the EEG signals during these tasks to detect varying levels of attention and cognitive load.

## Libraries Used

- **NumPy**: For numerical operations.
- **Pandas**: For handling and processing EEG data.
- **Scipy**: For signal processing (filtering and spectral analysis).
- **Scikit-learn**: For machine learning (classification, feature selection, scaling).
- **XGBoost**: For the XGBoost classifier.
- **Imbalanced-learn (SMOTE)**: For addressing class imbalance during training.
- **MNE**: For EEG data visualization and topographic mapping.
- **Matplotlib & Seaborn**: For data visualization (feature distribution, confusion matrix, etc.).

## Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/wizardwithcodehazard/Multi-classification-Data-Analysis.git
   cd Multi-classification-Data-Analysis
   ```

2. **Install dependencies**:
   You can install the required libraries by using `pip`

## Code Description

### 1. **Preprocessing and Filtering**:
   EEG data is bandpass filtered between 1 Hz and 50 Hz using a Butterworth filter to remove noise and focus on the relevant brainwave frequencies.

   ```python
   def butter_bandpass(lowcut, highcut, fs, order=5):
       # Bandpass filter function
   ```

### 2. **Feature Extraction**:
   Statistical and spectral features are extracted from the EEG data:
   - **Time-domain features**: Mean, Standard deviation, Min, Max
   - **Frequency-domain features**: Power Spectral Density (PSD) in frequency bands (Delta, Theta, Alpha, Beta, Gamma)
   - **Ratios**: Alpha/Beta ratio, Theta/Alpha ratio
   
   ```python
   def extract_features(eeg_data, fs=250):
       # Extracts statistical and spectral features from the EEG signals
   ```

### 3. **Data Processing**:
   The code processes EEG data for multiple subjects, loading the data from CSV files and applying filtering and feature extraction.

   ```python
   def process_subject(subject_folder):
       # Process EEG files for a single subject
   ```

   The data from all subjects is combined for training the classifier:
   
   ```python
   def process_all_subjects(base_path, num_subjects=24):
       # Process EEG data for all subjects
   ```

### 4. **Data Visualization**:
   - **Correlation Heatmap**: Visualizes correlations between different frequency band powers.
   - **Feature Distributions**: Boxplots to visualize the distribution of features across different paradigms.
   - **Topographic Map**: Visualizes EEG data in the form of topographic maps using MNE.

   ```python
   def visualize_data(data):
       # Visualize feature correlations
   ```

### 5. **Model Training**:
   A **Voting Classifier** (RandomForest, XGBoost, and SVM) is used to predict paradigms. The model also handles class imbalance using SMOTE and performs feature selection using Recursive Feature Elimination (RFE).

   ```python
   def train_classifier(data):
       # Train classifier and evaluate performance
   ```

### 6. **Evaluation**:
   The classifier's performance is evaluated using:
   - **Classification Report**
   - **Confusion Matrix**: To visualize the model’s accuracy on different tasks.

   ```python
   def plot_confusion_matrix(y_test, y_pred, label_encoder, title='Confusion Matrix'):
       # Plot confusion matrix with task names
   ```

### 7. **Main Execution**:
   The `main()` function processes all the data, trains the model, and generates visualizations and evaluation metrics.

   ```python
   def main():
       # Main function to process data and train classifier
   ```

This will process the EEG data, extract features, train the classifier, and display the results (including confusion matrix and classification report).

## Contributing

Feel free to fork the repository and contribute by:
- Adding new features for EEG signal processing.
- Improving the classification models.
- Creating new visualizations.
- Providing bug fixes or performance optimizations.

Please submit pull requests for any changes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

This work is based on EEG data processing techniques and machine learning methods for time-series classification. 
