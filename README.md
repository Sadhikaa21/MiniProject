# Brain Stroke Diagnosis Using Deep Learning and Machine Learning Models

## Project Overview

This project aims to develop a robust and accurate model for diagnosing brain strokes using CT scan images. It leverages both deep learning and classical machine learning techniques to classify CT scan slices, offering an automated, reliable solution to assist radiologists and medical practitioners.

The project workflow includes:
- Image preprocessing 
- Deep feature extraction using CNNs
- Feature selection using Genetic Algorithm (GA)
- Classification using ML models (RF, DT, LR, NB, SVM)
- Advanced sequential modeling using LSTM and BiLSTM
- Evaluation using classification and regression metrics

---

## Objective

To build a predictive system that can:
- Accurately detect stroke from brain CT scan images
- Improve on the baseline accuracy (96.5% with BiLSTM)
- Explore and compare multiple models (RF, DT, LR, NB, SVM, LSTM, BiLSTM)
- Use ensemble learning for performance enhancement

---

## Tools & Technologies

- **Language**: Python  
- **Libraries**: TensorFlow, Scikit-learn, NumPy, Pandas, Matplotlib, Seaborn  
- **Modeling**: CNN, LSTM, BiLSTM, Random Forest, SVM, Decision Tree, Logistic Regression, Naive Bayes  
- **Feature Selection**: Genetic Algorithm  
- **Evaluation**: Accuracy, Precision, Recall, F1-score, AUC, MAE, MSE, RMSE

---

## Dataset

- **Type**: Brain CT scan slices  
- **Structure**: 100 Subjects, Each subject has 30 sequential slices (30 timesteps)  
- **Classes**: Stroke / No Stroke  
- **Preprocessing**:
  - Resizing
  - Normalisation
  - Balancing the dataset

---

## Methodology

1. **Preprocessing**
   - Resize to fixed dimension
   - Normalize pixel values
   - Perform oversampling

2. **Feature Extraction**
   - Use pre-defined CNN architectures (VGG16, ResNet50, InceptionV3, MobileNetV2, DenseNet121)

3. **Feature Selection**
   - Genetic Algorithm to select most relevant features
   - Reduce dimensionality and improve model generalization

4. **Classification**
   - Train and evaluate traditional ML classifiers and deep models:
     - **Random Forest (RF)**
     - **Decision Tree (DT)**
     - **Support Vector Machine (SVM)**
     - **Logistic Regression (LR)**
     - **Naive Bayes (NB)**
     - **LSTM and BiLSTM (sequence-based models)**

---

## Results

### Comparison of Base Paper vs. Implementation Results

| **Model**           | **Base Acc.** | **Base Prec.** | **Base Recall** | **Base F1** | **Impl. Acc.** | **Impl. Prec.** | **Impl. Recall** | **Impl. F1** |
|---------------------|---------------|----------------|------------------|-------------|----------------|------------------|-------------------|---------------|
| **SVM**             | 91.45%        | 89.88%         | 92.00%           | 91.00%      | 90.18%         | 87.83%           | 94.10%            | 90.85%        |
| **Random Forest**   | 89.53%        | 91.50%         | 86.73%           | 90.00%      | 92.43%         | 97.91%           | 87.27%            | 92.28%        |
| **Decision Tree**   | 85.00%        | 80.65%         | 88.34%           | 85.00%      | 83.74%         | 82.60%           | 86.96%            | 84.72%        |
| **Logistic Reg.**   | 75.62%        | 68.56%         | 78.40%           | 75.00%      | 91.79%         | 89.50%           | 95.34%            | 92.33%        |
| **Naive Bayes**     | 66.21%        | 70.00%         | 62.15%           | 61.00%      | 65.54%         | 67.88%           | 63.66%            | 65.71%        |
| **LSTM**            | 95.35%        | 92.00%         | 90.89%           | 95.00%      | 96.70%         | 97.09%           | 97.08%            | 97.09%        |
| **BiLSTM**          | 96.45%        | 98.00%         | 93.50%           | 96.00%      | **97.95%**     | **97.71%**       | **98.22%**        | **97.96%**    |

> **Table 4.1**: Comparison of classification metrics between base paper and current implementation.

---

### Insights

- **BiLSTM** outperformed all models, achieving a **97.95% accuracy**, surpassing the base paper’s performance.
- **Random Forest** showed a significant improvement in accuracy and F1-score over the base version.
- **Logistic Regression** had the most notable improvement (+16% accuracy gain), indicating effective preprocessing and feature extraction.
- **Naive Bayes** remains the weakest model in both cases but can be improved with better feature engineering.


