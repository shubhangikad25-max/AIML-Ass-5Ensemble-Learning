# Practical Assignment 5: Ensemble Learning and Performance Evaluation

## Objective
To implement and compare different ensemble learning techniques and individual machine learning classifiers using classification performance metrics.

## Dataset
Breast Cancer Wisconsin Dataset from Scikit-learn.

- Total Samples: 569
- Features: 30
- Training Samples: 455
- Testing Samples: 114

## Algorithms Implemented

### Base Classifiers
1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)

### Ensemble Classifiers
1. Stacking Classifier
2. AdaBoost Classifier
3. Voting Classifier (Soft Voting)

## Technologies Used
- Python
- Google Colab
- Scikit-learn
- NumPy
- Pandas
- Matplotlib
- Seaborn

## Performance Evaluation

The models are evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 98.25% |
| Decision Tree | 93.86% |
| KNN | 95.61% |
| Stacking | 98.25% |
| AdaBoost | 95.61% |
| Voting | 95.61% |

Stacking achieved the highest F1-Score of 98.63%.

## How to Run

1. Download the `Ensemble_Learning.ipynb` file.
2. Open Google Colab.
3. Upload the notebook.
4. Run all cells.

The dataset is loaded directly from Scikit-learn, so no separate dataset download is required.

## Conclusion

This practical demonstrates how ensemble learning techniques combine multiple classifiers to make predictions. Their performance was compared with individual classifiers to understand their advantages and limitations.
