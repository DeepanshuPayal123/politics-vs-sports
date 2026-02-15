# Problem 4: Sports vs. Politics Classifier

**Name:** Deepanshu

**Roll Number:** B23CS1012 

**Assignment:** 1 

**Date:** February 15, 2026  

---

## Overview
This project implements a text classification system to distinguish between **Sports** and **Politics** articles. The goal is to compare the performance of three different Machine Learning algorithms on a text dataset.

The system uses a synthetic dataset of **labeled sentences** processed using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features for training.

---

## Methodology
- **Data Source:** Synthetic dataset generated within the script.  
- **Feature Extraction:** TF-IDF Vectorization.  
- **Algorithms Compared:**
  1. **Naive Bayes** (MultinomialNB)
  2. **Support Vector Machine** (SVM - Linear Kernel)
  3. **Logistic Regression**

---

## Prerequisites
The code requires Python and the `scikit-learn` library.

```bash
pip install scikit-learn numpy
```

---

## How to Run
1. Navigate to the directory containing the file.
2. Run the Python script using the terminal:

```bash
python B23CS1012_prob4.py
```


---

## Expected Output
The script outputs:
- Dataset statistics  
- A comparison table of evaluation metrics  
- Confusion matrices for each model  

---

## Sample Result Snapshot

Based on the experimental run, the models achieved the following performance:

```plaintext
=================================================================
Model Name           | Accuracy   | Precision* | Recall*
-----------------------------------------------------------------
Naive Bayes          | 0.8000     | 0.8700     | 0.8000
SVM (Linear)         | 0.8000     | 0.8700     | 0.8000
Logistic Reg         | 0.6500     | 0.8100     | 0.6500
=================================================================
*Precision and Recall values are weighted averages.
```

---

## Detailed Confusion Matrices

(Row = Actual, Column = Predicted)  
Order: `[Politics, Sports]`

### 1. Naive Bayes & SVM (Identical Matrices)

```plaintext
[[8 0]   <-- Correctly identified 8 Politics articles
 [4 8]]  <-- Misclassified 4 Sports articles as Politics
```

### 2. Logistic Regression

```plaintext
[[8 0]   <-- Correctly identified 8 Politics articles
 [7 5]]  <-- Misclassified 7 Sports articles as Politics
```

---

## Discussion

### Performance
Naive Bayes and SVM performed best with an accuracy of **80%**. They successfully identified all Politics articles (Recall = 1.00 for Politics) but struggled slightly with Sports articles, misclassifying approximately 33% of them.

### Logistic Regression Limitations
Logistic Regression achieved the lowest accuracy (**65%**). The confusion matrix shows it correctly identified Politics headlines but performed poorly on Sports headlines, misclassifying more than half of them (7 out of 12).

### Conclusion
For this specific dataset size, **Naive Bayes and SVM proved to be more robust classifiers** than Logistic Regression. Increasing the dataset size and diversity would likely improve performance across all models.
