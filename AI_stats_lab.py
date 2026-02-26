"""
Linear & Logistic Regression Lab

Follow the instructions in each function carefully.
DO NOT change function names.
Use random_state=42 everywhere required.
"""

import numpy as np

from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# =========================================================
# QUESTION 1 – Linear Regression Pipeline (Diabetes)
# =========================================================

def diabetes_linear_pipeline():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Split into train and test (80-20).
            Use random_state=42.
    STEP 3: Standardize features using StandardScaler.
            IMPORTANT:
            - Fit scaler only on X_train
            - Transform both X_train and X_test
    STEP 4: Train LinearRegression model.
    STEP 5: Compute:
            - train_mse
            - test_mse
            - train_r2
            - test_r2
    STEP 6: Identify indices of top 3 features
            with largest absolute coefficients.

    RETURN:
        train_mse,
        test_mse,
        train_r2,
        test_r2,
        top_3_feature_indices (list length 3)
    """

    data = load_diabetes()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)


    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)


    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    # Top 3 features by absolute coefficient value
    coef_abs = np.abs(model.coef_)
    top_3_feature_indices = list(np.argsort(coef_abs)[-3:][::-1])

    return train_mse, test_mse, train_r2, test_r2, top_3_feature_indices



# =========================================================
# QUESTION 2 – Cross-Validation (Linear Regression)
# =========================================================

def diabetes_cross_validation():
    """
    STEP 1: Load diabetes dataset.
    STEP 2: Standardize entire dataset (after splitting is NOT needed for CV,
            but use pipeline logic manually).
    STEP 3: Perform 5-fold cross-validation
            using LinearRegression.
            Use scoring='r2'.

    STEP 4: Compute:
            - mean_r2
            - std_r2

    RETURN:
        mean_r2,
        std_r2
    """

    data = load_diabetes()
    X = data.data
    y = data.target

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LinearRegression()
    cv_scores = cross_val_score(
        model, X_scaled, y, cv=5, scoring='r2'
    )

    mean_r2 = cv_scores.mean()
    std_r2 = cv_scores.std()

    return mean_r2, std_r2

# The standard deviation of cross-validation R² scores represents
# the variability of model performance across different data splits.
# A low standard deviation indicates stable and consistent performance,
# while a high standard deviation suggests the model is sensitive to
# how the data is split.


# Cross-validation reduces variance risk by evaluating the model on
# multiple train-validation splits instead of a single split.
# This prevents the model evaluation from depending on a lucky or
# unlucky data split and provides a more reliable estimate of
# generalization performance.


# =========================================================
# QUESTION 3 – Logistic Regression Pipeline (Cancer)
# =========================================================

def cancer_logistic_pipeline():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
            Use random_state=42.
    STEP 3: Standardize features.
    STEP 4: Train LogisticRegression(max_iter=5000).
    STEP 5: Compute:
            - train_accuracy
            - test_accuracy
            - precision
            - recall
            - f1
            - confusion matrix (optional to compute but not return)

    In comments:
        Explain what a False Negative represents medically.

    RETURN:
        train_accuracy,
        test_accuracy,
        precision,
        recall,
        f1
    """

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    
    model = LogisticRegression(max_iter=5000)
    model.fit(X_train_scaled, y_train)

    
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    
    cm = confusion_matrix(y_test, y_test_pred)

    # False Negative Explanation:
    # A False Negative in medical context means the model predicts
    # that a patient does NOT have cancer (Benign),
    # but in reality the patient actually has cancer (Malignant).
    # This is dangerous because the disease may go untreated.

    return train_accuracy, test_accuracy, precision, recall, f1


# =========================================================
# QUESTION 4 – Logistic Regularization Path
# =========================================================

def cancer_logistic_regularization():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Split into train-test (80-20).
    STEP 3: Standardize features.
    STEP 4: For C in [0.01, 0.1, 1, 10, 100]:
            - Train LogisticRegression(max_iter=5000, C=value)
            - Compute train accuracy
            - Compute test accuracy

    STEP 5: Store results in dictionary:
            {
                C_value: (train_accuracy, test_accuracy)
            }

    In comments:
        - What happens when C is very small?
        - What happens when C is very large?
        - Which case causes overfitting?

    RETURN:
        results_dictionary
    """

    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train models with different C values
    C_values = [0.01, 0.1, 1, 10, 100]
    results_dictionary = {}

    for C in C_values:
        model = LogisticRegression(max_iter=5000, C=C)
        model.fit(X_train_scaled, y_train)

        train_pred = model.predict(X_train_scaled)
        test_pred = model.predict(X_test_scaled)

        train_accuracy = accuracy_score(y_train, train_pred)
        test_accuracy = accuracy_score(y_test, test_pred)

        results_dictionary[C] = (train_accuracy, test_accuracy)

    # When C is very small, regularization is strong, coefficients shrink,
    # and the model becomes simpler, which may lead to underfitting.
    # When C is very large, regularization is weak, the model becomes more
    # complex and fits training data very closely.
    # Overfitting typically occurs when C is very large because the model
    # learns noise in the training data.

    return results_dictionary


# =========================================================
# QUESTION 5 – Cross-Validation (Logistic Regression)
# =========================================================

def cancer_cross_validation():
    """
    STEP 1: Load breast cancer dataset.
    STEP 2: Standardize entire dataset.
    STEP 3: Perform 5-fold cross-validation
            using LogisticRegression(C=1, max_iter=5000).
            Use scoring='accuracy'.

    STEP 4: Compute:
            - mean_accuracy
            - std_accuracy

    In comments:
        Explain why cross-validation is especially
        important in medical diagnosis problems.

    RETURN:
        mean_accuracy,
        std_accuracy
    """

    data = load_breast_cancer()
    X = data.data
    y = data.target

    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    
    model = LogisticRegression(C=1, max_iter=5000, random_state=42)
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')

    mean_accuracy = np.mean(cv_scores)
    std_accuracy = np.std(cv_scores)

    # COMMENT: Why cross-validation is critical in medical diagnosis
    # Cross-validation ensures that the model is evaluated on multiple
    # subsets of the data, which reduces the risk of overfitting.
    # In medical diagnosis, false results can be harmful, so CV provides
    # a more reliable estimate of the model's performance on unseen patients.

    return mean_accuracy, std_accuracy
