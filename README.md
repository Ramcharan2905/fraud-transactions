# Bank Transaction Fraud Detection (Anomaly Detection)

This repository implements an **unsupervised anomaly detection** model to identify fraudulent bank transactions.

This project moves away from traditional supervised classification and instead uses outlier detection algorithms like **Isolation Forest** and **Local Outlier Factor (LOF)** to find rare and suspicious activities that deviate from normal transaction patterns.

---

## 📘 Overview

This project demonstrates a complete unsupervised machine learning workflow for fraud detection. The process involves:
1.  Loading and exploring the transaction dataset.
2.  Preprocessing the data using `StandardScaler` to normalize features.
3.  Applying dimensionality reduction techniques (`PCA` and `t-SNE`) to visualize the high-dimensional data.
4.  Implementing and comparing multiple unsupervised models (`KMeans`, `IsolationForest`, `LocalOutlierFactor`) to identify anomalies.

---

## 🧠 Problem Statement

Given a dataset of bank transactions, identify fraudulent activities.

The key challenge is that "fraud" labels are often unavailable or unreliable. Therefore, the problem is framed as an **anomaly detection task**: can we identify transactions that are so unusual and different from the rest that they warrant investigation as potential fraud?

---

## ⚙️ Tech Stack

-   **Python 3.x**
-   **Jupyter Notebook**
-   **Pandas** (for data manipulation)
-   **NumPy** (for numerical operations)
-   **scikit-learn (Sklearn)** (for models, scaling, and metrics)
-   **Matplotlib** & **Seaborn** (for data visualization)

---

## 🚀 Features

-   **Data Preprocessing**: Feature scaling with `StandardScaler` to ensure models are not biased by feature ranges.
-   **Data Visualization**: Uses `PCA` and `t-SNE` to reduce data to 2D for visualizing clusters and potential outliers.
-   **Clustering as Anomaly Detection**:
    -   **K-Means**: Uses clustering to group transactions, with the assumption that anomalies will be far from cluster centers.
-   **Dedicated Anomaly Detection Models**:
    -   **Isolation Forest**: A tree-based model that explicitly "isolates" anomalies by finding data points that are easier to separate from the rest.
    -   **Local Outlier Factor (LOF)**: A density-based model that identifies anomalies by comparing their local density to that of their neighbors.
-   **Model Evaluation**: Uses unsupervised metrics like `silhouette_score` and `calinski_harabasz_score` to evaluate the quality of the clusters found.

---

## ▶️ How to Run

1.  **Clone the repository**
    ```bash
    git clone https://github.com/Ramcharan2905/fraud-transactions/
    cd bank-fraud-detection
    ```

2.  **Set up the environment** (Recommended)
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Get the Data**
    The notebook mentions downloading the data from Kaggle. You will need to download the appropriate CSV file, rename it (e.g., to `data.csv`), and place it in the project's root directory.

5.  **Run notebook**
    ```bash
    jupyter notebook
    ```
    Open `bank_transaction_fraud_detection.ipynb` to run the analysis.

---

## 📈 Learning Purpose

This project aims to help practitioners understand:
-   How to frame fraud detection as an **unsupervised learning** problem.
-   The difference between clustering (`KMeans`) and dedicated anomaly detection algorithms (`IsolationForest`, `LOF`).
-   The importance of feature scaling and dimensionality reduction for complex datasets.
-   How to apply and compare different anomaly detection techniques to find suspicious pat
