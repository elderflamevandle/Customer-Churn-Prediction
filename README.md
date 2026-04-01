# 📱 Telecom Customer Churn Prediction System

This project provides a comprehensive solution for predicting customer churn in a telecom company using machine learning. It includes a detailed data analysis and model training pipeline in a Jupyter Notebook (`code.ipynb`) and an interactive web application built with Streamlit (`main.py`) for single and bulk churn predictions.

## ✨ Features

*   **Data Preprocessing & EDA**: Thorough cleaning, transformation, and exploratory data analysis of telecom customer data.
*   **Multiple ML Models**: Training and evaluation of various classification models (Logistic Regression, Random Forest, XGBoost, Gradient Boosting, etc.) to identify the best performer.
*   **Hyperparameter Tuning**: Optimization of top-performing models for enhanced accuracy and robustness.
*   **Interactive Streamlit UI**:
    *   **Single Prediction**: Predict churn for individual customers by inputting their details.
    *   **Bulk Prediction**: Upload CSV/Excel files to get churn predictions for multiple customers.
    *   **Model & Data Insights**: Visualize model performance metrics and key dataset characteristics.
*   **Model Persistence**: Trained models and preprocessing objects are saved for efficient deployment.

## 🚀 Machine Learning Model

The `code.ipynb` notebook details the entire machine learning workflow:

1.  **Data Loading**: Reads the `telecom_customer_churn.csv` dataset.
2.  **Data Cleaning**: Handles missing values, identifies and processes the target variable.
3.  **Exploratory Data Analysis (EDA)**: Visualizes distributions, correlations, and churn rates across various features.
4.  **Feature Engineering**: Creates new features like `AvgChargesPerMonth`, `ChargesRatio`, `TenureGroup`, and `TotalServices`.
5.  **Feature Scaling & Encoding**: Applies `StandardScaler` to numerical features and `LabelEncoder`/One-Hot Encoding to categorical features.
6.  **Model Training & Evaluation**: Trains and evaluates multiple baseline models, performs hyperparameter tuning for top models (Random Forest, XGBoost, Gradient Boosting), and conducts cross-validation.
7.  **Model Saving**: The best performing models, `scaler.pkl`, `label_encoders.pkl`, and `feature_names.pkl` are saved in the `models/` directory for use in the Streamlit application.

## 📊 Dataset

The project utilizes the `telecom_customer_churn.csv` dataset, which contains various customer attributes and their churn status.

## 📦 Project Structure

*   `code.ipynb`: Jupyter Notebook containing the ML model training, preprocessing, and analysis.
*   `main.py`: Streamlit application code for the interactive UI.
*   `telecom_customer_churn.csv`: The raw dataset used for training.
*   `scaler.pkl`: Saved `StandardScaler` object.
*   `models/`: Directory containing all trained machine learning models (`random_forest_model.pkl`, `xgboost_model.pkl`, `gradient_boosting_model.pkl`, `logistic_regression_model.pkl`) and other preprocessing artifacts (`feature_names.pkl`, `label_encoders.pkl`, `model_performance.csv`).

## ⚙️ Setup Instructions

To get this project up and running on your local machine, follow these steps:

1.  **Download the Project**:
    Download the project ZIP file from the provided Google Drive link or clone the repository.
    Unzip the file to your desired location.

2.  **Create a Virtual Environment**:
    Open your terminal or command prompt in the project's root directory.
    Use `uv` to create a virtual environment (If you want to install uv, use the command: `pip install uv`):
    ```bash
    uv venv
    ```

3.  **Activate the Virtual Environment**:
    *   **Windows**:
        ```bash
        .venv\Scripts\activate
        ```
    *   **macOS/Linux**:
        ```bash
        source .venv/bin/activate
        ```

4.  **Install Dependencies**:
    With the virtual environment activated, install all required libraries using `uv sync`:
    ```bash
    uv sync
    ```

5.  **Run the Streamlit Application**:
    Once all dependencies are installed, you can launch the Streamlit application:
    ```bash
    streamlit run main.py
    ```
    This command will open the application in your default web browser.
