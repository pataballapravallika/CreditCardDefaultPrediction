# Credit Card Default Prediction

This project focuses on predicting whether a customer will default on their credit card payment in the next month using a logistic regression model. The dataset used is the **UCI Credit Card Dataset**, and the project involves data preprocessing, exploratory data analysis (EDA), feature engineering, model building, and evaluation.

---

## 📂 Dataset

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **File**: `UCI_Credit_Card.csv`
- **Features**:
  - Demographic: `LIMIT_BAL`, `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - Payment history: `PAY_0` to `PAY_6`
  - Bill amounts: `BILL_AMT1` to `BILL_AMT6`
  - Payment amounts: `PAY_AMT1` to `PAY_AMT6`
  - **Target**: `default.payment.next.month` (1 = default, 0 = not default)

---

## 🧪 Project Steps

### 1. **Data Loading**
Data is loaded from IBM Cloud Object Storage using the `ibm_boto3` library.

### 2. **Handling Missing Values**
- Checked for and handled missing values using forward fill.

### 3. **Exploratory Data Analysis**
- Visualized distributions (e.g., `LIMIT_BAL`)
- Correlation heatmap to identify strongly related features
- Boxplots to analyze relationships between features and the target

### 4. **Feature Engineering**
- Created new feature: `balance_limit_ratio = BILL_AMT1 / LIMIT_BAL`

### 5. **Data Preprocessing**
- Separated target variable
- Normalized features using `StandardScaler`

### 6. **Model Building**
- Used Logistic Regression
- Split data into train and test sets (80/20)
- Trained model and evaluated with ROC AUC Score

### 7. **Model Evaluation**
- Calculated **ROC AUC Score**
- Plotted **K-S Statistic** to measure performance

---

## 📊 Results

- **ROC AUC Score**: _(Displayed in console output)_
- **KS Statistic**: _(Displayed in console output and plotted)_

---

## 📦 Requirements

- Python 3.x
- pandas
- numpy
- seaborn
- matplotlib
- scikit-learn
- ibm_boto3
- botocore

Install dependencies using:
```bash
pip install pandas numpy seaborn matplotlib scikit-learn ibm_boto3
