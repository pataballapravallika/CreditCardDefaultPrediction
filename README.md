# 💳 Credit Card Default Prediction

This project aims to predict whether a credit card holder will default on payment next month using machine learning techniques. The analysis is done on the popular **UCI Credit Card Dataset**, and a **Logistic Regression** model is built using Python.

---

## 📂 Dataset Information

- **Source**: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
- **File**: `UCI_Credit_Card.csv`
- **Target Column**: `default.payment.next.month`
- **Total Instances**: 30,000
- **Features**: 23 (including credit limit, payment history, demographic info)

---

## 🛠️ Technologies Used

- Python 3.x  
- Jupyter Notebook  
- IBM Cloud Object Storage  
- Libraries:
  - `pandas`, `numpy` – Data manipulation
  - `matplotlib`, `seaborn` – Visualization
  - `scikit-learn` – Model building and evaluation

---

## 🚀 Workflow

### 1. **Data Loading**
- Data is accessed securely from IBM Cloud Object Storage using `ibm_boto3`.

### 2. **Preprocessing**
- Missing values handled using forward-fill (`ffill`)
- Created a new feature: `balance_limit_ratio = BILL_AMT1 / LIMIT_BAL`
- Normalized numerical features using `StandardScaler`

### 3. **Exploratory Data Analysis (EDA)**
- Histogram of credit limits
- Correlation heatmap of all features
- Boxplot: Credit limit vs Default status

### 4. **Model Building**
- Logistic Regression model using `scikit-learn`
- Data split: 80% training, 20% testing
- Evaluation Metrics:
  - **ROC AUC Score**
  - **KS Statistic** (Kolmogorov-Smirnov test)

---

## 📈 Evaluation Results

- **ROC AUC Score**: *Displayed in notebook output*
- **KS Statistic**: *Displayed in notebook output*
- Plotted ROC curve and K-S chart

---

## 📊 Visualizations

- 📉 Histogram of `LIMIT_BAL`
- 🔥 Heatmap of correlations
- 📦 Boxplot showing `LIMIT_BAL` distribution across default status
- 📊 K-S Chart to evaluate classifier performance

---

## 📁 Files

- `credit_card_default_prediction.ipynb` – Main notebook containing code and outputs
- `UCI_Credit_Card.csv` – Dataset (accessed via IBM Cloud)
- `README.md` – Project overview

---

## 📌 Future Improvements

- Try more advanced classifiers: Random Forest, XGBoost, etc.
- Hyperparameter tuning using GridSearchCV
- Address class imbalance using SMOTE or class weights
- Deploy model using Flask or Streamlit

---

## 🙋‍♀️ Author

**Pravallika Pataballa**  
📧 [pataballapravallika@gmail.com](mailto:pataballapravallika@gmail.com)  
🔗 [LinkedIn](https://www.linkedin.com/in/pravallika-pataballa-923572286/)  
💻 [GitHub](https://github.com/pataballapravallika)

---

## 📌 License

This project is licensed for educational and research purposes only.

