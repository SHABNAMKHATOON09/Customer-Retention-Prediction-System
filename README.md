# Customer Retention Prediction and Analytics System

A complete, live **Python + Streamlit** Machine Learning & Business Analytics application built according to the BCA Project Synopsis (*Maharishi University of Information Technology, Noida*).

---

## 1. Project Architecture & Synopsis Alignment

| Synopsis Process (DFD Level 1) | Implemented Module | Functionality |
|---|---|---|
| **Process 1 — Data Collection** | `modules/data_loader.py` | CSV upload, built-in sample dataset (`data/customer_retention_sample.csv`), schema & missing/duplicate diagnostics, automatic ID & target column detection |
| **Process 2 — Data Preprocessing** | `modules/preprocessing.py` | Missing value imputation, duplicate removal, numeric string coercion, Customer ID exclusion, leakage-safe `ColumnTransformer` (`SimpleImputer`, `StandardScaler`, `OneHotEncoder`) |
| **Process 3 — Data Analysis (EDA)** | `modules/eda.py` | Live Matplotlib & Seaborn charts: Retention & Churn distributions, Tenure analysis, Monthly Charges spread, Contract Type churn rate, Correlation Heatmap, Demographic & Service breakdowns |
| **Process 4 — Machine Learning** | `modules/model_trainer.py` | Stratified 80:20 train-test split, K-Fold Cross-Validation, Baseline Logistic Regression & Random Forest Classifier, Accuracy/Precision/Recall/F1/ROC-AUC, Confusion Matrix, Feature Importance, Joblib persistence (`models/`) |
| **Process 5 — Prediction & Reporting** | `modules/predictor.py`, `modules/report_generator.py` | Individual customer risk prediction + probabilities + retention actions, Batch CSV prediction + export, downloadable HTML (with embedded charts) & Markdown project reports |
| **Presentation Layer** | `app.py`, `modules/ui_styles.py` | Multi-page interactive Streamlit dashboard with sidebar navigation and executive KPI cards |

---

## 2. Folder Structure

```text
customer-retention-system/
├── app.py                              # Main Streamlit entry point (streamlit run app.py)
├── requirements.txt                    # Python dependencies
├── README.md                           # Installation, run, demo, and test guide
├── data/
│   └── customer_retention_sample.csv   # Built-in 1,215-row sample customer dataset
├── models/                             # Saved .joblib ML pipelines
├── reports/                            # Generated HTML & Markdown reports
├── modules/
│   ├── __init__.py
│   ├── data_loader.py                  # Process 1: Loading, inspection & target mapping
│   ├── preprocessing.py                # Process 2: Leakage-safe cleaning & transformers
│   ├── eda.py                          # Process 3: Matplotlib & Seaborn visualizations
│   ├── model_trainer.py                # Process 4: 80:20 ML training, CV & evaluation
│   ├── predictor.py                    # Process 5a: Single & batch prediction engine
│   ├── report_generator.py             # Process 5b: HTML & Markdown report builder
│   └── ui_styles.py                    # Dashboard CSS theme & KPI components
├── scripts/
│   └── generate_sample_data.py         # Script to regenerate sample dataset
└── tests/
    └── test_pipeline.py                # Automated end-to-end verification tests
```

---

## 3. Installation Instructions

Make sure **Python 3.9+** is installed on your system.

### Step 1: Open Terminal in the Project Folder
```bash
cd <project-folder>
```

### Step 2: Create & Activate a Virtual Environment (Recommended)

**Windows (Command Prompt / PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## 4. Exact Command to Run the Live Project

```bash
streamlit run app.py
```

Once started, Streamlit will print a local URL in the terminal (usually **`http://localhost:8501`**) and automatically open your default browser.

---

## 5. How to Demonstrate Each Module (College Evaluation Walkthrough)

1. **Dashboard Overview (`🏠 Dashboard Overview`)**
   - Opens automatically with the built-in sample dataset (`data/customer_retention_sample.csv`).
   - Show live KPIs: **Total Customers**, **Retained Customers**, **At-Risk / Churned**, **Retention Rate**, and **Churn Rate**.
   - Highlight the Retention Distribution, Churn Share Donut, and Contract Type Churn charts.

2. **Dataset Analysis (`🗂️ Dataset Analysis`)**
   - Demonstrate raw data diagnostics: 1,215 rows, 16 columns, detected missing values, and 15 duplicate rows.
   - Show target column selection (`Retention_Status`) and automatic exclusion of `Customer_ID`.
   - Click **"Apply Configuration & Run Preprocessing"** to show duplicate removal, missing value handling, and leakage-prevention audit.
   - Optionally upload any external customer churn CSV via **"Upload Customer Dataset (CSV)"**.

3. **Data Visualization (`📊 Data Visualization (EDA)`)**
   - Walk through all real-time Matplotlib & Seaborn graphs:
     - Retention & Churn Distributions
     - Retention by Tenure & Quartile Churn Rate
     - Monthly Charges Boxplot & KDE Density
     - Retention by Contract Type
     - Numerical Correlation Heatmap
     - Demographic & Service breakdowns (`Gender`, `Payment_Method`, `Tech_Support`, `Internet_Service`, etc.)

4. **Model Training & Evaluation (`🤖 Model Training & Evaluation`)**
   - Click **"Train Baseline & Random Forest Models"**.
   - Show the **80:20 Stratified Split** (960 train / 240 test), **5-Fold Cross-Validation**, and side-by-side comparison of **Baseline (Logistic Regression)** vs **Random Forest Classifier**.
   - Show **Accuracy, Precision, Recall, F1-Score, ROC-AUC**, **Confusion Matrix**, and **Top Feature Drivers**.
   - Show that the trained model is saved as a `.joblib` file inside `models/` and can be reloaded anytime.

5. **Customer Prediction (`🎯 Customer Prediction`)**
   - **Individual Prediction:** Enter customer attributes (e.g., short tenure `3`, `Month-to-Month`, high `Monthly_Charges`, `Electronic Check`, `Support_Tickets = 4`) and click **Predict Customer Retention** to show the predicted risk class, exact retention/churn probabilities, and retention recommendations.
   - **Batch CSV Prediction:** Download the sample 50-row batch CSV button or upload any CSV, run batch prediction, view risk tiers, and download `customer_retention_predictions.csv`.

6. **Reports & Export (`📄 Reports & Export`)**
   - Click **"Generate Project Report"** to compile dataset stats, preprocessing log, live embedded EDA charts, model evaluation metrics, and prediction summaries.
   - Download the self-contained **HTML Report** (open in Chrome/Edge and press `Ctrl+P` → *Save as PDF* if a PDF copy is needed) or the **Markdown Summary Report**.

---

## 6. Automated Verification / Testing Commands

Run the end-to-end test suite to verify every module:

```bash
pytest -v tests/test_pipeline.py
```
