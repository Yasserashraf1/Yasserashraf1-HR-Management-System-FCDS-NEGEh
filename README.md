# Yasserashraf1-HR-Management-System-FCDS-NEGEh
# 📌 Overview 
This **HR Management System** combines manual employee record management with **AI-driven predictive analytics** to help HR professionals streamline hiring decisions, analyze employee data, and automate repetitive tasks. 

This project offers two distinct implementations of an HR Management System:

1. Interactive_NEGEhr.ipynb - A terminal-based version for local execution.
2. FCDS_NEGEhr.py - A web-based interactive UI.
   
Both versions provide:

* **Complete employee record management (CRUD operations)**

* **AI-powered hiring prediction using machine learning**

* **SQLite database backend**

🔹 Manual HR Mode → CRUD operations for employee records. 

🔹 AI-Powered HR Mode → Machine learning models to predict employment suitability. 

🔹 Database Integration → SQLite for efficient data storage. 

🔹 Automated EDA & Preprocessing → Clean and analyze HR data efficiently. 

# Key Features
## 1️⃣ Manual HR Mode
**📝 Add Employees:** Store employee details (age, education, salary, skills, etc.). 

* **🔍 Retrieve Employees:** Search by ID or view database records. 

* **🔄 Update Records:** Modify employee information. 

* **❌ Delete Employees:** Remove records from the database. 

* **📊 View Database:** Preview the first N rows of the dataset.

## 2️⃣ AI-Powered HR Mode
* **🤖 Train ML Models:**
  * Logistic Regression
  * Decision Tree
  * Random Forest

* **📊 Model Evaluation:**

   * Accuracy scores

   * Classification reports

   * Confusion Matrix (visualized)

   * ROC-AUC curves

* **🔮 Predict Employment Suitability:**

    * Input candidate details (age, skills, salary, etc.)

    * Get "Accepted" or "Rejected" prediction

## 3️⃣ Data Management
* **📂 SQLite Database:** Stores all employee records

* **🔄 CSV Import:** Loads initial dataset (stackoverflow_full.csv)

* **🧹 Automated Preprocessing:**

    * Missing value handling

    * Outlier removal (IQR method)

    * Categorical encoding (Label Encoding)

# 📊 Data Flow & AI Model Workflow
1. **Data Loading** → Import from CSV into SQLite database
2. **Preprocessing →**
   * Handle missing values
   * Encode categorical data
   * Remove outliers
3. **Model Training →**
   * Split data (80% train, 20% test)
   * Train Logistic Regression, Decision Tree, Random Forest

4. **Prediction →**
   * Input new candidate details
    * Get AI-based hiring recommendation

# Running the Systems
1. **Command Line Interface Version**
  ``` python Interactive_NEGEhr.ipynb```
## **Features:**
* **✅ Pure Python implementation**
* **✅ Lightweight (no UI dependencies)**
* **✅ Ideal for scripting and automation**

2. Streamlit Web Interface Version
 ```streamlit run  FCDS_NEGEhr.py```
## Features:
* **✅ Interactive web UI**
* **✅ Visual data exploration**
* **✅ User-friendly forms and buttons**

# 📌 Usage Examples
## 1️⃣ Manual HR Mode**
* **Example: Adding an Employee**
```
 Enter Age: 30
 Enter Accessibility (1 or 0): 1
 Enter Education Level: Bachelor's
 Enter Gender: Male
 Worked Before (1 or 0): 1
 ...
 Employee added successfully!
```












