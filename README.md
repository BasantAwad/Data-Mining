# 📊 Data Mining Project

A comprehensive Data Mining GUI built with Streamlit, capable of performing data cleaning, Principal Component Analysis (PCA), and Association Rule Mining (ARM) via Apriori algorithm.

## ✨ Features

### 1️⃣ Project 1: Data Preprocessing & PCA
- **Data Quality Audit**: Automatically identifies missing values, outliers, negative/unrealistic numerics, and inconsistent category labels (e.g., gender).
- **Automated Cleaning**:
  - Handles missing values (Median/Mode imputation).
  - Caps outliers using the IQR method.
  - Standardizes column names and categorical values.
- **PCA Analysis**:
  - Performs Principal Component Analysis (PCA).
  - Displays Explained Variance Ratio.
  - Visualizes 2D projection of Principal Components.
  - Calculates 3D components (visualization optimized for 2D).

### 2️⃣ Project 2: Association Rule Mining (ARM)
- **Data Quality Audit**: Detects missing dates/amounts, duplicated transactions, and inconsistent formats.
- **Advanced Cleaning**:
  - Standardizes Date and Amount columns.
  - Cleans `Items_Purchased` (removing separators like `;`, `,`, and duplicates).
  - Normalizes `Store_Location` and `Discount_Applied` fields.
- **Association Rule Mining**:
  - Uses the **Apriori Algorithm** to find frequent itemsets.
  - Generates Association Rules (Antecedents → Consequents).
  - Metrics: Support, Confidence, Lift.
  - **Dynamic Parameters**: Adjust `Min Support` and `Min Confidence` via sidebar sliders.
  - Visualizes top rules by Lift.

## 🚀 Installation

### Prerequisites
- Python 3.8+
- [Git](https://git-scm.com/)

### Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/BasantAwad/Data-Mining.git
   cd Data-Mining
   ```

2. **Create a virtual environment (Optional but Recommended)**
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install streamlit pandas numpy scikit-learn matplotlib mlxtend
   ```

## 🛠️ Usage

1. **Run the Application**
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Navigate the Interface**
   - Use the **Sidebar** to switch between **Project 1 (PCA)** and **Project 2 (ARM)**.
   - **Upload Data**:
     - For Project 1: Upload `Non-Transactional_Dataset.csv`.
     - For Project 2: Upload `Transactional_Dataset.csv`.
   - **Adjust Parameters** (Project 2 only): Use sliders to tweak Support and Confidence.
   - **Run Analysis**: Click the "Run" button to execute cleaning and mining algorithms.

## 📂 Project Structure

```
Data-Mining/
├── data_mining_logic.py        # Core logic for data cleaning, PCA, and ARM
├── streamlit_app.py            # Main Streamlit GUI entry point
├── Non-Transactional_Dataset.csv # Dataset for Project 1
├── Transactional_Dataset.csv     # Dataset for Project 2
└── README.md                   # Project Documentation
```

## 📝 Technologies Used
- **Frontend**: [Streamlit](https://streamlit.io/)
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (PCA, Preprocessing)
- **Data Mining**: Mlxtend (Apriori, Association Rules)
- **Visualization**: Matplotlib