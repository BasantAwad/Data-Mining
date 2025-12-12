
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from data_mining_logic import DataMiningLogic

st.set_page_config(page_title="Data Mining Project", layout="wide")

# Custom CSS for better aesthetics
st.markdown("""
    <style>
        .main {
            background-color: #f5f5f5;
        }
        .stButton>button {
            width: 100%;
            background-color: #4CAF50;
            color: white;
            padding: 10px 24px;
            border-radius: 8px;
            border: none;
        }
        h1 {
            color: #2E86C1;
            text-align: center;
            padding-bottom: 20px;
        }
        h2 {
            color: #2874A6;
            border-bottom: 2px solid #2874A6;
            padding-bottom: 10px;
        }
        .metric-container {
            background-color: white;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

st.title("📊 Data Mining Project GUI")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Select Project", ["Project 1: PCA", "Project 2: ARM"])

if page == "Project 1: PCA":
    st.header("Project 1: Data Preprocessing & PCA")
    
    with st.container():
        uploaded_file = st.file_uploader("Upload 'Non-Transactional_Dataset.csv'", type="csv")
    
    if uploaded_file is not None:
        try:
            # 1. RAW DATA & AUDIT
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            # --- AUDIT REPORT ---
            st.subheader("🔍 Data Quality Audit Report")
            audit_results = DataMiningLogic.audit_project1_data(df.copy())
            
            # Display Audit Metrics
            cols = st.columns(3)
            col_idx = 0
            for check, value in audit_results.items():
                with cols[col_idx % 3]:
                    st.metric(label=check, value=value, delta_color="inverse")
                col_idx += 1
            
            with st.expander("View Raw Data Preview", expanded=False):
                st.dataframe(df.head())
                st.write(f"**Shape:** {df.shape}")

            st.write("---")

            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                run_btn = st.button("🚀 Run Cleaning, Preprocessing & PCA")

            if run_btn:
                with st.spinner("Processing Data..."):
                    # 2. CLEAN DATA
                    cleaned_df = DataMiningLogic.clean_project1_data(df.copy())
                    
                    st.subheader("✨ Cleaned Data Preview")
                    with st.expander("View Cleaned Data Details", expanded=True):
                        st.dataframe(cleaned_df.head())
                        st.write(f"**Shape:** {cleaned_df.shape}")
                        
                        # Verify cleaning (Simple Check)
                        remaining_nulls = cleaned_df.isna().sum().sum()
                        if remaining_nulls == 0:
                            st.success("✔ Data successfully cleaned: No missing values remaining.")
                        else:
                            st.warning(f"⚠ Warning: {remaining_nulls} missing values remaining.")
                    
                    # 3. PCA ANALYSIS
                    st.write("---")
                    st.header("📉 PCA Analysis Results")
                    results = DataMiningLogic.run_pca_analysis(cleaned_df)
                    
                    # Layout for PCA
                    pc_col1, pc_col2 = st.columns([1, 2])
                    
                    with pc_col1:
                        st.markdown("### Variance Ratio")
                        st.markdown("**(StandardScaler)**")
                        total_var = 0
                        for i, v in enumerate(results['std']['explained_variance'], start=1):
                            st.write(f"**PC{i}:** `{v:.4f}`")
                            total_var += v
                        st.metric("Total Explained Variance", f"{total_var:.4f}")
                    
                    with pc_col2:
                        X_pca = results['std']['X_pca']
                        
                        fig, ax = plt.subplots(figsize=(8, 5))
                        ax.scatter(X_pca[:,0], X_pca[:,1], alpha=0.7, c='#2E86C1')
                        ax.set_xlabel("Principal Component 1")
                        ax.set_ylabel("Principal Component 2")
                        ax.set_title("PCA 2D Projection")
                        ax.grid(True, linestyle='--', alpha=0.6)
                        st.pyplot(fig)
                        
                        if X_pca.shape[1] >= 3:
                            st.info("ℹ 3D components calculated but visualization is optimized for 2D in web view.")

        except Exception as e:
            st.error(f"Error processing file: {e}")

elif page == "Project 2: ARM":
    st.header("Project 2: Association Rule Mining")
    
    with st.container():
        uploaded_file = st.file_uploader("Upload 'Transactional_Dataset.csv'", type="csv")
    
    st.sidebar.markdown("### Mining Parameters")
    min_support = st.sidebar.slider("Min Support", 0.001, 0.5, 0.01, 0.001)
    min_threshold = st.sidebar.slider("Min Confidence Threshold", 0.001, 1.0, 0.01, 0.001)

    if uploaded_file is not None:
        try:
            # 1. RAW DATA & AUDIT
            df = pd.read_csv(uploaded_file)
            st.success("Dataset loaded successfully!")
            
            # --- AUDIT REPORT ---
            st.subheader("🔍 Data Quality Audit Report")
            audit_results = DataMiningLogic.audit_project2_data(df.copy())
            
            # Display Audit Metrics
            cols = st.columns(3)
            col_idx = 0
            for check, value in audit_results.items():
                with cols[col_idx % 3]:
                    val_str = str(value)
                    # Highlight issues in red if count > 0
                    if isinstance(value, int) and value > 0:
                        st.metric(label=check, value=val_str, delta="Issue Found", delta_color="inverse")
                    elif value == "Yes":
                         st.metric(label=check, value=val_str, delta="Issue Found", delta_color="inverse")
                    else:
                        st.metric(label=check, value=val_str)
                col_idx += 1
            
            with st.expander("View Raw Data Preview", expanded=False):
                st.dataframe(df.head())
            
            st.write("---")
            col1, col2, col3 = st.columns([1,2,1])
            with col2:
                run_btn = st.button("🚀 Run Cleaning & Rule Mining")

            if run_btn:
                with st.spinner("Mining Rules..."):
                    # 2. CLEAN DATA
                    cleaned_df = DataMiningLogic.clean_project2_data(df.copy())
                    
                    st.subheader("✨ Cleaned Data Preview")
                    with st.expander("View Cleaned Data Details", expanded=True):
                        st.dataframe(cleaned_df.head())
                        st.caption(f"Total Transactions processed: {len(cleaned_df)}")

                    # 3. ARM ANALYSIS
                    st.write("---")
                    st.header("🔗 Association Rules Results")
                    frequent_itemsets, rules = DataMiningLogic.run_association_rules(cleaned_df, min_support, min_threshold)
                    
                    if rules is None or rules.empty:
                        st.warning("No rules found. Try lowering the Support or Threshold.")
                    else:
                        m_col1, m_col2 = st.columns(2)
                        m_col1.metric("Frequent Itemsets", len(frequent_itemsets))
                        m_col2.metric("Association Rules", len(rules))
                        
                        st.markdown("### Top Rules (Sorted by Lift)")
                        top_rules = rules.sort_values("lift", ascending=False).head(10)
                        
                        # Format table
                        display_rules = top_rules[["antecedents", "consequents", "support", "confidence", "lift"]].copy()
                        display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: list(x))
                        display_rules["consequents"] = display_rules["consequents"].apply(lambda x: list(x))
                        
                        st.dataframe(display_rules.style.background_gradient(subset=['lift'], cmap='Blues'))
                        
                        # Plot
                        st.markdown("### Lift Visualization")
                        labels = [f"{list(r['antecedents'])} -> {list(r['consequents'])}" for _, r in top_rules.iterrows()]
                        
                        fig, ax = plt.subplots(figsize=(10, 6))
                        ax.barh(labels, top_rules["lift"], color='#2874A6')
                        ax.set_xlabel("Lift")
                        ax.set_title("Top 10 Rules by Lift")
                        ax.invert_yaxis()
                        st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing file: {e}")

