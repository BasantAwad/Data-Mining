
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.decomposition import PCA
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

class DataMiningLogic:
    @staticmethod
    def audit_project1_data(df):
        audit_results = {}
        
        # 1. Missing Values
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns] # Temp standardize for audit
        audit_results['Missing Values (Total)'] = df.isna().sum().sum()
        if 'height_cm' in df.columns:
            # Count NULL strings as missing
            null_str_count = (df['height_cm'].astype(str).str.upper() == 'NULL').sum()
            missing_height = df['height_cm'].isna().sum() + null_str_count
            audit_results['Missing Height Values'] = int(missing_height)
            
        # 2. Negative/Unrealistic Numerics
        unrealistic_count = 0
        if "age" in df.columns:
            unrealistic_count += ((df["age"] <= 0) | (df["age"] > 120)).sum()
        if "income" in df.columns:
            unrealistic_count += ((df["income"] < 0) | (df["income"] > 1e7)).sum()
        if "height_cm" in df.columns:
            height_numeric = pd.to_numeric(df["height_cm"].replace("NULL", np.nan), errors='coerce')
            unrealistic_count += ((height_numeric < 50) | (height_numeric > 250)).sum()
        if "score" in df.columns:
            unrealistic_count += ((df["score"] < 0) | (df["score"] > 1000)).sum()
        audit_results['Unrealistic/Negative Numerics'] = int(unrealistic_count)

        # 3. Extreme Outliers (IQR)
        outlier_count = 0
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].dropna().empty: continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        audit_results['Extreme Outliers (Total)'] = int(outlier_count)

        # 4. Incorrect Category Labels (Gender)
        if "gender" in df.columns:
            normalized = df["gender"].astype(str).str.lower().str.strip()
            mapped = normalized.replace({
                "m": "male", "male": "male", "mail": "male",
                "f": "female", "female": "female", "fm": "female", "fem": "female", "fmale": "female"
            })
            # Count values that didn't map to male/female
            bad_labels = (~mapped.isin(["male", "female"])).sum()
            audit_results['Incorrect Gender Labels'] = int(bad_labels)
            
            # Count typos specifically (mail, fm, fem, fmale)
            typo_count = normalized.isin(["mail", "fm", "fem", "fmale"]).sum()
            audit_results['Gender Typos (mail, fmale, etc.)'] = int(typo_count)

        # 5. Inconsistent Capitalization
        inconsistent_cap_count = 0
        if "gender" in df.columns:
            gender_inconsistent = df["gender"].dropna().apply(
                lambda x: str(x) not in ["Male", "Female", "male", "female"]
            ).sum()
            inconsistent_cap_count += gender_inconsistent
            
        if "education" in df.columns:
            edu_vals = df["education"].astype(str).str.strip()
            # Check for lowercase variants
            edu_inconsistent = edu_vals.apply(
                lambda x: x.lower() == x and x not in ["nan", ""]
            ).sum()
            inconsistent_cap_count += edu_inconsistent
            
        if "city" in df.columns:
            city_vals = df["city"].astype(str).str.strip()
            # Check for lowercase variants
            city_inconsistent = city_vals.apply(
                lambda x: x.lower() == x and x not in ["nan", ""]
            ).sum()
            inconsistent_cap_count += city_inconsistent
            
        audit_results['Inconsistent Capitalization'] = int(inconsistent_cap_count)
        
        # 6. Education Typos
        if "education" in df.columns:
            edu_vals = df["education"].astype(str).str.strip().str.title()
            edu_typos = edu_vals.isin(["Bachlors"]).sum()
            audit_results['Education Typos (Bachlors)'] = int(edu_typos)
            
        # 7. City Abbreviations
        if "city" in df.columns:
            city_vals = df["city"].astype(str).str.strip().str.title()
            city_abbrev = city_vals.isin(["Alex"]).sum()
            audit_results['City Abbreviations (Alex)'] = int(city_abbrev)

        return audit_results

    @staticmethod
    def audit_project2_data(df):
        audit_results = {}
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # 1. Missing Dates
        if "date" in df.columns:
            missing_dates = df["date"].replace(["NULL", "null", ""], np.nan).isna().sum()
            audit_results['Missing Dates'] = int(missing_dates)
        
        # 2. Missing Amounts
        if "amount" in df.columns:
            missing_amounts = df["amount"].isna().sum()
            audit_results['Missing Amounts'] = int(missing_amounts)

        # 3. Duplicated Transactions
        audit_results['Duplicated Transactions'] = int(df.duplicated().sum())

        # 4. Inconsistent Item Separators (';' vs ',')
        if "items_purchased" in df.columns:
            semi_count = df["items_purchased"].astype(str).apply(lambda x: ';' in x).sum()
            audit_results['Inconsistent Separators (;)'] = int(semi_count)
            
        # 5. Blank Items in List (e.g. "item1,,item2")
        if "items_purchased" in df.columns:
            blank_count = df["items_purchased"].astype(str).apply(
                lambda x: ',,' in x or ';;' in x or x.strip() == ''
            ).sum()
            audit_results['Blank Items in List'] = int(blank_count)
            
        # 6. Different Capitalizations in Items
        if "items_purchased" in df.columns:
            def has_inconsistent_caps(items_str):
                if pd.isna(items_str):
                    return False
                items_str = str(items_str).replace(";", ",")
                items = [i.strip() for i in items_str.split(",") if i.strip()]
                # Check if any item has uppercase letters (should be lowercase)
                return any(item != item.lower() for item in items)
            
            inconsistent_caps = df["items_purchased"].apply(has_inconsistent_caps).sum()
            audit_results['Inconsistent Capitalizations'] = int(inconsistent_caps)

        # 7. Inconsistent Date Formats
        if "date" in df.columns:
            date_strs = df["date"].astype(str).replace(["NULL", "null", "nan", ""], np.nan).dropna()
            # Check for DD/MM/YYYY format (vs YYYY-MM-DD)
            slash_format = date_strs.apply(lambda x: '/' in str(x)).sum()
            audit_results['Inconsistent Date Formats (/)'] = int(slash_format)
            
        return audit_results

    @staticmethod
    def clean_project1_data(df):
        # STANDARDIZE COLUMN NAMES
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        
        # REMOVE DUPLICATES
        df = df.drop_duplicates()
        
        # 1. FIX GENDER - Incorrect labels & inconsistent capitalization
        if "gender" in df.columns:
            df["gender"] = df["gender"].astype(str).str.lower().str.strip()
            df["gender"] = df["gender"].replace({
                "m": "Male", "male": "Male", "mail": "Male",
                "f": "Female", "female": "Female", "fm": "Female", "fem": "Female", "fmale": "Female"
            })
            # Standardize to Title case
            df["gender"] = df["gender"].str.title()
            
        # 2. FIX EDUCATION - Typos & inconsistent capitalization
        if "education" in df.columns:
            df["education"] = df["education"].astype(str).str.strip().str.title()
            education_corrections = {
                "Bachlors": "Bachelor",
                "High School": "High School",
                "Bachelor": "Bachelor", 
                "Master": "Master",
                "Phd": "PhD"
            }
            df["education"] = df["education"].replace(education_corrections)
            
        # 3. FIX CITY - Abbreviations & inconsistent capitalization
        if "city" in df.columns:
            df["city"] = df["city"].astype(str).str.strip().str.title()
            city_corrections = {
                "Alex": "Alexandria",
                "Cairo": "Cairo",
                "Giza": "Giza",
                "Luxor": "Luxor",
                "Aswan": "Aswan",
                "Alexandria": "Alexandria"
            }
            df["city"] = df["city"].replace(city_corrections)
            
        # 4. FIX MARITAL STATUS - Inconsistent capitalization
        if "marital_status" in df.columns:
            df["marital_status"] = df["marital_status"].astype(str).str.strip().str.title()
            
        # 5. CLEAN UNREALISTIC/NEGATIVE NUMERICAL VALUES
        if "age" in df.columns:
            df.loc[(df["age"] <= 0) | (df["age"] > 120), "age"] = np.nan
        if "income" in df.columns:
            df.loc[(df["income"] < 0) | (df["income"] > 1e7), "income"] = np.nan
        if "height_cm" in df.columns:
            # Handle NULL strings
            df["height_cm"] = pd.to_numeric(df["height_cm"].replace("NULL", np.nan), errors='coerce')
            df.loc[(df["height_cm"] < 50) | (df["height_cm"] > 250), "height_cm"] = np.nan
        if "score" in df.columns:
            df.loc[(df["score"] < 0) | (df["score"] > 1000), "score"] = np.nan
        if "children" in df.columns:
            df["children"] = pd.to_numeric(df["children"], errors='coerce')
            df.loc[(df["children"] < 0) | (df["children"] > 20), "children"] = np.nan
        if "credit_score" in df.columns:
            df["credit_score"] = pd.to_numeric(df["credit_score"], errors='coerce')
            df.loc[(df["credit_score"] < 300) | (df["credit_score"] > 850), "credit_score"] = np.nan
            
        # 6. CAP EXTREME OUTLIERS USING IQR
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            if df[col].dropna().empty:
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            low = Q1 - 1.5 * IQR
            high = Q3 + 1.5 * IQR
            df[col] = np.clip(df[col], low, high)
            
        # 7. HANDLE MISSING VALUES - Imputation
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        
        # Re-get numeric cols after conversions
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) > 0:
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for c in cat_cols:
            df[c] = cat_imputer.fit_transform(df[[c]]).ravel()
        
        # Reset index to be sequential (0, 1, 2, 3, ...)
        df = df.reset_index(drop=True)
            
        return df

    @staticmethod
    def run_pca_analysis(df):
        # PREPARE DATA FOR PCA
        drop_cols = [c for c in ["id", "identifier", "transaction_id"] if c in df.columns]
        df_pca = df.drop(columns=drop_cols, errors="ignore")
        
        # ENCODE CATEGORICAL VARIABLES
        cat_cols = df_pca.select_dtypes(exclude=[np.number]).columns.tolist()
        if len(cat_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded = encoder.fit_transform(df_pca[cat_cols])
            encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(cat_cols))
            df_pca = pd.concat([df_pca.drop(columns=cat_cols), encoded_df], axis=1)
            
        num_cols = df_pca.select_dtypes(include=[np.number]).columns.tolist()
        X = df_pca[num_cols].astype(float).values
        
        results = {}
        
        # StandardScaler
        scaler_std = StandardScaler()
        X_std = scaler_std.fit_transform(X)
        pca_std = PCA(n_components=3)
        X_pca_std = pca_std.fit_transform(X_std)
        results['std'] = {'X_pca': X_pca_std, 'explained_variance': pca_std.explained_variance_ratio_}
        
        # MinMaxScaler
        scaler_mm = MinMaxScaler()
        X_mm = scaler_mm.fit_transform(X)
        pca_mm = PCA(n_components=3)
        X_pca_mm = pca_mm.fit_transform(X_mm)
        results['mm'] = {'X_pca': X_pca_mm, 'explained_variance': pca_mm.explained_variance_ratio_}
        
        return results

    @staticmethod
    def clean_project2_data(df):
        # STANDARDIZE COLUMN NAMES
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
        
        # 1. REMOVE DUPLICATED TRANSACTIONS
        df = df.drop_duplicates()
        
        # 2. DATE HANDLING - Fix inconsistent formats & missing values
        if "date" in df.columns:
            # Replace NULL/null strings with NaN
            df["date"] = df["date"].replace(["NULL", "null", ""], np.nan)
            # Handle mixed formats (YYYY-MM-DD, DD/MM/YYYY, etc.)
            df["date"] = pd.to_datetime(df["date"], format='mixed', errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=["date"])
                
        # 3. AMOUNT HANDLING - Fix missing values
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
            mean_amount = df["amount"].mean()
            df["amount"] = df["amount"].fillna(mean_amount)
            df["amount"] = df["amount"].round(2)
            
        # 4. CLEAN ITEMS PURCHASED IN-PLACE
        # Fixes: inconsistent separators, different capitalizations, blank items
        if "items_purchased" in df.columns:
            def clean_items_string(items):
                if pd.isna(items) or str(items).strip() == '':
                    return ''
                # Convert to string and normalize
                items_str = str(items)
                # Fix inconsistent separators (';' -> ',')
                items_str = items_str.replace(";", ",")
                # Split, clean each item, remove blanks
                items_list = [i.strip().lower() for i in items_str.split(",") if i.strip()]
                # Remove duplicates while preserving order
                items_list = list(dict.fromkeys(items_list))
                # Rejoin as comma-separated string
                return ", ".join(items_list)
            
            df["items_purchased"] = df["items_purchased"].apply(clean_items_string)
            # Remove rows with empty items
            df = df[df["items_purchased"] != '']
            
        # 5. STORE LOCATION HANDLING - Fix capitalizations
        if "store_location" in df.columns:
            df["store_location"] = df["store_location"].astype(str).str.strip().str.title()
            location_corrections = {
                'Cai': 'Cairo', 'Cairo City': 'Cairo',
                'Ale': 'Alexandria', 'Alexandria City': 'Alexandria',
                'Lux': 'Luxor', 'Luxor City': 'Luxor',
                'Giz': 'Giza', 'Giza City': 'Giza',
                'Man': 'Mansoura', 'Mansoura City': 'Mansoura'
            }
            df["store_location"] = df["store_location"].replace(location_corrections)
            
        # 6. DISCOUNT APPLIED HANDLING
        if "discount_applied" in df.columns:
            df["discount_applied"] = df["discount_applied"].astype(str).str.strip().str.upper()
            discount_map = {
                'YES': True, 'Y': True, 'TRUE': True, '1': True, '1.0': True,
                'NO': False, 'N': False, 'FALSE': False, '0': False, '0.0': False, 'NAN': False
            }
            df["discount_applied"] = df["discount_applied"].map(discount_map)
            
            # Fill missing with mode
            if not df["discount_applied"].dropna().empty:
                mode_val = df["discount_applied"].mode()[0]
                df["discount_applied"] = df["discount_applied"].fillna(mode_val)
            else:
                df["discount_applied"] = df["discount_applied"].fillna(False)
            
        # 7. GENERIC HANDLING FOR REMAINING COLUMNS
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = [c for c in df.select_dtypes(include=["object"]).columns if c != "items_purchased"]
        
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        
        if len(num_cols) > 0:
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
            
        for c in cat_cols:
            df[c] = cat_imputer.fit_transform(df[[c]]).ravel()
        
        # Reset index to be sequential (0, 1, 2, 3, ...)
        df = df.reset_index(drop=True)
            
        return df

    @staticmethod
    def run_association_rules(df, min_support=0.01, min_threshold=0.01):
        # Convert items_purchased string to list for transaction encoding
        def parse_items(items_str):
            if pd.isna(items_str) or str(items_str).strip() == '':
                return []
            return [i.strip() for i in str(items_str).split(",") if i.strip()]
        
        transactions = df["items_purchased"].apply(parse_items).tolist()
        
        # Filter out empty transactions
        transactions = [t for t in transactions if len(t) > 0]
        
        if len(transactions) == 0:
            return None, None
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True, max_len=3)
        
        if frequent_itemsets.empty:
            return None, None
            
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
        return frequent_itemsets, rules
