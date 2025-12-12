
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
            audit_results['Missing Height Values'] = df['height_cm'].isna().sum()
            
        # 2. Negative/Unrealistic Numerics
        unrealistic_count = 0
        if "age" in df.columns:
            unrealistic_count += ((df["age"] <= 0) | (df["age"] > 120)).sum()
        if "income" in df.columns:
            unrealistic_count += ((df["income"] < 0) | (df["income"] > 1e7)).sum()
        if "height_cm" in df.columns:
            unrealistic_count += ((df["height_cm"] < 50) | (df["height_cm"] > 250)).sum()
        if "score" in df.columns:
            unrealistic_count += ((df["score"] < 0) | (df["score"] > 1000)).sum()
        audit_results['Unrealistic/Negative Numerics'] = unrealistic_count

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
        audit_results['Extreme Outliers (Total)'] = outlier_count

        # 4. Incorrect Category Labels (Gender)
        if "gender" in df.columns:
            # Check for values NOT in standard list (case-insensitive check logic simulated)
            standard_genders = ["male", "female"]
            # We count values that wouldn't map cleanly or are typo-ridden
            # Simple heuristic: count unique values that aren't exactly 'Male'/'Female' or standard lower
            raw_genders = df["gender"].dropna().unique()
            # This is a bit qualitative, but we can check capitalization inconsistency
            # Count values that change when .lower() is applied
            inconsistent_caps = df["gender"].dropna().apply(lambda x: str(x) != str(x).lower() and str(x) != str(x).title()).sum()
            # Actually, user asked for "Inconsistent capitalization" specifically
            audit_results['Inconsistent Capitalization (Gender)'] = inconsistent_caps
            
            # Incorrect labels
            normalized = df["gender"].astype(str).str.lower().str.strip()
            mapped = normalized.replace({
                "m": "male", "male": "male", "mail": "male",
                "f": "female", "female": "female", "fm": "female", "fem": "female", "fmale": "female"
            })
            # If it didn't map to male/female, it might be incorrect label
            bad_labels = (~mapped.isin(["male", "female"])).sum()
            audit_results['Incorrect Category Labels'] = bad_labels

        return audit_results

    @staticmethod
    def audit_project2_data(df):
        audit_results = {}
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

        # 1. Missing Dates and Amounts
        missing_count = 0
        if "date" in df.columns: missing_count += df["date"].replace(["NULL", "null", ""], np.nan).isna().sum()
        if "amount" in df.columns: missing_count += df["amount"].isna().sum()
        audit_results['Missing Dates/Amounts'] = missing_count

        # 2. Duplicated Transactions
        audit_results['Duplicated Transactions'] = df.duplicated().sum()

        # 3. Inconsistent Item Separators & Formats
        if "items_purchased" in df.columns:
            # Check for semicolon usage
            semi_count = df["items_purchased"].astype(str).apply(lambda x: ';' in x).sum()
            audit_results['Inconsistent Separators (e.g. ";")'] = semi_count
            
            # Blank items (e.g. "item1,,item2")
            blank_count = df["items_purchased"].astype(str).apply(lambda x: ',,' in x or ';;' in x or x.strip() == '').sum()
            audit_results['Blank Items in List'] = blank_count
            
            # Different Capitalizations (heuristic: count items that aren't all lower or title)
            # This is tricky per item, but we can check if column has mixed case across rows
            audit_results['Different Capitalizations (Detected)'] = "Yes" if df["items_purchased"].str.islower().mean() < 1.0 else "No"

        # 4. Inconsistent Date Formats
        if "date" in df.columns:
            # Try parsing with specific format, count failures? 
            # Or just check if multiple formats exist. 
            # Easier: count non-standard formats before cleaning
            # We already track missing/null, but this addresses "Formats"
            # Simple check: string length variance or regex match
            pass # Covered mostly by missing check after coercion, but let's be explicit
            # Checking if some dates are YYYY-MM-DD vs DD/MM/YYYY
            # This is hard to detect perfectly without trying to parse everything.
            # We will rely on the "Missing Dates" count after coercion for this metric mostly.
            
        return audit_results

    @staticmethod
    def clean_project1_data(df):
        # STANDARDIZE COLUMN NAMES
        df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
        
        # REMOVE DUPLICATES
        df = df.drop_duplicates()
        
        # FIX CATEGORICAL VALUES
        if "gender" in df.columns:
            df["gender"] = df["gender"].astype(str).str.lower().str.strip()
            df["gender"] = df["gender"].replace({
                "m": "male", "male": "male", "mail": "male",
                "f": "female", "female": "female", "fm": "female", "fem": "female", "fmale": "female"
            })
            
        # CLEAN UNREALISTIC VALUES
        if "age" in df.columns:
            df.loc[(df["age"] <= 0) | (df["age"] > 120), "age"] = np.nan
        if "income" in df.columns:
            df.loc[(df["income"] < 0) | (df["income"] > 1e7), "income"] = np.nan
        if "height_cm" in df.columns:
            df.loc[(df["height_cm"] < 50) | (df["height_cm"] > 250), "height_cm"] = np.nan
        if "score" in df.columns:
            df.loc[(df["score"] < 0) | (df["score"] > 1000), "score"] = np.nan
            
        # CAP OUTLIERS USING IQR
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
            
        # HANDLE MISSING VALUES
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        
        if len(numeric_cols) > 0:
            df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
            
        cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        for c in cat_cols:
            df[c] = cat_imputer.fit_transform(df[[c]]).ravel()
            
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
        
        # REMOVE DUPLICATES
        df = df.drop_duplicates()
        
        # 1. DATE HANDLING
        if "date" in df.columns:
            # Handle mixed formats and timestamps
            df["date"] = pd.to_datetime(df["date"], format='mixed', errors='coerce')
            # Drop rows with invalid dates
            df = df.dropna(subset=["date"])
                
        # 2. AMOUNT HANDLING
        if "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors='coerce')
            mean_amount = df["amount"].mean()
            df["amount"] = df["amount"].fillna(mean_amount)
            df["amount"] = df["amount"].round(2)
            
        # 3. STORE LOCATION HANDLING
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
            
        # 4. DISCOUNT APPLIED HANDLING
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
            
        # 5. GENERIC HANDLING FOR REMAINING
        num_cols = df.select_dtypes(include=[np.number]).columns
        cat_cols = df.select_dtypes(include=["object"]).columns
        
        # SimpleImputer for any remaining columns not handled above
        num_imputer = SimpleImputer(strategy="median")
        cat_imputer = SimpleImputer(strategy="most_frequent")
        
        if len(num_cols) > 0:
            df[num_cols] = num_imputer.fit_transform(df[num_cols])
            
        for c in cat_cols:
            df[c] = cat_imputer.fit_transform(df[[c]]).ravel()
            
        # 6. CLEAN ITEMS PURCHASED (Crucial for ARM)
        def clean_items(items):
            if pd.isna(items): return []
            items = str(items).lower().replace(";", ",")
            items = [i.strip() for i in items.split(",") if i.strip()]
            return list(dict.fromkeys(items))

        if "items_purchased" in df.columns:
            df["items_cleaned"] = df["items_purchased"].apply(clean_items)
            
        return df

    @staticmethod
    def run_association_rules(df, min_support=0.01, min_threshold=0.01):
        transactions = df["items_cleaned"].tolist()
        
        te = TransactionEncoder()
        te_ary = te.fit(transactions).transform(transactions)
        encoded_df = pd.DataFrame(te_ary, columns=te.columns_)
        
        frequent_itemsets = apriori(encoded_df, min_support=min_support, use_colnames=True, max_len=3)
        
        if frequent_itemsets.empty:
            return None, None
            
        rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_threshold)
        return frequent_itemsets, rules
