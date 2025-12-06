import os
from typing import Tuple, List, Dict
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, KBinsDiscretizer


def load_data(path: str) -> pd.DataFrame:
    """Load Excel or CSV (auto-detect)."""
    if path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # strip column names
    df.columns = [c.strip() for c in df.columns]
    return df


def clean_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Basic cleaning: drop empty rows/cols, dedupe, fill NAs and detect numeric/categorical cols."""
    df = df.copy()
    df.dropna(how="all", inplace=True)
    df.dropna(axis=1, how="all", inplace=True)
    df.drop_duplicates(inplace=True)

    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    # If attendance_pct is object -> convert
    if (
        "Average attendance on class" in df.columns
        and "attendance_pct" not in df.columns
    ):
        # handled by user mapping previously — keep simple here

        pass

    # if there is column already renamed to attendance_pct and in cat_cols
    if "attendance_pct" in cat_cols:
        cat_cols.remove("attendance_pct")
        df["attendance_pct"] = pd.to_numeric(df["attendance_pct"], errors="coerce")
        num_cols.append("attendance_pct")

    # fill numeric with median
    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())

    # fill categorical with mode
    for col in cat_cols:
        if df[col].isnull().any():
            df[col] = df[col].fillna(
                df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
            )

    # Optionally rename known columns (keeps user's mapping)
    col_renames = {
        "University Admission year": "admission_year",
        "Gender": "gender",
        "Age": "age",
        "H.S.C passing year": "hsc_year",
        "Program": "program",
        "Current Semester": "current_semester",
        "Do you have meritorious scholarship ?": "merit_scholarship",
        "Do you use University transportation?": "use_transport",
        "How many hour do you study daily?": "study_hours_daily",
        "How many times do you seat for study in a day?": "study_freq_daily",
        "What is your preferable learning mode?": "learning_mode",
        "Do you use smart phone?": "use_smartphone",
        "Do you have personal Computer?": "have_pc",
        "How many hour do you spent daily in social media?": "social_media_hours",
        "Status of your English language proficiency": "english_proficiency",
        "Average attendance on class": "attendance_pct",
        "Did you ever fall in probation?": "probation",
        "Did you ever got suspension?": "suspension",
        "Do you attend in teacher consultancy for any kind of academical problems?": "teacher_consultancy",
        "What are the skills do you have ?": "skills",
        "How many hour do you spent daily on your skill development?": "skill_hours_daily",
        "What is you interested area?": "interest_area",
        "What is your relationship status?": "relationship_status",
        "Are you engaged with any co-curriculum activities?": "co_curriculum",
        "With whom you are living with?": "living_with",
        "Do you have any health issues?": "health_issues",
        "What was your previous SGPA?": "prev_sgpa",
        "Do you have any physical disabilities?": "physical_disability",
        "What is your current CGPA?": "current_cgpa",
        "How many Credit did you have completed?": "credits_completed",
        "What is your monthly family income?": "monthly_income",
    }
    df = df.rename(columns={k: v for k, v in col_renames.items() if k in df.columns})

    # refresh detected columns
    num_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    return df, num_cols, cat_cols


def prepare_cluster_and_bayes(df: pd.DataFrame, num_cols: list, cat_cols: list):
    """
    Returns:
    - X_cluster_df: dataframe used for clustering (encoded)
    - X_scaled: ndarray scaled for clustering
    - encoder_scaler_dict: dict with encoder/scaler/kbins (for later reuse)
    - X_bayes: dataframe discretized for bayes net
    """
    df_copy = df.copy()

    # --- Ensure numeric columns are numeric and fill NA with median ---
    for col in num_cols:
        # coerce to numeric, fill any introduced NaNs with median (safe fallback)
        df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
        if df_copy[col].isnull().any():
            median = df_copy[col].median()
            df_copy[col] = df_copy[col].fillna(median)

    # --- Ensure categorical columns are strings and fill NA with "Unknown" ---
    for col in cat_cols:
        df_copy[col] = df_copy[col].fillna("Unknown").astype(str)

    # Ordinal encoding of categorical cols (now all strings)
    ordinal_encoder = OrdinalEncoder()
    if len(cat_cols) > 0:
        df_copy[cat_cols] = ordinal_encoder.fit_transform(df_copy[cat_cols])

    # Keep encoded but unscaled dataframe for inspection/clustering input
    X_cluster_df = df_copy.copy()

    # Scaling for clustering
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster_df.values)

    # Prepare bayes dataset: start from original df but coerce types same way
    X_bayes = df.copy()
    # numeric columns -> numeric
    for col in num_cols:
        X_bayes[col] = pd.to_numeric(X_bayes[col], errors="coerce")
        if X_bayes[col].isnull().any():
            X_bayes[col] = X_bayes[col].fillna(df_copy[col].median())

    # categorical columns -> string then ordinal-transform using the fitted encoder
    if len(cat_cols) > 0:
        for col in cat_cols:
            X_bayes[col] = X_bayes[col].fillna("Unknown").astype(str)
        X_bayes[cat_cols] = ordinal_encoder.transform(X_bayes[cat_cols])

    # discretize numeric columns for Bayes (if any)
    if len(num_cols) > 0:
        kb = KBinsDiscretizer(n_bins=4, encode="ordinal", strategy="quantile")
        X_bayes[num_cols] = kb.fit_transform(X_bayes[num_cols])
    else:
        kb = None

    encoder_scaler_dict = {
        "ordinal_encoder": ordinal_encoder,
        "scaler": scaler,
        "kbins": kb,
        "num_cols": num_cols,
        "cat_cols": cat_cols,
    }

    return X_cluster_df, X_scaled, encoder_scaler_dict, X_bayes
