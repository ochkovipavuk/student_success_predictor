import pandas as pd

file_path = "student-mat.csv"  # student-por.csv
df = pd.read_csv(file_path, sep=";")

df.columns = [c.strip() for c in df.columns]

print(df.head())
print(df.info())

df.dropna(how="all", inplace=True)
df.drop_duplicates(inplace=True)

target = "G3"

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

if target in num_cols:
    num_cols.remove(target)

for col in num_cols:
    df[col] = df[col].fillna(df[col].median())

for col in cat_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

df["avg_grade"] = (df["G1"] + df["G2"]) / 2
df["grade_trend"] = df["G2"] - df["G1"]

df["alc_total"] = df["Dalc"] + df["Walc"]

df["study_load"] = df["studytime"] * df["failures"]

binary_map = {
    "yes": 1, "no": 0,
    "GP": 1, "MS": 0,
    "F": 1, "M": 0,
    "U": 1, "R": 0,
    "LE3": 0, "GT3": 1,
    "T": 1, "A": 0
}

for col in cat_cols:
    df[col] = df[col].map(binary_map).fillna(df[col])

num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

if target in num_cols:
    num_cols.remove(target)

drop_for_cluster = ["G1", "G2", "G3", "avg_grade"]

X_cluster = df.drop(columns=drop_for_cluster).copy()

num_cols = X_cluster.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_cluster.select_dtypes(include=["object"]).columns.tolist()

from sklearn.preprocessing import OrdinalEncoder, StandardScaler

ordinal_encoder = OrdinalEncoder()

if len(cat_cols) > 0:
    X_cluster[cat_cols] = ordinal_encoder.fit_transform(X_cluster[cat_cols])

scaler = StandardScaler()
X_cluster_scaled = scaler.fit_transform(X_cluster)

from sklearn.preprocessing import KBinsDiscretizer

X_bayes = X_cluster.copy()
