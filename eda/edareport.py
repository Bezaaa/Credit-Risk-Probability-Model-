import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno

# Set visual style
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)

# Load data
df = pd.read_csv("../data/processed/high_risk_labels.csv", encoding="latin1")

# 1. Basic Info
print("📌 Data Overview:")
print(df.head(), "\n")
print("🔎 Data Types and Nulls:")
print(df.info(), "\n")
print("📊 Summary Statistics:")
print(df.describe(include="all"), "\n")

# 2. Missing Values
print("❓ Missing Values:")
print(df.isnull().sum())

msno.matrix(df)
plt.title("Missing Value Matrix")
plt.show()

# 3. Target Distribution
sns.countplot(x="is_high_risk", data=df, palette="Set2")
plt.title("Target Variable Distribution (is_high_risk)")
plt.xlabel("High Risk (1 = Yes, 0 = No)")
plt.ylabel("Count")
plt.show()

numeric_cols = df.select_dtypes(include=["number"]).copy()

# Step 2: Ensure 'is_high_risk' is included and exists
if "is_high_risk" in df.columns and "is_high_risk" not in numeric_cols.columns:
    numeric_cols["is_high_risk"] = df["is_high_risk"]

# Step 3: Plot safely if we have more than 1 numeric column
if numeric_cols.shape[1] > 1:
    sns.pairplot(numeric_cols, hue="is_high_risk", diag_kind="kde")
    plt.suptitle("Pairplot of Numeric Features by Risk Label", y=1.02)
    plt.show()
else:
    print("Not enough numeric columns found for pairplot.")



# 6. Boxplots for distribution vs risk
for col in numeric_cols.columns:
    sns.boxplot(data=df, x="is_high_risk", y=col, palette="Set3")
    plt.title(f"{col} vs Risk Label")
    plt.show()

# 7. Frequency of categorical columns (if any)
cat_cols = df.select_dtypes(include=["object"]).columns
for col in cat_cols:
    print(f"Value counts for '{col}':")
    print(df[col].value_counts(), "\n")

print("✅ EDA complete.")
