import pandas as pd

# ==========================================
# 1. LOAD DATASET
# ==========================================

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("Original dataset size:")
print(df.shape)


# ==========================================
# 2. CHECK MISSING VALUES
# ==========================================

print("\nMissing values:")
print(df.isnull().sum())


# ==========================================
# 3. CHECK DUPLICATES
# ==========================================

print("\nDuplicate rows before cleaning:")
print(df.duplicated().sum())


# ==========================================
# 4. REMOVE DUPLICATES
# ==========================================

df = df.drop_duplicates()

print("\nDataset size after removing duplicates:")
print(df.shape)


# ==========================================
# 5. CHECK CLASS DISTRIBUTION
# ==========================================

print("\nClass distribution:")
print(df["label"].value_counts())


# ==========================================
# 6. DISPLAY FIRST 5 ROWS
# ==========================================

print("\nFirst 5 rows after cleaning:")
print(df.head())