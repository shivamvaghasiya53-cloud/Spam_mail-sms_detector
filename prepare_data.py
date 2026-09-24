import pandas as pd
import os
import re
from sklearn.model_selection import train_test_split

# ============================================================
# SETTINGS
# ============================================================

DATASET_DIR = "dataset"

SMS_FILE = os.path.join(DATASET_DIR, "SMSSpamCollection")
EMAIL_FILE = os.path.join(DATASET_DIR, "email_text.csv")
TREC_FILE = os.path.join(DATASET_DIR, "processed_data.csv")

OUTPUT_TRAIN = os.path.join(DATASET_DIR, "train.csv")
OUTPUT_VALIDATION = os.path.join(DATASET_DIR, "validation.csv")
OUTPUT_TEST = os.path.join(DATASET_DIR, "test.csv")


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean email/SMS text before training.
    """

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove non-ASCII characters
    text = text.encode("ascii", "ignore").decode("ascii")

    # Remove excessive whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# 1. LOAD SMS DATASET
# ============================================================

print("\n========================================")
print("1. Loading SMS Spam Collection")
print("========================================")

sms = pd.read_csv(
    SMS_FILE,
    sep="\t",
    header=None,
    names=["label", "text"],
    encoding="latin-1"
)

# Convert labels
sms["label"] = sms["label"].map({
    "ham": 0,
    "spam": 1
})

sms = sms[["text", "label"]]

print("SMS records:", len(sms))


# ============================================================
# 2. LOAD YOUR EXISTING EMAIL DATASET
# ============================================================

print("\n========================================")
print("2. Loading existing email_text.csv")
print("========================================")

email = pd.read_csv(EMAIL_FILE)

print("Columns:", email.columns.tolist())
print("Records:", len(email))

# Make sure required columns exist
if "label" not in email.columns or "text" not in email.columns:
    raise ValueError(
        "email_text.csv must contain 'label' and 'text' columns."
    )

email = email[["text", "label"]].copy()

# Convert labels if they are text
if email["label"].dtype == "object":
    email["label"] = (
        email["label"]
        .astype(str)
        .str.lower()
        .map({
            "ham": 0,
            "not spam": 0,
            "spam": 1
        })
    )

email["label"] = pd.to_numeric(
    email["label"],
    errors="coerce"
)

email = email.dropna(subset=["label"])

email["label"] = email["label"].astype(int)

print("Existing email records:", len(email))


# ============================================================
# 3. LOAD NEW TREC EMAIL DATASET
# ============================================================

print("\n========================================")
print("3. Loading processed_data.csv")
print("========================================")

trec = pd.read_csv(
    TREC_FILE,
    low_memory=False
)

print("Columns:", trec.columns.tolist())
print("Records:", len(trec))


# ============================================================
# CREATE EMAIL TEXT
# ============================================================

# Subject + message are the main information used by the model.

trec["subject"] = trec["subject"].fillna("")
trec["message"] = trec["message"].fillna("")

trec["text"] = (
    "Subject: "
    + trec["subject"].astype(str)
    + "\n"
    + trec["message"].astype(str)
)


# Keep only required columns
trec = trec[["text", "label"]].copy()

trec["label"] = pd.to_numeric(
    trec["label"],
    errors="coerce"
)

trec = trec.dropna(subset=["label"])

trec["label"] = trec["label"].astype(int)

print("TREC usable records:", len(trec))


# ============================================================
# 4. COMBINE ALL DATASETS
# ============================================================

print("\n========================================")
print("4. Combining datasets")
print("========================================")

df = pd.concat(
    [sms, email, trec],
    ignore_index=True
)

print("Total records before cleaning:", len(df))


# ============================================================
# 5. CLEAN TEXT
# ============================================================

print("\nCleaning text...")

df["text"] = df["text"].apply(clean_text)

# Remove empty messages
df = df[df["text"].str.len() > 0]

# Make sure labels are only 0 or 1
df = df[df["label"].isin([0, 1])]


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

before_duplicates = len(df)

df = df.drop_duplicates(
    subset=["text", "label"]
)

after_duplicates = len(df)

print(
    "Duplicates removed:",
    before_duplicates - after_duplicates
)


# ============================================================
# 7. CHECK CLASS DISTRIBUTION
# ============================================================

print("\n========================================")
print("CLASS DISTRIBUTION")
print("========================================")

print(
    df["label"]
    .value_counts()
    .sort_index()
)

print("\n0 = NOT SPAM")
print("1 = SPAM")


# ============================================================
# 8. SHUFFLE DATA
# ============================================================

df = df.sample(
    frac=1,
    random_state=42
).reset_index(drop=True)


# ============================================================
# 9. TRAIN / VALIDATION / TEST SPLIT
# ============================================================

print("\n========================================")
print("Creating train / validation / test")
print("========================================")

# First:
# 80% training
# 20% temporary

train_df, temp_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

# Split remaining 20% into:
# 10% validation
# 10% test

validation_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    random_state=42,
    stratify=temp_df["label"]
)


# ============================================================
# 10. SAVE DATASETS
# ============================================================

train_df.to_csv(
    OUTPUT_TRAIN,
    index=False,
    encoding="utf-8"
)

validation_df.to_csv(
    OUTPUT_VALIDATION,
    index=False,
    encoding="utf-8"
)

test_df.to_csv(
    OUTPUT_TEST,
    index=False,
    encoding="utf-8"
)


# ============================================================
# 11. FINAL INFORMATION
# ============================================================

print("\n========================================")
print("DATASET PREPARATION COMPLETE")
print("========================================")

print("\nTotal:", len(df))

print("\nTraining:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))

print("\nTraining distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation distribution:")
print(validation_df["label"].value_counts().sort_index())

print("\nTesting distribution:")
print(test_df["label"].value_counts().sort_index())

print("\nFiles created:")
print(OUTPUT_TRAIN)
print(OUTPUT_VALIDATION)
print(OUTPUT_TEST)

print("\n========================================")
print("READY FOR TENSORFLOW TRAINING")
print("========================================")