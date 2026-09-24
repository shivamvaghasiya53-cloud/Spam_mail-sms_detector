import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

TEST_FILE = "dataset/test.csv"
MODEL_FILE = "model/spam_model.keras"
GRAPH_FILE = "graphs/confusion_matrix.png"


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if pd.isna(text):
        return ""

    text = str(text)

    text = text.encode(
        "ascii",
        "ignore"
    ).decode("ascii")

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


# ============================================================
# LOAD TEST DATA
# ============================================================

print("\n========================================")
print("LOADING TEST DATA")
print("========================================")

test_df = pd.read_csv(TEST_FILE)

print("Test samples:", len(test_df))

test_df["text"] = test_df["text"].apply(clean_text)
test_df["label"] = test_df["label"].astype(int)


# ============================================================
# LOAD MODEL
# ============================================================

print("\n========================================")
print("LOADING TRAINED MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_FILE
)

print("Model loaded successfully!")


# ============================================================
# PREPARE INPUT
# ============================================================

x_test = tf.constant(
    test_df["text"].values,
    dtype=tf.string
)

y_true = test_df["label"].values


# ============================================================
# PREDICTIONS
# ============================================================

print("\n========================================")
print("MAKING PREDICTIONS")
print("========================================")

probabilities = model.predict(
    x_test,
    batch_size=64,
    verbose=1
).flatten()


# 0.5 threshold
y_pred = (
    probabilities >= 0.5
).astype(int)


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("FINAL TEST RESULTS")
print("========================================")

print(
    f"Accuracy  : {accuracy * 100:.2f}%"
)

print(
    f"Precision : {precision * 100:.2f}%"
)

print(
    f"Recall    : {recall * 100:.2f}%"
)

print(
    f"F1 Score  : {f1 * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print("\n========================================")
print("CLASSIFICATION REPORT")
print("========================================")

print(
    classification_report(
        y_true,
        y_pred,
        target_names=[
            "NOT SPAM",
            "SPAM"
        ],
        zero_division=0
    )
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)

print("\n========================================")
print("CONFUSION MATRIX")
print("========================================")

print(cm)


# ============================================================
# FALSE POSITIVE / FALSE NEGATIVE
# ============================================================

tn, fp, fn, tp = cm.ravel()

print("\n========================================")
print("ERROR ANALYSIS")
print("========================================")

print(
    "True Negatives  (correct NOT SPAM):",
    tn
)

print(
    "False Positives (NOT SPAM predicted SPAM):",
    fp
)

print(
    "False Negatives (SPAM predicted NOT SPAM):",
    fn
)

print(
    "True Positives   (correct SPAM):",
    tp
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

os.makedirs(
    "graphs",
    exist_ok=True
)

plt.figure(
    figsize=(7, 6)
)

plt.imshow(
    cm
)

plt.title(
    "Spam Detection Confusion Matrix"
)

plt.xlabel(
    "Predicted Label"
)

plt.ylabel(
    "Actual Label"
)

plt.xticks(
    [0, 1],
    ["NOT SPAM", "SPAM"]
)

plt.yticks(
    [0, 1],
    ["NOT SPAM", "SPAM"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    GRAPH_FILE,
    dpi=300
)

plt.close()


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    "Confusion matrix saved at:"
)

print(
    GRAPH_FILE
)