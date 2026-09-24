import os
import re
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers

# ============================================================
# SETTINGS
# ============================================================

TRAIN_FILE = "dataset/train.csv"
VALIDATION_FILE = "dataset/validation.csv"
MODEL_PATH = "model/spam_model.keras"

MAX_TOKENS = 30000
SEQUENCE_LENGTH = 300
EMBEDDING_DIM = 128

BATCH_SIZE = 64
EPOCHS = 15


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)

    # Remove characters that can cause Windows encoding problems
    text = text.encode("ascii", "ignore").decode("ascii")

    # Normalize spaces
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ============================================================
# LOAD DATA
# ============================================================

print("\n========================================")
print("LOADING DATA")
print("========================================")

train_df = pd.read_csv(TRAIN_FILE)
validation_df = pd.read_csv(VALIDATION_FILE)

print("Training samples:", len(train_df))
print("Validation samples:", len(validation_df))


# ============================================================
# CLEAN DATA
# ============================================================

print("\nCleaning text...")

train_df["text"] = train_df["text"].apply(clean_text)
validation_df["text"] = validation_df["text"].apply(clean_text)

train_df["label"] = train_df["label"].astype("int32")
validation_df["label"] = validation_df["label"].astype("int32")


# ============================================================
# CREATE TF.DATA DATASETS
# ============================================================

print("\nCreating TensorFlow datasets...")

x_train = tf.constant(
    train_df["text"].values,
    dtype=tf.string
)

y_train = tf.constant(
    train_df["label"].values,
    dtype=tf.int32
)

x_validation = tf.constant(
    validation_df["text"].values,
    dtype=tf.string
)

y_validation = tf.constant(
    validation_df["label"].values,
    dtype=tf.int32
)

train_dataset = tf.data.Dataset.from_tensor_slices(
    (x_train, y_train)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (x_validation, y_validation)
)

train_dataset = (
    train_dataset
    .shuffle(20000, seed=42)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = (
    validation_dataset
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)


# ============================================================
# TEXT VECTORIZATION
# ============================================================

print("\n========================================")
print("CREATING TEXT VECTORIZER")
print("========================================")

vectorizer = layers.TextVectorization(
    max_tokens=MAX_TOKENS,
    output_mode="int",
    output_sequence_length=SEQUENCE_LENGTH,
    standardize="lower_and_strip_punctuation"
)

# Learn vocabulary from training data only
vectorizer.adapt(
    train_dataset.map(lambda text, label: text)
)

print("Vocabulary size:",
      len(vectorizer.get_vocabulary()))


# ============================================================
# BUILD MODEL
# ============================================================

print("\n========================================")
print("BUILDING TENSORFLOW MODEL")
print("========================================")

inputs = tf.keras.Input(
    shape=(),
    dtype=tf.string,
    name="message"
)

x = vectorizer(inputs)

x = layers.Embedding(
    input_dim=MAX_TOKENS,
    output_dim=EMBEDDING_DIM,
    mask_zero=True
)(x)

# Bidirectional LSTM helps understand word order and context
x = layers.Bidirectional(
    layers.LSTM(
        64,
        return_sequences=True
    )
)(x)

x = layers.GlobalMaxPooling1D()(x)

x = layers.Dense(
    128,
    activation="relu"
)(x)

x = layers.Dropout(0.4)(x)

x = layers.Dense(
    64,
    activation="relu"
)(x)

x = layers.Dropout(0.3)(x)

outputs = layers.Dense(
    1,
    activation="sigmoid",
    name="spam_probability"
)(x)

model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs
)


# ============================================================
# COMPILE
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),
    loss="binary_crossentropy",
    metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
    ]
)

model.summary()


# ============================================================
# CALLBACKS
# ============================================================

os.makedirs("model", exist_ok=True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True,
    verbose=1
)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    MODEL_PATH,
    monitor="val_loss",
    save_best_only=True,
    verbose=1
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=1,
    min_lr=0.00001,
    verbose=1
)


# ============================================================
# TRAIN
# ============================================================

print("\n========================================")
print("STARTING TRAINING")
print("========================================")

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=[
        early_stopping,
        reduce_lr,
        checkpoint
    ]
)


# ============================================================
# LOAD BEST MODEL
# ============================================================

print("\n========================================")
print("LOADING BEST MODEL")
print("========================================")

model = tf.keras.models.load_model(
    MODEL_PATH
)


# ============================================================
# FINAL VALIDATION
# ============================================================

print("\n========================================")
print("FINAL VALIDATION RESULTS")
print("========================================")

results = model.evaluate(
    validation_dataset,
    verbose=1
)

for name, value in zip(
    model.metrics_names,
    results
):
    print(f"{name}: {value:.4f}")


print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print("Model saved at:")
print(MODEL_PATH)

print("\nThe best model is now ready for testing.")