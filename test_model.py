import tensorflow as tf

MODEL_PATH = "model/spam_model.keras"

print("Loading spam detection model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!\n")


def predict_message(message):
    input_text = tf.constant([message], dtype=tf.string)

    prediction = model.predict(input_text, verbose=0)[0][0]

    if prediction >= 0.5:
        result = "SPAM"
        confidence = prediction * 100
    else:
        result = "NOT SPAM"
        confidence = (1 - prediction) * 100

    print("\n" + "=" * 60)
    print("PREDICTION RESULT")
    print("=" * 60)
    print(f"Result     : {result}")
    print(f"Confidence : {confidence:.2f}%")
    print("=" * 60)


while True:
    print("\nPaste your SMS or email below.")
    print("For a multi-line email, type END on a new line when finished.")
    print("Type EXIT to close the program.")

    lines = []

    while True:
        text = input()

        if text.strip().upper() == "EXIT":
            print("\nProgram closed.")
            exit()

        if text.strip().upper() == "END":
            break

        lines.append(text)

    message = "\n".join(lines).strip()

    if not message:
        print("Please enter some text.")
        continue

    predict_message(message)