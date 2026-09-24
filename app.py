from flask import Flask, render_template, request, jsonify
import tensorflow as tf

app = Flask(__name__)

MODEL_PATH = "model/spam_model.keras"

print("Loading spam detection model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data:
            return jsonify({
                "error": "No data received."
            }), 400

        message = data.get("message", "").strip()

        if not message:
            return jsonify({
                "error": "Please enter an SMS or email."
            }), 400

        # Convert text into TensorFlow input
        input_text = tf.constant([message], dtype=tf.string)

        # Get prediction
        prediction = float(model.predict(input_text, verbose=0)[0][0])

        # 0 = NOT SPAM
        # 1 = SPAM
        if prediction >= 0.5:
            result = "SPAM"
            confidence = prediction * 100
        else:
            result = "NOT SPAM"
            confidence = (1 - prediction) * 100

        return jsonify({
            "result": result,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("Error:", e)

        return jsonify({
            "error": "Something went wrong while analyzing the message."
        }), 500


if __name__ == "__main__":
    app.run(debug=True)