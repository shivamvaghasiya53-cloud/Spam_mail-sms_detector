# AI Spam Email & SMS Detector

An AI-powered Spam Email and SMS Detection System built using TensorFlow, Keras, Bidirectional LSTM, and Flask.

The system analyzes SMS messages and email content and classifies them as either **SPAM** or **NOT SPAM**. It also provides a confidence score and maintains recent prediction history in the browser.

## Features

- SPAM / NOT SPAM classification
- Supports SMS and long email messages
- Deep Learning-based text classification
- Bidirectional LSTM model
- Prediction confidence score
- Real-world message testing
- Flask web application
- Prediction history using browser LocalStorage
- Confusion matrix and evaluation metrics
- Responsive web interface

## Technologies Used

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- Scikit-learn
- Flask
- HTML
- CSS
- JavaScript

## Deep Learning Model

The project uses the following architecture:

Text Input  
↓  
TextVectorization  
↓  
Embedding  
↓  
Bidirectional LSTM  
↓  
Global Max Pooling  
↓  
Dense Layers  
↓  
Dropout  
↓  
Sigmoid Output  
↓  
SPAM / NOT SPAM

## Dataset

The project uses a combination of SMS and email datasets.

The data preparation process includes:

- Data loading
- Text cleaning
- Label processing
- Duplicate removal
- Dataset combination
- Shuffling
- Stratified train/validation/test splitting

Final dataset distribution:

- NOT SPAM: 33,253
- SPAM: 40,903
- Total: 74,156

## Model Performance

The final model was evaluated on 7,416 unseen test samples.

| Metric | Score |
|---|---:|
| Accuracy | 99.24% |
| Precision | 99.46% |
| Recall | 99.17% |
| F1 Score | 99.31% |

## Confusion Matrix

The evaluation generated a confusion matrix showing the number of correctly and incorrectly classified messages.

## Project Structure

```text
spam_email_detector/
│
├── dataset/
│   ├── SMSSpamCollection
│   ├── email_text.csv
│   ├── processed_data.csv
│   ├── train.csv
│   ├── validation.csv
│   └── test.csv
│
├── model/
│   └── spam_model.keras
│
├── graphs/
│   └── confusion_matrix.png
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── app.py
├── prepare_data.py
├── train_model.py
├── evaluate_model.py
├── test_model.py
├── check_dataset.py
├── requirements.txt
└── README.md