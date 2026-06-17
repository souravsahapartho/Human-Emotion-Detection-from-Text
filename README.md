# Human Emotion Detection from Text

A machine learning and NLP-based project for detecting human emotions from text data.

---

## Project Overview

This project analyzes text messages and predicts 11 different human emotions. The project applies text preprocessing, TF-IDF feature extraction, and evaluates multiple machine learning models to find the best classifier. The final optimized model is deployed via an interactive web application.

### Detected Emotions
The model is capable of classifying text into the following categories:
* Anger
* Empty
* Enthusiasm
* Fun
* Happiness
* Hate
* Love
* Neutral
* Relief
* Sadness
* Surprise

---

## Features

- **Text Data Preprocessing**: Handles URL removal, lowercasing, and removal of non-alphabetic characters using Regex.
- **Feature Extraction**: Utilizes `TfidfVectorizer` (up to 5000 features, analyzing unigrams and bigrams) for text vectorization.
- **Model Comparison**: Trains and evaluates multiple algorithms to ensure the highest accuracy.
- **Web App Interface**: Deployed as a functional UI using Gradio for real-time predictions.

---

## Technologies Used

- **Python**
- **Pandas & Numpy** (Data Manipulation)
- **Matplotlib & Seaborn** (Data Visualization)
- **Scikit-learn** (Preprocessing, Evaluation, Naive Bayes, Logistic Regression, AdaBoost)
- **XGBoost** (Advanced Classification)
- **Gradio** (Web Interface Deployment)
- **Pickle** (Model Serialization)

---

## Model Selection & Performance

Four different machine learning algorithms were trained and tested against the dataset. **XGBoost** proved to be the best-performing model and was chosen for final deployment.

| Model | Accuracy Score |
|---|---|
| **XGBoost** | **~80.85%** |
| Logistic Regression | ~80.60% |
| Multinomial Naive Bayes | ~70.02% |
| AdaBoost | ~57.84% |

---

## Application Deployment

The project successfully deploys the trained XGBoost model using a **Gradio** web interface. The application loads the saved `emotion_model.pkl`, `tfidf.pkl`, and `label_encoder.pkl` files to take user text input, clean it using identical preprocessing rules, and immediately output the predicted emotion.

### How to Run the App

1. Ensure all dependencies are installed (e.g., `gradio`, `xgboost`, `scikit-learn`).
2. Run the deployment script:
   ```bash
   python app.py
