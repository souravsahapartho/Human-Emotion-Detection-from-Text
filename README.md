# Human Emotion Detection from Text

A machine learning and NLP-based project for detecting human emotions from text data.

---

## Project Overview

This project analyzes text messages and predicts human emotions such as:

- Joy
- Sadness
- Anger
- Surprise
- Neutral
- Fun
- Enthusiasm

The project uses machine learning and deep learning model selection techniques for text classification.

---

## Features

- Text-based emotion detection
- Data preprocessing and cleaning
- Emotion label encoding
- Emotion distribution visualization
- Multiple model selection
- NLP-based workflow

---

## Technologies Used

- Python
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Jupyter Notebook

---

## Dataset

The dataset contains two columns:

| Column | Description |
|---|---|
| text | Input sentence/text |
| emotion | Emotion label |

Example:

| text | emotion |
|---|---|
| i feel very happy today | joy |
| i am upset | sadness |

---

## Data Preprocessing

The following preprocessing steps were performed:

- Removing null values
- Removing duplicate rows
- Converting text to lowercase
- Removing punctuation
- Removing numbers
- Removing extra spaces
- Label encoding

---

## Selected Models

The following models were selected for emotion classification:

- Naive Bayes
- Support Vector Machine (SVM)
- Random Forest
- LSTM

---

## Why These Models?

### Naive Bayes
Fast and effective for text classification tasks.

### SVM
Provides good accuracy for NLP and text classification.

### Random Forest
Improves prediction stability and reduces overfitting.

### LSTM
Understands text sequence and context better.

---

## Visualization

Emotion distribution was visualized using Seaborn countplot.

---

## Project Workflow

```text
Dataset Collection
        ↓
Data Preprocessing
        ↓
Label Encoding
        ↓
Data Visualization
        ↓
Model Selection
```

---

## Future Improvements

- Model training and testing
- Accuracy comparison
- Real-time emotion prediction
- Web application deployment
