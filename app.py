import os
import re
import pickle
import gradio as gr

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(BASE_DIR, "emotion_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(BASE_DIR, "tfidf.pkl"), "rb") as f:
    vectorizer = pickle.load(f)

with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    encoder = pickle.load(f)

URL_PATTERN = re.compile(r"https?://\S+|www\.\S+")
NON_LETTER_PATTERN = re.compile(r"[^a-z\s]+")
SPACE_PATTERN = re.compile(r"\s+")

def clean_text(text):
    text = text.lower()
    text = URL_PATTERN.sub("", text)
    text = NON_LETTER_PATTERN.sub("", text)
    text = SPACE_PATTERN.sub(" ", text)
    return text.strip()

def predict_emotion(text):
    if not text or not text.strip():
        return "Please enter valid text."

    cleaned = clean_text(text)

    if not cleaned:
        return "Please enter valid text."

    prediction = model.predict(vectorizer.transform([cleaned]))[0]
    emotion = encoder.inverse_transform([prediction])[0]

    return f"Detected Emotion: {emotion}"

interface = gr.Interface(
    fn=predict_emotion,
    inputs=gr.Textbox(
        lines=4,
        placeholder="Type your text here...",
        label="Input Text"
    ),
    outputs=gr.Text(
        label="Prediction"
    ),
    title="Emotion Detection",
    description="Enter text and predict emotion."
)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))

    interface.launch(
        server_name="0.0.0.0",
        server_port=port,
        enable_monitoring=False
    )