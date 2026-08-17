import streamlit as st
import joblib
import torch
import re
import string
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification

st.set_page_config(page_title="Spam Detector: Classical vs Transformer", page_icon="📨")

# --- Load models (cached so they only load once) ---
@st.cache_resource
def load_models():
    tfidf = joblib.load('tfidf_vectorizer.pkl')
    nb_model = joblib.load('naive_bayes_model.pkl')
    lr_model = joblib.load('logistic_regression_model.pkl')
    bert_tokenizer = AutoTokenizer.from_pretrained('Pratiiiiiiiik/distilbert-spam-detector')
    bert_model = AutoModelForSequenceClassification.from_pretrained('Pratiiiiiiiik/distilbert-spam-detector')
    bert_model.eval()
    return tfidf, nb_model, lr_model, bert_tokenizer, bert_model

tfidf, nb_model, lr_model, bert_tokenizer, bert_model = load_models()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.strip()
    return text

def predict_classical(text, model):
    cleaned = clean_text(text)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    prob = model.predict_proba(vec)[0]
    return pred, prob

def predict_bert(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = bert_model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)[0]
    pred = torch.argmax(probs).item()
    return pred, probs.numpy()

# --- UI ---
st.title("📨 Spam Detector")
st.caption("Comparing a classical TF-IDF baseline against a fine-tuned DistilBERT model")

model_choice = st.radio(
    "Choose model:",
    ["Naive Bayes (TF-IDF)", "Logistic Regression (TF-IDF)", "DistilBERT (fine-tuned)", "Compare All Three"],
    horizontal=True
)

message = st.text_area(
    "Enter a message to classify:",
    placeholder="e.g. Congratulations! You've won a free prize, click here to claim now",
    height=100
)

if st.button("Classify", type="primary") and message.strip():

    def show_result(name, pred, prob, elapsed):
        label = "🚨 SPAM" if pred == 1 else "✅ HAM"
        confidence = prob[pred] * 100
        st.metric(label=name, value=label, delta=f"{confidence:.1f}% confidence")
        st.caption(f"Inference time: {elapsed*1000:.2f} ms")

    if model_choice == "Compare All Three":
        col1, col2, col3 = st.columns(3)

        with col1:
            start = time.time()
            pred, prob = predict_classical(message, nb_model)
            show_result("Naive Bayes", pred, prob, time.time() - start)

        with col2:
            start = time.time()
            pred, prob = predict_classical(message, lr_model)
            show_result("Logistic Regression", pred, prob, time.time() - start)

        with col3:
            start = time.time()
            pred, prob = predict_bert(message)
            show_result("DistilBERT", pred, prob, time.time() - start)

    else:
        start = time.time()
        if model_choice == "Naive Bayes (TF-IDF)":
            pred, prob = predict_classical(message, nb_model)
        elif model_choice == "Logistic Regression (TF-IDF)":
            pred, prob = predict_classical(message, lr_model)
        else:
            pred, prob = predict_bert(message)
        elapsed = time.time() - start
        show_result(model_choice, pred, prob, elapsed)

st.divider()
with st.expander("📊 Model performance on test set"):
    st.markdown("""
    | Model | Accuracy | Spam F1 |
    |---|---|---|
    | Naive Bayes | 97% | 0.88 |
    | Logistic Regression | 96% | 0.84 |
    | DistilBERT (fine-tuned) | 99.4% | 0.976 |

    DistilBERT trades inference speed for meaningfully better spam recall,
    catching cases the classical models miss.
    """)