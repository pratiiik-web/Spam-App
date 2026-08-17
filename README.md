# 📨 Spam Detector: Classical ML vs Fine-Tuned Transformer

A side-by-side comparison of classical NLP (TF-IDF + Naive Bayes / Logistic Regression) against a fine-tuned DistilBERT transformer for SMS spam classification, deployed as an interactive Streamlit app.

## Overview

This project answers a practical question: **when does it actually pay off to reach for a transformer instead of a classical model?**

Three models are trained on the same SMS spam dataset and compared head-to-head on accuracy, spam recall, and inference latency, so the tradeoffs are visible rather than assumed.

## Models

| Model | Type | Accuracy | Spam F1 |
|---|---|---|---|
| Naive Bayes | TF-IDF + MultinomialNB | 97% | 0.88 |
| Logistic Regression | TF-IDF + LogisticRegression | 96% | 0.84 |
| DistilBERT (fine-tuned) | Transformer, HuggingFace | 99.4% | 0.976 |

The classical models are fast but conservative, missing 20-28% of actual spam messages (low recall). Fine-tuning DistilBERT closed most of that gap, at the cost of significantly higher inference latency.

### Latency (single prediction)

| Model | CPU inference time |
|---|---|
| Naive Bayes (TF-IDF) | ~3-6 ms |
| DistilBERT | ~40-80 ms (GPU) / higher on CPU |

TF-IDF + Naive Bayes remains roughly 10-100x faster per prediction, which matters for high-throughput use cases like filtering millions of messages. DistilBERT's accuracy gain is more valuable when spam is costlier to miss than the added compute.

## Dataset

[SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection) — 5,572 labeled SMS messages (4,825 ham / 747 spam), split 80/20 with `random_state=42` shared identically across both the classical and transformer pipelines for a fair comparison.

## Tech Stack

- **Classical NLP:** scikit-learn (TF-IDF vectorization, Naive Bayes, Logistic Regression)
- **Transformer:** HuggingFace `transformers` (DistilBERT, fine-tuned via `Trainer` API)
- **Training environment:** Google Colab (free GPU tier)
- **Deployment:** Streamlit
- **Model hosting:** [HuggingFace Hub](https://huggingface.co/Pratiiiiiiiik/distilbert-spam-detector)

## Project Structure

```
spam_app/
├── app.py                          # Streamlit app
├── tfidf_vectorizer.pkl            # Fitted TF-IDF vectorizer
├── naive_bayes_model.pkl           # Trained Naive Bayes model
├── logistic_regression_model.pkl   # Trained Logistic Regression model
└── baseline_results.json           # Saved classical model metrics
```

DistilBERT weights are hosted on HuggingFace Hub and loaded at runtime rather than bundled locally, keeping the repo lightweight.

## Running Locally

```bash
pip install streamlit torch transformers scikit-learn joblib
streamlit run app.py
```

Then open `http://localhost:8501` in your browser.

## Key Findings

1. **Accuracy vs. latency is a real tradeoff, not a formality.** DistilBERT wins on every accuracy metric, but at roughly 10-100x the inference cost of the classical models.
2. **Overfitting showed up early.** Validation loss bottomed out at epoch 2 (F1 = 0.976) and slightly worsened by epoch 3, even as training loss kept dropping — a reminder to track eval metrics alongside training loss rather than just training for a fixed number of epochs.
3. **Class imbalance shaped the baseline weaknesses.** With spam making up only ~13% of the dataset, both classical models optimized for overall accuracy at the expense of spam recall — exactly the kind of blind spot fine-tuning helped correct.

## Future Improvements

- Address class imbalance directly (class weighting, oversampling) to see if classical models can close the recall gap without a transformer
- Add sentence-embedding models (Sentence-BERT) as a middle ground between TF-IDF and full fine-tuning
- Quantize the DistilBERT model to shrink the latency gap for production use

## Author

Pratik Karale ([pratik.logs](https://huggingface.co/Pratiiiiiiiik))
