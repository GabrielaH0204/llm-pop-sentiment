# llm-pop-sentiment
A lightweight sentiment analysis model trained on 32k Reddit posts about pop artists and pop culture, using a DistilBERT transformer fine-tuned for positive, neutral, and negative sentiment classification.
# Pop Sentiment LLM — Reddit Music Posts Classifier

This project trains a lightweight sentiment analysis model on 32,000+ short Reddit posts about pop music, pop artists, and fan culture.  
The dataset includes posts referencing artists like Taylor Swift, Olivia Rodrigo, and Billie Eilish, as well as trending topics such as the Grammys and Billboard charts.

The model is fine-tuned using a DistilBERT transformer for 3-class sentiment classification:
- positive
- neutral
- negative

---

## Dataset

- ~32k English Reddit posts  
- Posts ≤280 characters (similar to tweet-style content)  
- Collected from subreddits:  
  - r/popheads  
  - r/Music  
  - r/Billboard  
- Labels (pos/neu/neg) generated via CardiffNLP Twitter RoBERTa model  
- File included: `reddit_artist_posts_sentiment.csv`

Columns:
- `text` → the post (title + body merged)
- `label` → 0 = negative, 1 = neutral, 2 = positive

---

#Model

The model uses:
- **DistilBERT** (`distilbert-base-uncased`)
- Tokenization with Transformers
- PyTorch backend
- HuggingFace Trainer API (simplified training loop)

Why DistilBERT?
- Smaller  
- Faster  
- Cheaper to train  
- Excellent for sentiment tasks on short text

---

# Training Script Overview

Main steps:

1. Load CSV using pandas  
2. Map text + label into a HuggingFace Dataset  
3. Tokenize using DistilBERT tokenizer  
4. Train a classification head with Trainer  
5. Save the fine-tuned model locally

Core libraries:
- `transformers`
- `datasets`
- `pandas`
- `torch`
- `scikit-learn` (for evaluation metrics)

---

# How to Train

Run the notebook or script:
python train_pop_sentiment.py

or in a notebook:
python
Copiar código
trainer.train()
Training runs quickly because:

Dataset is small

Model is compact

Only 1 epoch is used (demo-friendly)

# Evaluation
The evaluation uses:

-Accuracy
-F1-score
-Prec/Rec (macro)

-Example output:

text
Copiar código
Accuracy: ~0.83
Macro F1: ~0.81

# Inference Example
text = "Taylor Swift just dropped a masterpiece!"
inputs = tokenizer(text, return_tensors="pt")
with torch.no_grad():
    logits = model(**inputs).logits
prediction = torch.argmax(logits, dim=1).item()
print(prediction)

Output:
positive

# License
For research and educational use.
Dataset originally sourced from publicly available Reddit posts.

# Author

Luisa Gabriela Hernandez
