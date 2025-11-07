# 📧 Email Spam Checker

A simple email spam classifier with a web interface (Streamlit) and a Python package for preprocessing, feature extraction, training, and inference.  
This repository contains code to train a model (**Email Spam Checker**), save/load the trained model (`model.joblib`), and run a Streamlit app to classify raw email text as **spam** or **ham (not spam)**.

---

## 🚀 Features

- Text preprocessing (lowercasing, punctuation removal, stopword removal, stemming)  
- TF-IDF vectorization (configurable)  
- Trainable classification models (Naive Bayes)  
- Model persistence using `joblib`  
- Streamlit web app for interactive predictions  
- Example dataset processing and evaluation with confusion matrix & classification report  

---

## 📁 Project Structure email-spam-checker/
├─ data/
│ ├─ spam.csv
│
├─ notebooks/
│ └─ Email Spam Checker.ipynb
├─ src/
│ ├─ vectorizer.joblib # TF-IDF / vectorizers
├─ app.py # Streamlit app
├─ requirements.txt
├─ model/
│ └─ model.joblib # saved trained model (optional)
├─ README.md
└─ LICENSE


---

## ▶️ Run the Streamlit Demo Locally

```bash
pip install -r requirements.txt
streamlit run app.py

⚙️ Note

Since we are using NLTK, make sure to download the required resources before running the app:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
