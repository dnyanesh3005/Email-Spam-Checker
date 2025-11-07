Email Spam Checker

A simple email spam classifier with a web interface (Streamlit) and a Python package for preprocessing, feature extraction, training, and inference. 
This repository contains code to train a model(Email Spam Checker), save/load the trained model(model.joblib), and run a Streamlit app to classify raw email text as spam or ham (not spam).

Features

Text preprocessing (lowercasing, punctuation removal, stopword removal, stemming)

TF-IDF vectorization (configurable)

Trainable classification models (Naive Bayes)

Model persistence (joblib)

Streamlit web app for interactive predictions

Example dataset processing and evaluation with confusion matrix & classification report


email-spam-checker/
├─ data/
│  ├─ spam.csv                
│  
├─ notebooks/
│  └─ Email Spam Checker.ipynb
├─ src/
│  ├─ vectorizer.joblib        # TF-IDF / vectorizers
├─ app.py                      # Streamlit app
├─ requirements.txt
├─ model/
│  └─ model.joblib             # saved trained model (optional)
├─ README.md
└─ LICENSE

Run the Streamlit demo locally:
pip install -r requirements.txt
streamlit run app.py


Note: We are using NLTK, we must need to download resources:
import nltk
nltk.download('punkt')
nltk.download('stopwords')

