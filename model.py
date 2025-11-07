import streamlit as st 
import joblib 
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')


ps = PorterStemmer()
stopword_list = stopwords.words('english')


st.title("Spam Email Checker")
st.write("This app checks whether an email is spam or not using a pre-trained model.")
# Load the pre-trained model
model = joblib.load("model.joblib")
vectorizer = joblib.load("vectorizer.joblib")

def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopword_list and i not in string.punctuation:
            y.append(i)
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    return " ".join(y)


# Input email text from user
email_text = st.text_area("Enter the email text here:")

if st.button("Check Email"):
    if email_text:
        input_transformed = transform_text(email_text)
        input_vectorized = vectorizer.transform([input_transformed])

        prediction = model.predict(input_vectorized)
        if prediction[0] == 0:
            st.error("The email is classified as SPAM.")
        elif prediction[0] == 1:
            st.success("The email is classified as NOT SPAM.")
        else:
            st.warning("Unexpected prediction result.")
    else:
        st.warning("Please enter some email text to check.") 

st.markdown("Note: This is a demo application built by Dnyaneshwar Kale for educational purposes.")




