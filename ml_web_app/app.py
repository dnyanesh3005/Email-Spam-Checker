from flask import Flask, render_template, request
import joblib
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load model and vectorizer
model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

@app.route('/')
def home():
    return render_template('index.html', prediction_text='')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get the email text from form
        email_text = request.form.get('email', '')

        # Handle empty input
        if not email_text.strip():
            return render_template('index.html', prediction_text='Please enter email text.')

        # Transform text using the saved vectorizer
        X = vectorizer.transform([email_text])

        # Make prediction
        prediction = model.predict(X)[0]

        # Convert result to readable output
        if isinstance(prediction, (np.ndarray, list)):
            prediction = prediction[0]

        if int(prediction) == 1:
            result = "🚨 Spam Email"
        elif int(prediction) == 0:
            result = "✅ Not Spam Email"
        else:
            result = "❓ Unknown Prediction"

        return render_template('index.html', prediction_text=result)

    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
