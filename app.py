from flask import Flask, render_template, request
import joblib
import numpy as np


# تعريف التطبيق Flask
app = Flask(__name__)

# تحميل النموذج المدرب مسبقًا
model = joblib.load('news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        news_text = request.form['news_text']
        # تحويل النص المدخل إلى تمثيل رقمي
        news_vectorized = vectorizer.transform([news_text])
        prediction = model.predict(news_vectorized)
        category = prediction[0]
        
        return render_template('index.html', prediction_text=f'The news is classified as: {category}')

if __name__ == "__main__":
    app.run(debug=True)
