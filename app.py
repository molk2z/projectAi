from flask import Flask, render_template, request
import joblib
import numpy as np


# تعريف التطبيق Flask
app = Flask(__name__)

# تحميل النموذج المدرب مسبقًا
model = joblib.load('news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')
