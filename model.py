import os
import pandas as pd

def load_data(data_dir):
    texts = []
    labels = []
    categories = os.listdir("./data")  # أسماء المجلدات (مثل politics, sports)

    for category in categories:
        category_path = os.path.join(data_dir, category)
        if os.path.isdir(category_path):
            for filename in os.listdir(category_path):
                file_path = os.path.join(category_path, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    texts.append(file.read())  # النصوص من ملفات .txt
                    labels.append(category)  # التصنيف

    return pd.DataFrame({'text': texts, 'label': labels})

# استخدام الدالة لتحميل البيانات
data_dir = './data'  # مسار مجلد البيانات
dataset = load_data(data_dir)
print(dataset.head())  # عرض أول 5 أسطر من البيانات


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
import joblib


# تقسيم البيانات
X = dataset['text']
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تحويل النصوص إلى ميزات عددية
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)



# تدريب النموذج
model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# اختبار النموذج
predictions = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy:.2f}")

# حفظ النموذج والمتجه
joblib.dump(model, 'news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
