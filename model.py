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
