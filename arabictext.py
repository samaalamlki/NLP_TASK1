import re
import string
import pickle
from collections import Counter

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from nltk import word_tokenize
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder


nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')


def load_data():
    data = pd.read_csv("data/ar_reviews_100k.tsv", sep="\t")
    return data


def eda(data):
    class_distribution(data)
    text_length(data)
    most_words(data)
    word_cloud(data)


def class_distribution(data):
    print('Class Distribution ')
    print(data['label'].value_counts()) # value Distribution


def text_length(data):
    data['text_length'] = data['text'].apply(lambda x: len(str(x).split()))
    print("\nText Length Stats:")  #lenght
    print(data["text_length"].describe())


def most_words(data):
    print("\nMost Frequent Words Per Class:")

    for class_name in data['label'].unique():
        print("\nClass:", class_name)

        class_data = data[data['label'] == class_name]

        word_occurance = Counter(" ".join(class_data["clean_text"]).split()) ##Most Frequent Words

        for i, j in word_occurance.most_common(20):
            print(i, '--->', j)


def word_cloud(data):
    text = " ".join(data['clean_text'])

    wordcloud = WordCloud(
        width=1000,
        height=500,
        background_color='white',
        font_path="/System/Library/Fonts/Supplemental/Arial Unicode.ttf"
    ).generate(text)

    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud)
    plt.axis('off')
    plt.title("Word Cloud")
    plt.show()


def clean_text(text):
    text = str(text)

    text = text.strip()
    text=re.sub(r'\s+',' ',text).strip() #white space normalization
    text = re.sub(r'[،؛؟…]', ' ', text)       # remove Arabic punctuation
    text = re.sub(r'\.+', ' ', text)
    text = text.replace('،', ' ')
    text = text.replace('؛', ' ')
    text = text.replace('؟', ' ')
    text = text.replace('…', ' ')    
    tokens= word_tokenize(text)

    tokens= [i for i in tokens if i.isdigit()==0] ##remove number 

    punctuation=set(string.punctuation)
    tokens=[i for i in tokens if i not in punctuation] ##remove punctuation

    arabic_stopwords=set(stopwords.words('arabic'))
    tokens=[i for i in tokens if i not in arabic_stopwords ]
    
    return ' '.join(tokens)



def predict_arabic(text):
    with open("arabic_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("arabic_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("arabic_label.pkl", "rb") as f:
        encoder = pickle.load(f)

    text = clean_text(text)

    x = vectorizer.transform([text])
    prediction = model.predict(x.toarray())

    result = encoder.inverse_transform(prediction)

    return result[0] 


if __name__ == "__main__":
    data = load_data()

    data['clean_text'] = data['text'].apply(clean_text)
    eda(data)

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(data['clean_text'])

    print(features.shape)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(data['label'])

    print(len(labels))

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    model = GaussianNB()
    model.fit(x_train.toarray(), y_train)
    with open("arabic_model.pkl", "wb") as f:
      pickle.dump(model, f)

    with open("arabic_vectorizer.pkl", "wb") as f:
      pickle.dump(vectorizer, f)

    with open("arabic_label.pkl", "wb") as f:
      pickle.dump(encoder, f)

    y_pred = model.predict(x_test.toarray())
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="inferno")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()




