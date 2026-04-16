import re
from collections import Counter
import pickle

import nltk
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder

nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')


def load_data():
    data = pd.read_csv("data/sentiment_data.csv")
    data = data.dropna()
    return data


def eda(data):
    class_distribution(data)
    text_length(data)
    most_words(data)
    word_cloud(data)


def class_distribution(data):
    print('Class Distribution ')
    print(data['Sentiment'].value_counts()) # value Distribution


def text_length(data):
    data['text_length'] = data['Comment'].apply(lambda x: len(str(x).split()))
    print("\nText Length Stats:")  #lenght
    print(data["text_length"].describe())


def most_words(data):
    print("\nMost Frequent Words Per Class:")

    for class_name in data['Sentiment'].unique():
        print("\nClass:", class_name)

        class_data = data[data['Sentiment'] == class_name]

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


def get_pos(tag):
    first_letter = tag[0]

    pos_map = {
        'J': wordnet.ADJ,
        'V': wordnet.VERB,
        'R': wordnet.ADV,
        'N': wordnet.NOUN
    }

    return pos_map.get(first_letter, wordnet.NOUN)

def clean_text(text):
    text = str(text)

    text = text.lower()
    text = text.strip()

    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip() #white space normalization


    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()
    tags = pos_tag(tokens)

    new_tokens = []
    for word, tag in tags:
        lemmatized_word = lemmatizer.lemmatize(word, get_pos(tag))
        new_tokens.append(lemmatized_word)
    
    return ' '.join(new_tokens)

def predict_english(text):
    with open("english_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("english_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("english_label.pkl", "rb") as f:
        encoder = pickle.load(f)

    text = clean_text(text)

    x = vectorizer.transform([text])
    prediction = model.predict(x.toarray())

    result = encoder.inverse_transform(prediction)

    return result[0]



if __name__ == "__main__":
    data = load_data()

    data['clean_text'] = data['Comment'].apply(clean_text)
    eda(data)  # EDA

    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(data['clean_text'])

    print(features.shape)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(data['Sentiment'])

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
    with open("english_model.pkl", "wb") as f:
     pickle.dump(model, f)

    with open("english_vectorizer.pkl", "wb") as f:
     pickle.dump(vectorizer, f)

    with open("english_label.pkl", "wb") as f:
     pickle.dump(encoder, f)

    y_pred = model.predict(x_test.toarray())
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="inferno")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()