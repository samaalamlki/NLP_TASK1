import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, confusion_matrix

def load_data():
    arabic_data = pd.read_csv("data/ar_reviews_100k.tsv", sep="\t")
    english_data = pd.read_csv("data/sentiment_data.csv")

    arabic_data = arabic_data.dropna()
    english_data = english_data.dropna()

    arabic_text = arabic_data[["text"]]
    english_text = english_data[["Comment"]]

    english_text = english_text.rename(columns={"Comment": "text"})

    arabic_text["language"] = "Arabic"
    english_text["language"] = "English"

    arabic_text = arabic_text.sample(50000, random_state=42)
    english_text = english_text.sample(50000, random_state=42)

    data = pd.concat([arabic_text, english_text])

    return data


def clean_text(text):
    text = str(text)
    text = text.strip()
    text = re.sub(r"\s+", " ", text).strip() #white space normalization
    return text

def predict_language(text):
    text = str(text)

    if re.search(r'[\u0600-\u06FF]', text):
        return "Arabic"

    with open("lang_model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("lang_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    with open("lang_label.pkl", "rb") as f:
        encoder = pickle.load(f)

    text = clean_text(text)

    x = vectorizer.transform([text])
    prediction = model.predict(x.toarray())

    result = encoder.inverse_transform(prediction)

    return result[0]


if __name__ == "__main__":
    data = load_data()

    data["clean_text"] = data["text"].apply(clean_text)

    print("Language Distribution")
    print(data["language"].value_counts())
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    features = vectorizer.fit_transform(data['clean_text'])

    print(features.shape)

    encoder = LabelEncoder()
    labels = encoder.fit_transform(data['language'])

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
    with open("lang_model.pkl", "wb") as f:
     pickle.dump(model, f)

    with open("lang_vectorizer.pkl", "wb") as f:
     pickle.dump(vectorizer, f)

    with open("lang_label.pkl", "wb") as f:
     pickle.dump(encoder, f)

    y_pred = model.predict(x_test.toarray())
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="inferno")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()
