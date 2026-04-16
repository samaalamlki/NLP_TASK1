from Langclassifier import predict_language
from arabictext import predict_arabic
from Englishtext import predict_english


text = input("Enter your text: ")

language = predict_language(text)
if language == "Arabic":
    prediction = predict_arabic(text)
else:
    prediction = predict_english(text)

    if prediction == 0:
        prediction = "Negative"
    elif prediction == 1:
        prediction = "Neutral"
    else:
        prediction = "Positive"
print("Language:", language)
print("Predicted Class:", prediction)