# NLP_TASK1

Arabic and English sentiment classification system with language detection.

## Files
- main.py: handles user input and displays final output
- Langclassifier.py: detects if the input text is Arabic or English
- arabictext.py: classifies Arabic text sentiment
- Englishtext.py: classifies English text sentiment

## Notes
The datasets and trained model files are not included in this repository.
Run the training files first to generate the `.pkl` model files:

```bash
python arabictext.py
python Englishtext.py
python Langclassifier.py
