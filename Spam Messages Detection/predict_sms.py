from src.preprocessing import clean_text
from src.data_loader import load_data
from models.spam_classifier import SpamClassifier

# Load and preprocess full dataset
df = load_data("data/spam.csv")
df['clean_text'] = df['text'].apply(clean_text)

clf = SpamClassifier()
X_train, X_test, y_train, y_test = clf.prepare_data(df)
clf.train(X_train, y_train)

# Predict on custom input
while True:
    sms = input("Enter an SMS message (or type 'exit'): ")
    if sms.lower() == 'exit':
        break
    cleaned = clean_text(sms)
    vectorized = clf.vectorizer.transform([cleaned])
    pred = clf.model.predict(vectorized)
    label = clf.encoder.inverse_transform(pred)
    print("Prediction:", label[0])
