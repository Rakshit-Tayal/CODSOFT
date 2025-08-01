from src.data_loader import load_data
from src.preprocessing import clean_text
from models.spam_classifier import SpamClassifier

# Load data
df = load_data("data/spam.csv")

# Clean text
df['clean_text'] = df['text'].apply(clean_text)

# Initialize and train model
clf = SpamClassifier()
X_train, X_test, y_train, y_test = clf.prepare_data(df)
clf.train(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
clf.evaluate(y_test, y_pred)
