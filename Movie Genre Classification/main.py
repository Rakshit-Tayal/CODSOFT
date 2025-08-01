from src.data_loader import load_train_data, load_test_data
from src.preprocessing import clean_text
from models.genre_classifier import GenreClassifier
import pandas as pd

# Load data
train_df = load_train_data("data/train_data.txt")
test_df = load_test_data("data/test_data.txt")
test_sol_df = load_test_data("data/test_data_solution.txt", has_genre=True)

# Clean text
train_df['clean'] = train_df['description'].apply(clean_text)
test_df['clean'] = test_df['description'].apply(clean_text)
test_sol_df['clean'] = test_sol_df['description'].apply(clean_text)

# Initialize model
clf = GenreClassifier()

# Prepare inputs
y_train = clf.encode_labels(train_df['genre'])
X_train, X_test = clf.prepare_data(train_df['clean'], test_df['clean'])

# Train model
clf.train(X_train, y_train)

# Predict
y_pred_encoded = clf.predict(X_test)
y_pred = clf.decode_labels(y_pred_encoded)

# Save predictions
test_df['predicted_genre'] = y_pred
test_df[['title', 'predicted_genre']].to_csv("predicted_genres.csv", index=False)

# Evaluation
y_true = test_sol_df['genre']
clf.evaluate(y_true, y_pred)
