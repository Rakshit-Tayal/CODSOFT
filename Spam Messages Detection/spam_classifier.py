from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

class SpamClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=3000)
        self.model = LogisticRegression(max_iter=1000)
        self.encoder = LabelEncoder()

    def prepare_data(self, df):
        df['label'] = self.encoder.fit_transform(df['label'])  # ham = 0, spam = 1
        X = self.vectorizer.fit_transform(df['clean_text'])
        y = df['label']
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_test, y_pred):
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("\nReport:\n", classification_report(y_test, y_pred, target_names=['ham', 'spam']))
