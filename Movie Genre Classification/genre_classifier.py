from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import classification_report, accuracy_score
from collections import Counter

class GenreClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=7000,
            min_df=5,
            ngram_range=(1, 2),
            sublinear_tf=True,
            stop_words='english'
        )
        self.selector = SelectKBest(chi2, k=3000)
        self.encoder = LabelEncoder()

    def compute_class_weights(self, y):
        counts = Counter(y)
        total = sum(counts.values())
        weights = {cls: total / (len(counts) * count) for cls, count in counts.items()}
        return weights

    def prepare_data(self, train_texts, test_texts):
        X_train = self.vectorizer.fit_transform(train_texts)
        X_test = self.vectorizer.transform(test_texts)
        return X_train, X_test

    def select_features(self, X_train, y_train, X_test):
        X_train_new = self.selector.fit_transform(X_train, y_train)
        X_test_new = self.selector.transform(X_test)
        return X_train_new, X_test_new

    def encode_labels(self, labels):
        return self.encoder.fit_transform(labels)

    def decode_labels(self, encoded_labels):
        return self.encoder.inverse_transform(encoded_labels)

    def train(self, X_train, y_train):
        class_weights = self.compute_class_weights(y_train)
        lr = LogisticRegression(max_iter=200, class_weight=class_weights)
        svm_raw = LinearSVC(class_weight=class_weights)
        svm = CalibratedClassifierCV(svm_raw)

        self.model = VotingClassifier(
            estimators=[('lr', lr), ('svm', svm)],
            voting='soft'
        )
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def evaluate(self, y_true, y_pred):
        print("Accuracy:", accuracy_score(y_true, y_pred))
        print(classification_report(y_true, y_pred))
