from sklearn.svm import SVC
import joblib


class FaceClassifier:
    def __init__(self):
        self.clf = SVC(kernel="linear", probability=True)

    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)

    def predict(self, X_test):
        return self.clf.predict(X_test)

    def save_model(self, filename):
        joblib.dump(self.clf, filename)

    def load_model(self, filename):
        self.clf = joblib.load(filename)
