import joblib
from sklearn.svm import SVC
import numpy as np
from numpy.typing import NDArray

class FaceClassifier:
    def __init__(self) -> None:
        self.clf: SVC = SVC(kernel="linear", probability=True)

    def train(self, X_train: NDArray[np.float16], y_train: NDArray[np.int8]) -> None:
        """
        Train the SVM classifier on the provided training data.

        Parameters:
        X_train (NDArray[np.float16]): Training data features, a 2D array.
        y_train (NDArray[np.int8]): Training data labels, a 1D array.
        """
        self.clf.fit(X_train, y_train)

    def predict(self, X_test: NDArray[np.float16]) -> NDArray[np.int8]:
        """
        Predict the class labels for the given test data.

        Parameters:
        X_test (NDArray[np.float16]): Test data features, a 2D array.

        Returns:
        NDArray[np.int8]: Predicted class labels, a 1D array.
        """
        return self.clf.predict(X_test)

    def save_model(self, filename: str) -> None:
        joblib.dump(self.clf, filename)

    def load_model(self, filename: str) -> None:
        self.clf: SVC = joblib.load(filename)
