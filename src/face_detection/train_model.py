import logging

import numpy as np

from src.utils import normalize_images
from src.haar_features import HaarFeature
from src.classifier import FaceClassifier
from src.feature_extractor import extract_features
from src.data_loader import load_images_from_folder


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Start...")
    # Load data
    face_images = load_images_from_folder("data_10pics/faces/", (320, 240))
    non_face_images = load_images_from_folder("data_10pics/non_faces/", (320, 240))
    logging.info("Data loaded")

    # Preprocess data
    face_images = normalize_images(face_images)
    non_face_images = normalize_images(non_face_images)

    # Generate Haar features
    feature_list = [
        HaarFeature("two_horizontal", (0, 0), 1, 1),
        HaarFeature("two_vertical", (0, 0), 1, 1),
        HaarFeature("three_horizontal", (0, 0), 1, 1),
        HaarFeature("three_vertical", (0, 0), 1, 1),
    ]

    # Extract features
    X_faces = extract_features(face_images, feature_list)
    X_non_faces = extract_features(non_face_images, feature_list)
    X = np.vstack((X_faces, X_non_faces))
    y = np.hstack((np.ones(len(X_faces)), np.zeros(len(X_non_faces))))

    # Train classifier
    logging.info("Training...")
    classifier = FaceClassifier()
    classifier.train(X, y)

    # Save the model
    classifier.save_model("model/face_classifier.joblib")
