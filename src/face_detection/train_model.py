import numpy as np
import logging

from data_loader import load_images_from_folder
from integral_image import compute_integral_image
from haar_features import HaarFeature
from classifier import FaceClassifier
from utils import normalize_images, sliding_window


def extract_features(images, feature_list):
    feature_vectors = []
    for img_index, img in enumerate(images):
        try:
            integral_img = compute_integral_image(img)
            # window_sizes = [(24, 24), (48, 48), (72, 72), (96, 96)]
            window_sizes = [(96, 96)]
            step_size = 48
            for window_size in window_sizes:
                for (x, y, window) in sliding_window(img, step_size, window_size):
                    features = [
                        feature.compute_feature(integral_img, (x, y), window_size)
                        for feature in feature_list
                    ]
                    feature_vectors.append(features)
            
            # Log the number of features extracted for the current image
            logging.info(f"Extracted {len(feature_vectors)} features from image")
        except Exception as e:
            logging.error(f"Error processing image {img_index}: {e}")

    return np.array(feature_vectors)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Start...")
    # Load data
    face_images = load_images_from_folder("small_data/faces/")
    non_face_images = load_images_from_folder("small_data/non_faces/")
    logging.info("Data loaded")

    # Preprocess data
    face_images = normalize_images(face_images)
    non_face_images = normalize_images(non_face_images)

    # Generate Haar features
    feature_list = [
        HaarFeature("two_horizontal", (0, 0), 1, 1),
        HaarFeature("two_vertical", (0, 0), 1, 1),
        # HaarFeature("three_horizontal", (0, 0), 1, 1),
        # HaarFeature("three_vertical", (0, 0), 1, 1),
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
    classifier.save_model("face_classifier.joblib")
