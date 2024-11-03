import numpy as np

from data_loader import load_images_from_folder
from integral_image import compute_integral_image
from haar_features import HaarFeature
from classifier import FaceClassifier
from utils import normalize_images


def extract_features(images, feature_list):
    feature_vectors = []
    for img in images:
        integral_img = compute_integral_image(img)
        image_height, image_width = img.shape
        top_left = (0, 0)
        window_sizes = [(24, 24), (48, 48), (72, 72), (96, 96)]
        for window_size in window_sizes:
            features = [
                feature.compute_feature(integral_img, top_left, window_size)
                for feature in feature_list
            ]
            feature_vectors.append(features)
    return np.array(feature_vectors)


if __name__ == "__main__":
    # Load data
    face_images = load_images_from_folder("data/faces/")
    non_face_images = load_images_from_folder("data/non_faces/")

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
    classifier = FaceClassifier()
    classifier.train(X, y)

    # Save the model
    classifier.save_model("face_classifier.joblib")
