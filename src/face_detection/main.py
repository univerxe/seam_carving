import cv2
import numpy as np
from integral_image import compute_integral_image
from haar_features import HaarFeature
from classifier import FaceClassifier
from utils import normalize_images, sliding_window


def detect_face():
    # Load the pre-trained classifier
    classifier = FaceClassifier()
    classifier.load_model("face_classifier.joblib")

    # Define the Haar features used during training
    # For simplicity, we'll define a small set of features
    feature_list = [
        HaarFeature("two_horizontal", (0, 0), 1, 1),
        HaarFeature("two_vertical", (0, 0), 1, 1),
        # HaarFeature("three_horizontal", (0, 0), 1, 1),
        # HaarFeature("three_vertical", (0, 0), 1, 1),
    ]

    # Parameters
    # window_sizes = [(24, 24), (48, 48), (72, 72), (96, 96)] 
    window_sizes = [(96, 96)]
    step_size = 48  # Pixels to move the window

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (320, 240))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        integral_img_full = compute_integral_image(gray)

        detections = []

        for window_size in window_sizes:
            # Loop over the sliding window
            for x, y, window in sliding_window(gray, step_size, window_size):
                if (
                    window.shape[0] != window_size[1]
                    or window.shape[1] != window_size[0]
                ):
                    continue  # Skip incomplete windows at the edges

                # Compute integral image for the window
                integral_img = integral_img_full[
                    y : y + window_size[1] + 1, x : x + window_size[0] + 1
                ]

                # Extract features
                features = [
                    feature.compute_feature(integral_img, (0, 0), window_size)
                    for feature in feature_list
                ]
                features = np.array(features).reshape(1, -1)

                # Predict using the classifier
                prediction = classifier.predict(features)

                if prediction == 1:
                    detections.append((x, y, window_size[0], window_size[1]))

        # Draw rectangles around detections
        for x, y, w, h in detections:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Real-Time Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


detect_face()
