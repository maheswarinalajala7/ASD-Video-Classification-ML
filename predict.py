
import os
import cv2
import numpy as np
import pandas as pd

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

ROOT = r"C:\Users\nalaj\Downloads\autism_data_anonymized\autism_data_anonymized\testing_set"
OUTPUT = "testing_results.csv"

print("Loading LSTM model...")
model = load_model("small_test_model.keras")

print("Loading VGG16...")
vgg = VGG16(weights="imagenet", include_top=True)

def get_features(video_path):
    cap = cv2.VideoCapture(video_path)
    features = []

    if not cap.isOpened():
        return None

    for second in range(5):
        cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
        success, frame = cap.read()

        if not success:
            break

        frame = cv2.resize(frame, (224, 224))
        image = img_to_array(frame)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        feature = vgg.predict(image, verbose=0)[0]
        features.append(feature)

    cap.release()

    if not features:
        return None

    while len(features) < 5:
        features.append(np.zeros(1000, dtype=np.float32))

    return np.array(features[:5], dtype=np.float32)

videos = []

for label in ["ASD", "TD"]:
    folder = os.path.join(ROOT, label)

    for current_root, _, files in os.walk(folder):
        for name in files:
            if name.lower().endswith(
                (".mp4", ".avi", ".mov", ".mkv", ".mpeg", ".mpg")
            ):
                videos.append((os.path.join(current_root, name), label))

print("Videos found:", len(videos))

if not videos:
    raise SystemExit("No videos found. Check the testing_set path.")

if os.path.exists(OUTPUT):
    old = pd.read_csv(OUTPUT)
    results = old.to_dict("records")
    completed = set(old["Video Path"].astype(str))
else:
    results = []
    completed = set()

for number, (path, actual) in enumerate(videos, start=1):
    if path in completed:
        continue

    try:
        features = get_features(path)

        if features is None:
            row = {
                "Video Path": path,
                "Actual Label": actual,
                "Predicted Label": "READ ERROR",
                "ASD Probability": np.nan,
                "TD Probability": np.nan
            }
        else:
            prediction = model.predict(
                np.expand_dims(features, axis=0),
                verbose=0
            )[0]

            asd = float(prediction[0])
            td = float(prediction[1])

            row = {
                "Video Path": path,
                "Actual Label": actual,
                "Predicted Label": "ASD" if asd > td else "TD",
                "ASD Probability": round(asd * 100, 2),
                "TD Probability": round(td * 100, 2)
            }

    except Exception as error:
        print("Error:", path, error)
        row = {
            "Video Path": path,
            "Actual Label": actual,
            "Predicted Label": "ERROR",
            "ASD Probability": np.nan,
            "TD Probability": np.nan
        }

    results.append(row)

    if len(results) % 10 == 0:
        pd.DataFrame(results).to_csv(OUTPUT, index=False)

    print(f"Processed {number}/{len(videos)}")

pd.DataFrame(results).to_csv(OUTPUT, index=False)

print("\nFinished processing videos.")
print("Results saved to:", os.path.abspath(OUTPUT))
print("Total recorded:", len(results))
