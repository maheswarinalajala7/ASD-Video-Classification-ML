
import os
import glob
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix

model = load_model("ASD_VGG16_LSTM_model.keras")

X, y = [], []

for label, name in [(0, "ASD"), (1, "TD")]:
    folder = f"proper_features_5frames/test/{name}"

    for file in glob.glob(os.path.join(folder, "*.npy")):
        X.append(file)
        y.append(label)

print("Test videos:", len(X))

predictions = []

for i in range(0, len(X), 32):
    batch = np.array([np.load(f) for f in X[i:i+32]])
    predictions.extend(np.argmax(model.predict(batch, verbose=0), axis=1))

print("\nConfusion Matrix:")
print(confusion_matrix(y, predictions))

print("\nClassification Report:")
print(classification_report(
    y, predictions, target_names=["ASD", "TD"], zero_division=0
))
