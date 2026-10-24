
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score

model = load_model("ASD_VGG16_LSTM_model.keras")

X, y = [], []

for label, folder in enumerate(["ASD", "TD"]):
    import glob, os
    files = sorted(glob.glob(f"proper_features_5frames/test/{folder}/*.npy"))

    for file in files:
        X.append(np.load(file))
        y.append(label)

X = np.array(X)
y = np.array(y)

pred = model.predict(X, batch_size=32, verbose=1)
classes = np.argmax(pred, axis=1)

print("Actual ASD:", np.sum(y == 0))
print("Actual TD:", np.sum(y == 1))
print("Predicted ASD:", np.sum(classes == 0))
print("Predicted TD:", np.sum(classes == 1))
print("Accuracy:", accuracy_score(y, classes))
