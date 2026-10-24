
import os
import glob
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

FEATURE_DIR = "proper_features_5frames"

def load_features(split):
    X, y = [], []

    for class_name, label in [("ASD", 0), ("TD", 1)]:
        folder = os.path.join(FEATURE_DIR, split, class_name)
        files = sorted(glob.glob(os.path.join(folder, "*.npy")))

        if not files:
            raise FileNotFoundError(
                f"No .npy feature files found in {folder}"
            )

        for file in files:
            features = np.load(file).astype(np.float32)

            if features.shape != (5, 1000):
                raise ValueError(
                    f"Unexpected shape {features.shape}: {file}"
                )

            X.append(features.reshape(-1))
            y.append(label)

        print(f"{split} {class_name}: {len(files)} videos")

    return np.asarray(X), np.asarray(y)


print("Loading existing features...")
X_train, y_train = load_features("train")
X_test, y_test = load_features("test")

print("Training shape:", X_train.shape)
print("Testing shape:", X_test.shape)

model = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, svd_solver="full")),
    ("svm", SVC(
        kernel="rbf",
        C=1.0,
        class_weight="balanced",
        random_state=42
    ))
])

print("\nTraining PCA + SVM...")
model.fit(X_train, y_train)

print("\nEvaluating model...")
predictions = model.predict(X_test)

print("PCA components:", model.named_steps["pca"].n_components_)
print("Accuracy:", round(accuracy_score(y_test, predictions) * 100, 2), "%")
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    labels=[0, 1],
    target_names=["ASD", "TD"],
    zero_division=0
))

joblib.dump(model, "ASD_PCA_SVM_model.pkl")
print("\nModel saved as ASD_PCA_SVM_model.pkl")
