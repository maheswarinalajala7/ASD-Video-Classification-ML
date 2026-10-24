
import os
import glob
import cv2
import numpy as np
import tensorflow as tf

from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import Sequence
from sklearn.model_selection import train_test_split

# =========================
# SETTINGS
# =========================

TRAIN_DIR = "very_large_data/autism_data/training_set"
TEST_DIR = "very_large_data/autism_data/testing_set"

# Use a new folder to preserve the existing 207 feature files.
FEATURE_DIR = "proper_features_5frames"

NUM_FRAMES = 5
FEATURE_SIZE = 1000
BATCH_SIZE = 8
EPOCHS = 3

os.makedirs(FEATURE_DIR, exist_ok=True)

# =========================
# LOAD VGG16
# =========================

print("Loading VGG16...", flush=True)

vgg = VGG16(
    weights="imagenet",
    include_top=True
)

print("VGG16 loaded.", flush=True)


# =========================
# EXTRACT 5 FRAMES
# =========================

def extract_features(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Cannot open:", video_path, flush=True)
        return None

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total <= 0:
        cap.release()
        return None

    indices = np.linspace(
        0, total - 1, NUM_FRAMES, dtype=int
    )

    frames = []

    for index in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        success, frame = cap.read()

        if not success:
            continue

        frame = cv2.resize(frame, (224, 224))
        frames.append(frame)

    cap.release()

    if not frames:
        return None

    images = np.stack(frames).astype(np.float32)
    images = preprocess_input(images)

    # Process sampled frames together.
    features = vgg.predict(images, batch_size=NUM_FRAMES, verbose=0)

    # Keep exactly 5 time steps.
    result = np.zeros(
        (NUM_FRAMES, FEATURE_SIZE), dtype=np.float32
    )

    result[:len(features)] = features

    return result


# =========================
# FIND VIDEOS
# =========================

def get_videos(folder):

    videos = []

    for label, class_name in [(0, "ASD"), (1, "TD")]:
        class_dir = os.path.join(folder, class_name)

        files = glob.glob(
            os.path.join(class_dir, "*.mp4")
        )

        videos.extend((path, label) for path in files)

    videos.sort(key=lambda item: item[0])

    return videos


train_videos = get_videos(TRAIN_DIR)
test_videos = get_videos(TEST_DIR)

print("Training videos:", len(train_videos), flush=True)
print("Testing videos:", len(test_videos), flush=True)

if not train_videos or not test_videos:
    raise RuntimeError(
        "Videos not found. Check TRAIN_DIR and TEST_DIR."
    )


# =========================
# CACHE FEATURES TO DISK
# =========================

def feature_path(video_path, split):

    class_name = os.path.basename(
        os.path.dirname(video_path)
    )

    filename = os.path.splitext(
        os.path.basename(video_path)
    )[0] + ".npy"

    folder = os.path.join(
        FEATURE_DIR, split, class_name
    )

    os.makedirs(folder, exist_ok=True)

    return os.path.join(folder, filename)


def prepare_features(videos, split):

    records = []
    total = len(videos)

    for i, (video, label) in enumerate(videos, start=1):

        saved = feature_path(video, split)

        valid_cache = False

        if os.path.exists(saved):
            try:
                cached = np.load(saved, mmap_mode="r")
                valid_cache = cached.shape == (
                    NUM_FRAMES, FEATURE_SIZE
                )
            except (OSError, ValueError):
                valid_cache = False

        if not valid_cache:

            features = extract_features(video)

            if features is None:
                print(
                    "Skipping unreadable video:", video,
                    flush=True
                )
                continue

            np.save(saved, features)

        records.append((saved, label))

        if i % 25 == 0 or i == total:
            print(
                f"{split}: {i}/{total} videos checked; "
                f"usable: {len(records)}",
                flush=True
            )

    return records


print("\nPreparing training features...", flush=True)
train_records = prepare_features(train_videos, "train")

print("\nPreparing testing features...", flush=True)
test_records = prepare_features(test_videos, "test")

if not train_records or not test_records:
    raise RuntimeError("No usable features were created.")


# =========================
# BATCHED DATA LOADER
# =========================

class VideoSequence(Sequence):

    def __init__(self, records, batch_size=8, shuffle=False):
        super().__init__()
        self.records = list(records)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.order = np.arange(len(self.records))
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.records) / self.batch_size))

    def __getitem__(self, index):

        indices = self.order[
            index * self.batch_size:
            (index + 1) * self.batch_size
        ]

        batch_x = []
        batch_y = []

        for i in indices:
            path, label = self.records[i]
            feature = np.load(path)

            batch_x.append(feature)
            batch_y.append(label)

        return (
            np.asarray(batch_x, dtype=np.float32),
            np.asarray(batch_y, dtype=np.int32)
        )

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.order)


# Use part of the training set for validation.
paths = [record[0] for record in train_records]
labels = [record[1] for record in train_records]

train_part, val_part = train_test_split(
    train_records,
    test_size=0.10,
    random_state=42,
    stratify=labels
)

train_loader = VideoSequence(
    train_part, BATCH_SIZE, shuffle=True
)

val_loader = VideoSequence(
    val_part, BATCH_SIZE
)

test_loader = VideoSequence(
    test_records, BATCH_SIZE
)


# =========================
# BUILD VGG16 + LSTM MODEL
# =========================

model = Sequential([
    tf.keras.Input(shape=(NUM_FRAMES, FEATURE_SIZE)),
    LSTM(64),
    Dense(32, activation="relu"),
    Dropout(0.3),
    Dense(2, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()


# =========================
# TRAIN
# =========================

print("\nStarting training...", flush=True)

model.fit(
    train_loader,
    validation_data=val_loader,
    epochs=EPOCHS,
    verbose=1
)


# =========================
# FINAL TEST
# =========================

print("\nEvaluating on separate test set...", flush=True)

loss, accuracy = model.evaluate(
    test_loader, verbose=1
)

print("\nFinal test loss:", loss)
print("Final test accuracy:", accuracy)


# =========================
# SAVE MODEL
# =========================

model.save("ASD_VGG16_LSTM_model.keras")

print(
    "\nModel saved as ASD_VGG16_LSTM_model.keras",
    flush=True
)
