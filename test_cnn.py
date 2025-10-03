import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf  # For tf.data
import pandas as pd
import os
import keras
from keras import layers,models
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import EarlyStopping

model = load_model("regression_model_200.h5")

# Parameters
IMAGE_DIR = "/"
CSV_PATH = "./train.filenamec5"
IMG_SIZE = (224, 224)  # Adjust as needed
BATCH_SIZE = 32

# Load CSV
df = pd.read_csv(CSV_PATH)
df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

# --- 2. Split filenames and labels into train/test ---
train_files, test_files, train_labels, test_labels = train_test_split(
    df['filepath'].values,
    df['label'].values.astype('float32'),
    test_size=0.9,
    random_state=42
)

# --- 3. Build TF datasets for train/test ---

def load_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0  # normalize
    return image, label

train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
train_ds = train_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
test_ds = test_ds.map(load_image).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

loss, mae = model.evaluate(test_ds)

print(f"\nTest Mean Absolute Error: {mae:.2f}")

calib_sum = 0
i=0
# Iterate through the test dataset
for batch_images, batch_labels in test_ds:
    # Predict
    batch_preds = model.predict(batch_images)

    # Flatten if needed
    batch_preds = batch_preds.flatten()
    batch_labels = batch_labels.numpy().flatten()

    # Print error per sample
    for true_val, pred_val in zip(batch_labels, batch_preds):
        error = pred_val - true_val  # signed error
        print(f"{i:.1f}, True = {true_val:.4f}, Predicted = {pred_val:.4f}, Error = {error:.4f}")
        calib_sum += error
        i = i+1
calib = calib_sum / i
print(f"calib={calib:.2f},{calib_sum:.2f},{len(batch_labels):.1f}")

error_sum = 0
error_sum_orig = 0
for batch_images, batch_labels in test_ds:
    # Predict
    batch_preds = model.predict(batch_images)

    # Flatten if needed
    batch_preds = batch_preds.flatten()
    batch_labels = batch_labels.numpy().flatten()

    # Print error per sample
    for true_val, pred_val in zip(batch_labels, batch_preds):
        error_sum += abs(pred_val - true_val - calib)
        error_sum_orig += abs(pred_val - true_val)
print(error_sum/i)
print(error_sum_orig/i)
