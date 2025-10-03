import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf  # For tf.data
import pandas as pd
import os
import keras
from keras import layers,models
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping

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
    test_size=0.2,
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

# Define a simple CNN for regression
model = models.Sequential([
    layers.Input(shape=(224, 224, 3)),
    
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)  # Output: single regression value
])

model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=['mean_absolute_error']
)

early_stop = EarlyStopping(
    monitor='val_loss',       # You can monitor 'val_loss' or 'val_mean_absolute_error'
    patience=15,               # Number of epochs to wait before stopping after no improvement
    restore_best_weights=True # Restore model weights from the epoch with the best value
)

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=80,
    callbacks=[early_stop]
)

model.save("regression_modelc5.h5")

loss, mae = model.evaluate(test_ds)
print(f"\nTest Mean Absolute Error: {mae:.2f}")
