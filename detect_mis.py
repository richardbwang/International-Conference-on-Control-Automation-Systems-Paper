import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import MeanAbsoluteError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Layer
import pandas as pd
import os
import numpy as np

# ------------------------------
# LOAD THE SAVED MODEL
# ------------------------------
class PositionMask(Layer):
    def call(self, inputs):
        embedding, position = inputs
        mask = tf.cast(tf.not_equal(position, 0), tf.float32)  # (None, 1)
        return embedding * mask  # broadcasting to (None, 16)
model = load_model("regression_model_palall.keras", custom_objects={'PositionMask': PositionMask}, compile=False)

# ------------------------------
# FREEZE IMAGE/RELEVANCE TOWER
# ------------------------------
for layer in model.layers:
    if layer.name.startswith("relevance") or layer.name == "image_embedding":
        layer.trainable = False

# ------------------------------
# RECOMPILE MODEL
# ------------------------------
model.compile(
    optimizer=Adam(),
    loss=MeanSquaredError(),
    metrics=[MeanAbsoluteError()]
)


# ------------------------------
# PARAMETERS
# ------------------------------
IMAGE_DIR = "/"  # Adjust as needed
CSV_PATH = "./train.filenamecmed"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 300

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(CSV_PATH)

# Option 1: Add an integer field to the CSV, e.g., "position"
#if 'position' not in df.columns:
#    df['position'] = 0  # Or some rule-based integer (0–9)

df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

test_files = df['filepath'].values
test_labels = df['label'].values.astype('float32')
test_positions = df['position'].values.astype('int32')

# ------------------------------
# TF DATASET LOADING
# ------------------------------
def load_data(image_path, label, position):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, IMG_SIZE)
    image = image / 255.0
    return {
        "image_input": image,
        "position_input": position
    }, label

test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels, test_positions))
test_ds = test_ds.map(load_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ------------------------------
# OPTIONAL: RE-EVALUATE
# ------------------------------
# (Assumes you already have test_ds prepared)
#loss, mae = model.evaluate(test_ds)
#print(f"Test MAE after freezing image tower: {mae:.2f}")

# ------------------------------
# OPTIONAL: CONTINUE TRAINING
# ------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

#model.fit(
#    train_ds,
#    validation_data=test_ds,
#    epochs=EPOCHS,
#    callbacks=[early_stop]
#)

# ------------------------------
# EVALUATION
# ------------------------------
#loss, mae = model.evaluate(test_ds)
#print(f"Test MAE: {mae:.2f}")
#model.save("regression_model_palall.keras")

def count_dataset_elements(ds):
    count = 0
    for _ in ds:
        count += 1
    return count

def replace_position(inputs, new_pos):
    """
    inputs: dict with keys "image_input" and "position_input"
    new_pos: scalar int to replace all positions
    """
    images = inputs["image_input"]
    batch_size = tf.shape(images)[0]
    new_positions = tf.fill([batch_size], tf.cast(new_pos, dtype=inputs["position_input"].dtype))
    return {
        "image_input": images,
        "position_input": new_positions
    }

def mean_absolute_error_np(y_true, y_pred):
    assert len(y_true) == len(y_pred), "Lists must have the same length"
    n = len(y_true)
    total_abs_error = sum(abs(t - p) for t, p in zip(y_true, y_pred))
    return float(total_abs_error / n) if n > 0 else float(-1.0)

test_ds_unbatched = test_ds.unbatch()
window_size = 3
total_aligned = 0
total_sample = 0
for orig_pos in range(1, 7):
    # Filter dataset by original position, unbatch to list
    filtered = test_ds_unbatched.filter(
        lambda x, y: tf.equal(x["position_input"], tf.constant(orig_pos, dtype=tf.int32))
    )

    # Convert to list of (inputs, labels)
    samples = list(filtered.as_numpy_iterator())

    num_samples = len(samples)
    if num_samples < window_size:
        print(f"Position {orig_pos}: less than {window_size} samples, skipping sliding window.")
        continue

    print(f"\nOriginal position {orig_pos}, total samples: {num_samples}")

    aligned_count = 0
    total_windows = num_samples - window_size + 1

    for start_idx in range(total_windows):
        window_samples = samples[start_idx : start_idx + window_size]

        # Separate inputs and labels
        inputs_batch = {
            "image_input": np.stack([s[0]["image_input"] for s in window_samples]),
            "position_input": np.array([s[0]["position_input"] for s in window_samples])
        }
        labels_batch = np.array([s[1] for s in window_samples])

        mae_per_pos = []
        # For each replacement position 1 to 6
        for pos_replace in range(1, 7):
            # Replace position_input with pos_replace
            new_positions = np.full(window_size, pos_replace, dtype=inputs_batch["position_input"].dtype)
            replaced_inputs = {
                "image_input": inputs_batch["image_input"],
                "position_input": new_positions
            }

            preds = model.predict(replaced_inputs, verbose=0).flatten()
            true_label_first_sample = window_samples[0][1]
            mae = np.mean(np.abs(preds - true_label_first_sample))
#            mae = np.mean(np.abs(preds - labels_batch))
            mae_per_pos.append(mae)

        # Find best replacement
        best_replacement_idx = int(np.argmin(mae_per_pos)) + 1
        best_mae = mae_per_pos[best_replacement_idx - 1]
        is_aligned = (best_replacement_idx == orig_pos)
        if is_aligned:
            aligned_count += 1
            total_aligned += 1
        total_sample += 1

#        print(f"Window {start_idx+1}-{start_idx+window_size}: Best replacement pos {best_replacement_idx} with MAE {best_mae:.4f} - Aligned? {'Yes' if is_aligned else 'No'}")

    ratio_aligned = aligned_count / total_windows if total_windows > 0 else 0.0
    print(f"Position {orig_pos}: Aligned ratio = {ratio_aligned:.3f} ({aligned_count}/{total_windows})")
ratio_total = total_aligned / total_sample
print(f"{ratio_total:.3f}")
