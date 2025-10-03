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

# ------------------------------
# LOAD THE SAVED MODEL
# ------------------------------
class PositionMask(Layer):
    def call(self, inputs):
        embedding, position = inputs
        mask = tf.cast(tf.not_equal(position, 0), tf.float32)  # (None, 1)
        return embedding * mask  # broadcasting to (None, 16)
model = load_model("regression_model_pal0.keras", custom_objects={'PositionMask': PositionMask}, compile=False)

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
CSV_PATH = "./train.filenamecall"
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

train_files, test_files, train_labels, test_labels, train_positions, test_positions = train_test_split(
    df['filepath'].values,
    df['label'].values.astype('float32'),
    df['position'].values.astype('int32'),
    test_size=0.95,
    random_state=42
)

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

train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels, train_positions))
train_ds = train_ds.map(load_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels, test_positions))
test_ds = test_ds.map(load_data).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ------------------------------
# OPTIONAL: RE-EVALUATE
# ------------------------------
# (Assumes you already have test_ds prepared)
loss, mae = model.evaluate(test_ds)
print(f"Test MAE after freezing image tower: {mae:.2f}")

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

# Create filtered datasets for each position value (1 to 6)
test_ds_by_position = {}
test_ds_unbatched = test_ds.unbatch()
for i in range(1, 7):
    test_ds_by_position[i] = test_ds_unbatched.filter(
        lambda x, y: tf.equal(x["position_input"], i)
    )
    sample_count = count_dataset_elements(test_ds_by_position[i])
    print(f"Position {i}: {sample_count} samples")

    test_ds_by_position[i] = test_ds_by_position[i].batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
    loss, mae = model.evaluate(test_ds_by_position[i])


    calib_sum = 0
    calib_i=0
    # Iterate through the test dataset
    for batch_images, batch_labels in test_ds_by_position[i]:
        # Predict
        batch_preds = model.predict(batch_images)

        # Flatten if needed
        batch_preds = batch_preds.flatten()
        batch_labels = batch_labels.numpy().flatten()

        # Print error per sample
        for true_val, pred_val in zip(batch_labels, batch_preds):
            error = pred_val - true_val  # signed error
            calib_sum += error
            calib_i = calib_i+1
    calib = calib_sum / i
    print(f"calib={calib:.2f},{calib_sum:.2f},{len(batch_labels):.1f}")

    error_sum = 0
    error_sum_orig = 0
    for batch_images, batch_labels in test_ds_by_position[i]:
        # Predict
        batch_preds = model.predict(batch_images)

        # Flatten if needed
        batch_preds = batch_preds.flatten()
        batch_labels = batch_labels.numpy().flatten()

        # Print error per sample
        for true_val, pred_val in zip(batch_labels, batch_preds):
            error_sum += abs(pred_val - true_val - calib)
            error_sum_orig += abs(pred_val - true_val)
    print(error_sum/calib_i)
    print(error_sum_orig/calib_i)
    print(f"Test MAE: {mae:.2f}")
