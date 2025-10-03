import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.layers import Layer
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import MeanAbsoluteError
from sklearn.model_selection import train_test_split
import pandas as pd
import os

# ------------------------------
# PARAMETERS
# ------------------------------
IMAGE_DIR = "/"  # Adjust as needed
CSV_PATH = "./train.filename"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 150

# ------------------------------
# LOAD DATA
# ------------------------------
df = pd.read_csv(CSV_PATH)

# Option 1: Add an integer field to the CSV, e.g., "position"
if 'position' not in df.columns:
    df['position'] = 1  # Or some rule-based integer (0–9)

df['filepath'] = df['filename'].apply(lambda x: os.path.join(IMAGE_DIR, x))

train_files, test_files, train_labels, test_labels, train_positions, test_positions = train_test_split(
    df['filepath'].values,
    df['label'].values.astype('float32'),
    df['position'].values.astype('int32'),
    test_size=0.2,
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
# MODEL DEFINITION
# ------------------------------

# Tower 1: Image Input Tower
image_input = Input(shape=(224, 224, 3), name="image_input")
x1 = layers.Conv2D(128, (3, 3), activation='relu', name="relevance1")(image_input)
x1 = layers.MaxPooling2D((2, 2), name="relevance2")(x1)
x1 = layers.Conv2D(64, (3, 3), activation='relu', name="relevance3")(x1)
x1 = layers.MaxPooling2D((2, 2), name="relevance4")(x1)
x1 = layers.Conv2D(32, (3, 3), activation='relu', name="relevance5")(x1)
x1 = layers.MaxPooling2D((2, 2), name="relevance6")(x1)
x1 = layers.Flatten()(x1)
image_embedding = layers.Dense(64, activation='relu', name="relevance_image_embedding")(x1)

# Tower 2: Integer Input Tower
position_input = Input(shape=(1,), dtype='int32', name="position_input")
x2 = layers.Embedding(input_dim=10, output_dim=8)(position_input)  # vocab size 10
x2 = layers.Flatten()(x2)
position_embedding = layers.Dense(16, activation='relu', name="position_embedding")(x2)
# Mask: zero the output if input == 0
class PositionMask(Layer):
    def call(self, inputs):
        embedding, position = inputs
        mask = tf.cast(tf.not_equal(position, 0), tf.float32)  # (None, 1)
        return embedding * mask  # broadcasting to (None, 16)

masked_output = layers.Lambda(
    lambda x: tf.where(
        tf.equal(tf.cast(x[1], tf.int32), 0),  # check if position == 0
        tf.zeros_like(x[0]),                  # then zero the embedding
        x[0]                                  # else keep it
    ),
    output_shape=(16,)
)([position_embedding, position_input])
masked_output = PositionMask(name="masked_position")([position_embedding, position_input])

# Merge towers
combined = layers.Concatenate()([image_embedding, masked_output])
x = layers.Dense(64, activation='relu')(combined)
output = layers.Dense(1, name="output")(x)  # regression scalar output

# Build and compile model
model = Model(inputs=[image_input, position_input], outputs=output)
model.compile(
    optimizer='adam',
    loss='mean_squared_error',
    metrics=[MeanAbsoluteError()]
)

model.summary()

# ------------------------------
# TRAINING
# ------------------------------
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=25,
    restore_best_weights=True
)

model.fit(
    train_ds,
    validation_data=test_ds,
    epochs=EPOCHS,
    callbacks=[early_stop]
)

# ------------------------------
# EVALUATION
# ------------------------------
loss, mae = model.evaluate(test_ds)
print(f"Test MAE: {mae:.2f}")
model.save("regression_model_pal0.keras")
