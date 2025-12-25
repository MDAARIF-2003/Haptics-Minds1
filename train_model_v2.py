print("--- SCRIPT IS RUNNING ---")
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# --- 1. CONFIGURATION ---
IMG_SIZE = 64
BATCH_SIZE = 32

# --- UPDATE THESE PATHS ---
# This assumes your data is in:
# D:\Vision Beyond Sight\dataset\train
# D:\Vision Beyond Sight\dataset\validation
BASE_PATH = r"D:\Vision Beyond Sight\dataset" 
TRAIN_DATA_PATH = os.path.join(BASE_PATH, "train")
VALID_DATA_PATH = os.path.join(BASE_PATH, "validation")

# Check if paths exist
if not os.path.exists(TRAIN_DATA_PATH):
    print(f"[ERROR] Training path not found: {TRAIN_DATA_PATH}")
    print("Please make sure your 'train' folder is inside 'D:\\Vision Beyond Sight\\dataset\\'")
    exit()
if not os.path.exists(VALID_DATA_PATH):
    print(f"[ERROR] Validation path not found: {VALID_DATA_PATH}")
    print("Please make sure your 'validation' folder is inside 'D:\\Vision Beyond Sight\\dataset\\'")
    exit()

# --- 2. DATA AUGMENTATION ---
print("[INFO] Setting up Data Augmentation generators...")

# This is the "magic" that creates fake lighting and angles
train_datagen = ImageDataGenerator(
    rescale=1./255,            # Normalize pixel values
    rotation_range=20,         # Randomly rotate
    width_shift_range=0.2,     # Randomly shift horizontally
    height_shift_range=0.2,    # Randomly shift vertically
    shear_range=0.2,           # "Stretch" the image
    zoom_range=0.2,            # Randomly zoom in
    horizontal_flip=True,      # Randomly flip
    fill_mode='nearest',
    brightness_range=[0.5, 1.5]  # <-- THIS IS THE CRITICAL FIX (dark & bright)
)

# The validation data should NOT be augmented, only rescaled
validation_datagen = ImageDataGenerator(rescale=1./255)

# --- 3. LOAD DATA ---
print("[INFO] Loading data from directories...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Assumes 5 classes (checks, dots, etc.)
)

validation_generator = validation_datagen.flow_from_directory(
    VALID_DATA_PATH,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Get the number of classes
num_classes = len(train_generator.class_indices)
print(f"[INFO] Found {train_generator.samples} training images belonging to {num_classes} classes.")
print(f"[INFO] Found {validation_generator.samples} validation images.")
if num_classes != 5:
    print(f"[WARNING] You have {num_classes} classes, but the project is built for 5. Make sure this is correct!")


# --- 4. DEFINE A MORE ROBUST MODEL ---
print("[INFO] Building a robust CNN model...")
model = Sequential([
    # Block 1
    Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flatten and Dense Layers
    Flatten(),
    Dense(512, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),
    Dense(num_classes, activation='softmax') # Use the auto-detected number of classes
])

model.summary()

# --- 5. COMPILE AND TRAIN ---
print("[INFO] Compiling model...")
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

print("[INFO] Starting training... (This will take 15-30 minutes)")
EPOCHS = 50 

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# --- 6. SAVE THE NEW, SMARTER MODEL ---
print("[INFO] Training complete. Saving model as 'pattern_model_v2.h5'...")
model.save("pattern_model_v2.h5")
print("[INFO] Done!")