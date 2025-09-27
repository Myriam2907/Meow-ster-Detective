import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam

# ------------------------------
# PARAMETERS
# ------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 16
LEARNING_RATE = 1e-4
EPOCHS = 20

DATA_DIR = 'cats-dataset'  # path to your 4 cat folders

# ------------------------------
# DATA AUGMENTATION WITH TRAIN/VAL SPLIT
# ------------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode='nearest',
    validation_split=0.2  # 20% for validation
)

train_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training',  # set as training data
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation',  # set as validation data
    shuffle=False
)

NUM_CLASSES = train_generator.num_classes
print(f"Detected {NUM_CLASSES} classes: {list(train_generator.class_indices.keys())}")

# ------------------------------
# BUILD MODEL
# ------------------------------
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
output = Dense(NUM_CLASSES, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# ------------------------------
# TRAINING
# ------------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS
)

# ------------------------------
# FINE-TUNING (OPTIONAL)
# ------------------------------
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

history_finetune = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=10
)

# ------------------------------
# SAVE MODEL
# ------------------------------
os.makedirs("models", exist_ok=True)
model.save("models/meow_recognizer.h5")
print("Model saved to models/meow_recognizer.h5")

# ------------------------------
# PREDICTION FUNCTION
# ------------------------------
from tensorflow.keras.preprocessing import image

def predict_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    pred = model.predict(img_array)
    pred_class = np.argmax(pred)
    pred_confidence = pred[0][pred_class]
    
    class_labels = list(train_generator.class_indices.keys())
    print(f"Predicted: {class_labels[pred_class]}, Confidence: {pred_confidence:.2f}")

# Example usage
# predict_image('cats-dataset/Cat1/example.jpg')
