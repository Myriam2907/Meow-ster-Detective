import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from sklearn.model_selection import train_test_split

# ------------------------------
# PARAMETERS
# ------------------------------
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
LEARNING_RATE = 1e-4
DATA_DIR = "cats-dataset"  # Your original dataset folder with 4 cat folders
RESULTS_FILE = "augmentation_comparison.csv"
VAL_SPLIT = 0.2

# ------------------------------
# Create temporary train/val folders
# ------------------------------
TEMP_DIR = "cats_temp"
train_dir = os.path.join(TEMP_DIR, "train")
val_dir = os.path.join(TEMP_DIR, "val")

# Remove old temp folder if exists
if os.path.exists(TEMP_DIR):
    shutil.rmtree(TEMP_DIR)

os.makedirs(train_dir)
os.makedirs(val_dir)

# Split images
cat_classes = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
for cat in cat_classes:
    cat_path = os.path.join(DATA_DIR, cat)
    images = [f for f in os.listdir(cat_path) if f.lower().endswith(('.jpg','.png'))]
    
    train_imgs, val_imgs = train_test_split(images, test_size=VAL_SPLIT, random_state=42)
    
    # Create class folders
    os.makedirs(os.path.join(train_dir, cat))
    os.makedirs(os.path.join(val_dir, cat))
    
    # Copy files
    for img in train_imgs:
        shutil.copy(os.path.join(cat_path, img), os.path.join(train_dir, cat, img))
    for img in val_imgs:
        shutil.copy(os.path.join(cat_path, img), os.path.join(val_dir, cat, img))

print("Dataset split completed: train and val folders created.")

# ------------------------------
# AUGMENTATION PIPELINES
# ------------------------------
augmentation_dict = {
    "None": ImageDataGenerator(preprocessing_function=None, rescale=1./255),
    "Rotation": ImageDataGenerator(rotation_range=30, rescale=1./255),
    "Flip": ImageDataGenerator(horizontal_flip=True, rescale=1./255),
    "Brightness": ImageDataGenerator(brightness_range=[0.7,1.3], rescale=1./255),
    "Combined": ImageDataGenerator(
        rotation_range=30,
        horizontal_flip=True,
        brightness_range=[0.7,1.3],
        zoom_range=0.2,
        rescale=1./255
    )
}

# ------------------------------
# Train and evaluate
# ------------------------------
results = []

for aug_name, datagen in augmentation_dict.items():
    print(f"\n--- Training with augmentation: {aug_name} ---\n")
    
    train_gen = datagen.flow_from_directory(
        train_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
        val_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Build model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(train_gen.num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    
    for layer in base_model.layers:
        layer.trainable = False
    
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[early_stop],
        verbose=1
    )
    
    # Evaluate
    val_gen.reset()
    y_true = val_gen.classes
    y_pred_probs = model.predict(val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"{aug_name} -> Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}")
    
    results.append({
        "Augmentation": aug_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    })

# Save results
df = pd.DataFrame(results)
df.to_csv(RESULTS_FILE, index=False)
print(f"\nResults saved to {RESULTS_FILE}")

# Plot comparison
plt.figure(figsize=(8,5))
plt.bar(df['Augmentation'], df['Accuracy'], color='skyblue')
plt.ylabel("Accuracy")
plt.title("Model Accuracy Comparison Across Augmentation Techniques")
plt.ylim(0,1)
for i, v in enumerate(df['Accuracy']):
    plt.text(i, v+0.01, f"{v:.2f}", ha='center')
plt.show()
