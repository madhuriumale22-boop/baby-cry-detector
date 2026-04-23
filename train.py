import os
import glob
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from audio_utils import extract_features

# Configuration
DATASET_DIR = "dataset"
MODEL_PATH = "model.h5"
ENCODER_PATH = "labels.pkl"
EPOCHS = 30
BATCH_SIZE = 32
CLASSES = ["hungry", "sleepy", "belly_pain", "discomfort", "normal"]

def generate_synthetic_audio(directory, class_name, num_samples=20):
    """Generates synthetic dummy audio files for demonstration if no dataset is provided."""
    class_dir = os.path.join(directory, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    sr = 22050
    duration = 5 # 5 seconds
    t = np.linspace(0, duration, sr * duration, False)
    
    print(f"Generating {num_samples} synthetic audio files for class '{class_name}'...")
    for i in range(num_samples):
        # Generate some distinct noise/sine waves based on class
        if class_name == "hungry":
            # High frequency sine wave
            audio = 0.5 * np.sin(2 * np.pi * 440 * t) + np.random.normal(0, 0.1, len(t))
        elif class_name == "sleepy":
            # Low frequency
            audio = 0.5 * np.sin(2 * np.pi * 200 * t) + np.random.normal(0, 0.05, len(t))
        elif class_name == "belly_pain":
            # Modulated wave
            audio = 0.5 * np.sin(2 * np.pi * 600 * t) * np.sin(2 * np.pi * 2 * t)
        elif class_name == "discomfort":
            # White noise
            audio = np.random.normal(0, 0.5, len(t))
        else: # normal
            # Very quiet wave
            audio = 0.2 * np.sin(2 * np.pi * 300 * t)
            
        file_path = os.path.join(class_dir, f"synthetic_{i}.wav")
        sf.write(file_path, audio, sr)

def prepare_dataset():
    """Checks if dataset exists, generates synthetic if empty, and loads features."""
    if not os.path.exists(DATASET_DIR):
        os.makedirs(DATASET_DIR)
        
    # Check if dataset has subdirectories and files
    is_empty = True
    for cls in CLASSES:
        cls_dir = os.path.join(DATASET_DIR, cls)
        if not os.path.exists(cls_dir) or len(glob.glob(os.path.join(cls_dir, "*.wav"))) == 0:
            generate_synthetic_audio(DATASET_DIR, cls)
            is_empty = False
    
    if not is_empty:
        print("Generated synthetic dataset successfully.")
        
    X = []
    y = []
    
    print("\nExtracting features from audio files...")
    for cls in os.listdir(DATASET_DIR):
        cls_dir = os.path.join(DATASET_DIR, cls)
        if os.path.isdir(cls_dir):
            for file_name in os.listdir(cls_dir):
                if file_name.endswith((".wav", ".mp3", ".m4a")):
                    file_path = os.path.join(cls_dir, file_name)
                    features = extract_features(file_path)
                    if features is not None:
                        X.append(features)
                        y.append(cls)
                        
    return np.array(X), np.array(y)

def build_model(input_shape, num_classes):
    """Builds the Artificial Neural Network architecture."""
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=input_shape),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def plot_history(history):
    """Plots and saves training history."""
    # Plot Accuracy
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.savefig('training_accuracy.png')
    plt.close()
    
    # Plot Loss
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.savefig('training_loss.png')
    plt.close()
    print("Saved training_accuracy.png and training_loss.png")

def main():
    # 1. Load data
    X, y = prepare_dataset()
    if len(X) == 0:
        print("No audio files found. Exiting.")
        return
        
    print(f"Total valid audio files loaded: {len(X)}")
    
    # 2. Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)
    
    joblib.dump(le, ENCODER_PATH)
    print(f"Saved label encoder to {ENCODER_PATH}. Classes: {le.classes_}")
    
    # 3. Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)
    
    # 4. Build and train model
    model = build_model(input_shape=(X_train.shape[1],), num_classes=num_classes)
    model.summary()
    
    print("\nStarting training...")
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_split=0.2,
        verbose=1
    )
    
    # 5. Evaluate
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"\nFinal Test Accuracy: {accuracy * 100:.2f}%")
    
    # 6. Save Model & Plots
    model.save(MODEL_PATH)
    print(f"Saved trained model to {MODEL_PATH}")
    
    plot_history(history)
    print("\nTraining Pipeline Completed Successfully.")

if __name__ == "__main__":
    main()
