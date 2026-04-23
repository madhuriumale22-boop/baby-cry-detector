"""One-time script: Extract weights from model.h5 and save as model_weights.npz"""
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("model.h5")
model.summary()

weights = {}
for i, layer in enumerate(model.layers):
    w = layer.get_weights()
    if w:  # Skip layers with no weights (Dropout, Input)
        weights[f"layer_{i}_weights"] = w[0]
        weights[f"layer_{i}_bias"] = w[1]
        print(f"Layer {i} ({layer.name}): weights {w[0].shape}, bias {w[1].shape}")

np.savez("model_weights.npz", **weights)
print("\nSaved model_weights.npz successfully!")
