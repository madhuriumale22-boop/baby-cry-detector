import os
import joblib
import numpy as np

def _relu(x):
    """ReLU activation function."""
    return np.maximum(0, x)

def _softmax(x):
    """Softmax activation function."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

class NumpyANN:
    """A simple feed-forward ANN using only NumPy for inference.
    Replicates the Keras Sequential model: Dense(128,relu) -> Dense(64,relu) -> Dense(5,softmax)
    """
    def __init__(self, weights_path):
        data = np.load(weights_path)
        # Layer 0: Dense(128, relu) — input 40 -> 128
        self.w1 = data["layer_0_weights"]
        self.b1 = data["layer_0_bias"]
        # Layer 2: Dense(64, relu) — 128 -> 64
        self.w2 = data["layer_2_weights"]
        self.b2 = data["layer_2_bias"]
        # Layer 4: Dense(5, softmax) — 64 -> 5
        self.w3 = data["layer_4_weights"]
        self.b3 = data["layer_4_bias"]
    
    def predict(self, x):
        """Run forward pass. x shape: (num_features,) or (1, num_features)"""
        if x.ndim == 2:
            x = x[0]
        # Forward pass (Dropout is no-op during inference)
        h1 = _relu(x @ self.w1 + self.b1)
        h2 = _relu(h1 @ self.w2 + self.b2)
        out = _softmax(h2 @ self.w3 + self.b3)
        return out

def load_trained_model(weights_path="model_weights.npz"):
    """Loads the trained ANN model from NumPy weights."""
    if os.path.exists(weights_path):
        return NumpyANN(weights_path)
    return None

def load_label_encoder(encoder_path="labels.pkl"):
    """Loads the fitted LabelEncoder."""
    if os.path.exists(encoder_path):
        return joblib.load(encoder_path)
    return None

def get_class_description(label):
    """Returns a parent-friendly description for the predicted cry reason."""
    descriptions = {
        "hungry": "Baby may need feeding. Check if it's time for a meal.",
        "sleepy": "Baby may be tired and needs rest. Try soothing them to sleep.",
        "belly_pain": "Baby may be uncomfortable due to stomach pain or gas. Gentle burping might help.",
        "discomfort": "Baby may need attention, perhaps a diaper change or a change in clothing/temperature.",
        "normal": "Cry does not indicate a specific medical or immediate issue. Often just normal vocalization."
    }
    # Fallback if label is slightly different
    return descriptions.get(label.lower(), "Unknown reason. Please observe your baby closely.")

def get_risk_color(label):
    """Returns the CSS color associated with a specific class."""
    colors = {
        "hungry": "#FFB347",      # Orange/Yellow
        "sleepy": "#4da6ff",      # Blue
        "belly_pain": "#ff4d4d",  # Red
        "discomfort": "#b366ff",  # Purple
        "normal": "#4dff4d"       # Green
    }
    return colors.get(label.lower(), "#808080")

def format_confidence(probabilities, classes):
    """
    Formats the probability array into a readable dictionary.
    Args:
        probabilities (np.ndarray): Array of probabilities.
        classes (list): List of class names.
    Returns:
        dict: Mapping of class name to probability percentage.
    """
    conf_dict = {}
    for i, cls in enumerate(classes):
        # Convert to percentage
        conf_dict[cls] = float(probabilities[i]) * 100
    
    # Sort by probability descending
    conf_dict = dict(sorted(conf_dict.items(), key=lambda item: item[1], reverse=True))
    return conf_dict
