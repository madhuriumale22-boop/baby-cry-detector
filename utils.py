import os
import joblib
import tensorflow as tf

def load_trained_model(model_path="model.h5"):
    """Loads the trained Keras ANN model."""
    if os.path.exists(model_path):
        return tf.keras.models.load_model(model_path)
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
