# 👶 Baby Cry Detection System

A complete, production-ready machine learning application designed to detect the likely reason a baby is crying by analyzing the acoustic properties of the cry. Built with TensorFlow/Keras, Librosa, and Streamlit.

## 🌟 Project Overview

This project provides a parent-friendly web dashboard where users can upload an audio recording of their baby crying. The system processes the audio, extracts acoustic features, and uses an Artificial Neural Network (ANN) to predict whether the baby is:
- **Hungry**
- **Sleepy**
- **Experiencing Belly Pain**
- **In Discomfort**
- **Just Normal**

## 🚀 Features
- **Audio Processing**: Fast MFCC extraction using Librosa.
- **Machine Learning**: Deep learning ANN built with TensorFlow/Keras.
- **Beautiful UI**: Modern, parenting-themed Streamlit dashboard with probability charts and audio visualizers (Waveform & Spectrogram).
- **Zero-Touch Setup**: If you don't have a dataset, the system automatically generates synthetic dummy `.wav` files to allow immediate testing!

## 📂 Dataset Structure
To train the model on your own data, place `.wav`, `.mp3`, or `.m4a` files into the `dataset/` directory inside their respective class folders:
```text
dataset/
├── hungry/
├── sleepy/
├── belly_pain/
├── discomfort/
└── normal/
```
*(If empty, running `train.py` generates synthetic files to populate this structure).*

## 🧠 How It Works

### MFCC Extraction
When an audio file is uploaded, the system uses `librosa` to extract **Mel-Frequency Cepstral Coefficients (MFCCs)**. 
1. The audio is loaded and standardized to a fixed duration (5 seconds).
2. It extracts 40 MFCCs across the time domain.
3. The mean across the time axis is taken, producing a single flat feature vector of length 40. This perfectly represents the unique acoustic "signature" of the cry.

### ANN Architecture
The extracted 40-feature vector is fed into a Keras Sequential model:
- **Input Layer**: 40 neurons (MFCC features).
- **Hidden Layer 1**: 128 neurons (ReLU activation) + 30% Dropout to prevent overfitting.
- **Hidden Layer 2**: 64 neurons (ReLU activation) + 30% Dropout.
- **Output Layer**: 5 neurons (Softmax activation) corresponding to the probability of each of the 5 classes.
- Compiled with `Adam` optimizer and `sparse_categorical_crossentropy` loss.

## 🛠️ Installation & Local Usage

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd baby-cry-detector
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model:**
   ```bash
   python train.py
   ```
   *This will generate `model.h5`, `labels.pkl`, and training graphs.*

4. **Run the App:**
   ```bash
   streamlit run app.py
   ```

## ☁️ Deployment Instructions

Because TensorFlow can be heavy and memory-intensive, deploying to the free tier of Streamlit Community Cloud might fail due to memory limits. Below are robust deployment alternatives.

### Option 1: Render (Recommended)
1. Push this project to GitHub.
2. Go to [Render](https://render.com/) and create a new **Web Service**.
3. Connect your GitHub repository.
4. Set the following settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
5. Click **Create Web Service**.

### Option 2: Railway
1. Go to [Railway](https://railway.app/) and click **New Project** > **Deploy from GitHub repo**.
2. Railway will automatically detect the Python environment.
3. Add a Custom Start Command in Settings:
   `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
4. Deploy.

### Lightweight Alternative
If the TensorFlow `model.h5` is too large for your deployment host, you can convert it to **TensorFlow Lite**:
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
```
*(You will need to update `app.py` to use `tf.lite.Interpreter` to load and run inferences).*

## 📸 Screenshots

![Dashboard Placeholder](https://via.placeholder.com/800x400.png?text=Baby+Cry+Detector+Screenshot)
