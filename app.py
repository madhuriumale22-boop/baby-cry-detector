import streamlit as st
import numpy as np
import pandas as pd
import librosa
import librosa.display
import matplotlib.pyplot as plt
import io

from audio_utils import extract_features
from utils import load_trained_model, load_label_encoder, get_class_description, get_risk_color, format_confidence

# Must be the first Streamlit command
st.set_page_config(
    page_title="Baby Cry Detection",
    page_icon="👶",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Modern Parenting Theme (works in both light & dark modes)
st.markdown("""
<style>
    /* Headers */
    h1, h2, h3 {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 600;
    }
    
    /* Sidebar text — ensure visibility in both themes */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] li,
    [data-testid="stSidebar"] .stMarkdown {
        color: #2563eb !important;
        font-size: 14px;
    }
    
    /* Sidebar headers */
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #1e40af !important;
    }
    
    /* Sidebar caption (Disclaimer) */
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] caption {
        color: #dc2626 !important;
        font-size: 13px;
    }
    
    /* Upload Box */
    .stFileUploader > div > div {
        border: 2px dashed #6c5ce7;
        border-radius: 12px;
    }
    
    /* Upload button inside file uploader */
    .stFileUploader button {
        background-color: #6c5ce7 !important;
        color: #ffffff !important;
        border: 2px solid #6c5ce7 !important;
        border-radius: 8px;
        font-weight: 600;
    }
    .stFileUploader button:hover {
        background-color: #5b4bc4 !important;
        border-color: #5b4bc4 !important;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background-color: #6c5ce7 !important;
        color: #ffffff !important;
        border: 2px solid #6c5ce7 !important;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 600;
    }
    .stDownloadButton > button:hover {
        background-color: #5b4bc4 !important;
        border-color: #5b4bc4 !important;
        color: #ffffff !important;
    }
    
    /* Predict Button */
    .stButton > button {
        background-color: #6c5ce7;
        color: #ffffff;
        border: none;
        border-radius: 20px;
        padding: 10px 24px;
        font-weight: 600;
        font-size: 16px;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 4px 6px rgba(108, 92, 231, 0.2);
    }
    .stButton > button:hover {
        background-color: #5b4bc4;
        color: #ffffff;
        box-shadow: 0 6px 12px rgba(108, 92, 231, 0.3);
    }
    
    /* Result Cards */
    .result-card {
        padding: 30px;
        border-radius: 16px;
        margin-top: 20px;
        margin-bottom: 20px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 10px 20px rgba(0,0,0,0.1);
        transition: transform 0.2s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    /* Technical Details Expander */
    .streamlit-expanderHeader {
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

def plot_waveform_and_spectrogram(audio_bytes):
    """Generates waveform and spectrogram plots from uploaded audio bytes."""
    try:
        # Load audio from bytes using librosa (requires SoundFile back-end)
        audio, sr = librosa.load(io.BytesIO(audio_bytes), sr=22050)
        
        fig, ax = plt.subplots(2, 1, figsize=(10, 6))
        
        # Waveform
        librosa.display.waveshow(audio, sr=sr, ax=ax[0], color='#6c5ce7')
        ax[0].set_title('Audio Waveform', fontsize=12)
        ax[0].set_xlabel('Time (s)')
        ax[0].set_ylabel('Amplitude')
        
        # Spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        img = librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax[1], cmap='magma')
        ax[1].set_title('Spectrogram', fontsize=12)
        fig.colorbar(img, ax=ax[1], format="%+2.0f dB")
        
        plt.tight_layout()
        return fig, len(audio) / sr, sr
    except Exception as e:
        st.error(f"Could not generate visualizer: {e}")
        return None, 0, 0

def main():
    # --- Title & Subtitle ---
    st.title("👶 Baby Cry Detection")
    st.markdown("### Upload a baby cry recording and detect the likely reason using AI.")
    st.markdown("---")

    # --- Sidebar ---
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/822/822123.png", width=80)
        st.header("🍼 About the Project")
        st.write("This tool uses an Artificial Neural Network (ANN) trained on MFCC audio features to classify the reason behind a baby's cry.")
        
        st.header("📂 Supported Formats")
        st.write("`.wav`, `.mp3`, `.m4a`")
        
        st.header("📊 Model Status")
        model = load_trained_model("model_weights.npz")
        le = load_label_encoder("labels.pkl")
        
        if model and le:
            st.success("✅ Neural Network ready.")
        else:
            st.error("❌ Model not found! Please run `python train.py` first.")
            st.stop()
            
        st.header("⚠️ Disclaimer")
        st.caption("This tool is for educational and demonstrative purposes only. It is not a substitute for medical advice or parental intuition. Always consult a pediatrician if you are concerned about your baby's health.")

    # --- Main Input Area ---
    uploaded_file = st.file_uploader("Upload a recording of your baby crying...", type=["wav", "mp3", "m4a"])
    
    if uploaded_file is not None:
        st.audio(uploaded_file, format=f'audio/{uploaded_file.name.split(".")[-1]}')
        
        if st.button("🧠 Analyze Cry Reason"):
            with st.spinner("Analyzing acoustic features..."):
                # Save file temporarily or process bytes
                file_bytes = uploaded_file.read()
                
                # 1. Extract Features
                # We need to pass a file-like object to librosa.load
                # Since librosa.load supports soundfile which accepts BytesIO
                audio_io = io.BytesIO(file_bytes)
                features = extract_features(audio_io)
                
                if features is None:
                    st.error("Failed to extract features from the uploaded audio. Please ensure the file is a valid audio format.")
                    st.stop()
                
                # 2. Predict
                pred_proba = model.predict(features)
                pred_index = np.argmax(pred_proba)
                
                label_str = le.inverse_transform([pred_index])[0]
                confidence_dict = format_confidence(pred_proba, le.classes_)
                
                color = get_risk_color(label_str)
                description = get_class_description(label_str)
                confidence = confidence_dict[label_str]

                # Format Display Label
                display_label = label_str.replace("_", " ").title()

                # --- Results Display ---
                st.markdown(f"""
                <div class="result-card" style="background-color: {color};">
                    <h2 style="color: #ffffff !important; margin: 0; font-size: 32px;">{display_label}</h2>
                    <h4 style="color: rgba(255,255,255,0.9) !important; margin-top: 5px;">Confidence: {confidence:.1f}%</h4>
                    <p style="font-size: 18px; margin-top: 15px; font-weight: 500;">{description}</p>
                </div>
                """, unsafe_allow_html=True)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Probability Distribution")
                    labels = [c.replace("_", " ").title() for c in confidence_dict.keys()]
                    values = list(confidence_dict.values())
                    fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
                    bars = ax_bar.barh(labels[::-1], values[::-1], color='#6c5ce7', edgecolor='none', height=0.5)
                    ax_bar.set_xlabel("Probability (%)")
                    ax_bar.set_xlim(0, 100)
                    ax_bar.spines['top'].set_visible(False)
                    ax_bar.spines['right'].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_bar)
                        
                with col2:
                    st.subheader("Audio Visualizations")
                    # Needs a fresh BytesIO since it was consumed
                    fig, duration, sr = plot_waveform_and_spectrogram(file_bytes)
                    if fig:
                        st.pyplot(fig)
                
                # --- Advanced Details ---
                st.markdown("---")
                with st.expander("🛠️ Show Technical Details"):
                    st.write("**Audio Properties:**")
                    st.json({
                        "Filename": uploaded_file.name,
                        "Estimated Duration (s)": round(duration, 2),
                        "Sample Rate (Hz)": sr,
                        "Extracted MFCC Features": len(features),
                    })
                    
                    st.write("**Extracted Feature Vector (First 10 values):**")
                    st.code(str(features[:10]), language="python")

if __name__ == "__main__":
    main()
