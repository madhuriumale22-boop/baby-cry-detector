import librosa
import numpy as np

def extract_features(file_path, duration=5, sr=22050, n_mfcc=40):
    """
    Loads an audio file, standardizes its length, and extracts MFCC features.
    
    Args:
        file_path (str or file-like object): Path to the audio file.
        duration (int): Standard duration to pad/truncate to (in seconds).
        sr (int): Sample rate.
        n_mfcc (int): Number of MFCCs to extract.
        
    Returns:
        np.ndarray: A 1D array of extracted MFCC features.
    """
    try:
        # Load audio file
        audio, _ = librosa.load(file_path, sr=sr, duration=duration)
        
        # Calculate target length in samples
        target_length = duration * sr
        
        # Standardize length: Pad or Truncate
        if len(audio) < target_length:
            # Pad with zeros (silence)
            audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')
        else:
            # Truncate
            audio = audio[:target_length]
            
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        
        # Take the mean across the time axis to return a fixed-length vector (n_mfcc,)
        mfcc_mean = np.mean(mfcc.T, axis=0)
        
        return mfcc_mean
        
    except Exception as e:
        print(f"Error processing audio file {file_path}: {e}")
        return None
