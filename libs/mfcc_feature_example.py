import librosa
import numpy as np
import python_speech_features

def extract_audio_features(audio_path, sample_rate=16000, num_mfcc=13, win_length=0.025, hop_length=0.01):
    """
    Extract MFCC features from an audio file.
    
    MFCC (Mel-frequency cepstral coefficients) can quickly help with code testing 
    without needing to install many environments. The output shape of audio_feature 
    will be (T, 39), where T is the number of time steps. Note that this feature 
    is not robust and is only suitable for early code testing.

    For detailed usage, refer to mfcc_feature_example.py in the libs directory.
    
    Parameters:
    audio_path (str): Path to the audio file
    sample_rate (int): Sampling rate, default is 16000
    num_mfcc (int): Number of MFCC features, default is 13
    win_length (float): Window length in seconds, default is 0.025
    hop_length (float): Hop length in seconds, default is 0.01
    
    Returns:
    numpy.ndarray: Feature matrix containing MFCC features and their first and second order deltas
    """
    # Load the audio file
    wav, sr = librosa.load(audio_path, sr=sample_rate)
    
    # Extract MFCC features
    mfcc_features = python_speech_features.mfcc(
        signal=wav, 
        samplerate=sr, 
        numcep=num_mfcc, 
        winlen=win_length, 
        winstep=hop_length
    )
    
    # Compute first-order delta of MFCC
    delta_mfcc = python_speech_features.base.delta(mfcc_features, 1)
    
    # Compute second-order delta of MFCC
    delta2_mfcc = python_speech_features.base.delta(mfcc_features, 2)
    
    # Stack MFCC features and their first and second order deltas horizontally
    audio_features = np.hstack((mfcc_features, delta_mfcc, delta2_mfcc))
    
    return audio_features

# Usage example
# audio_features = extract_audio_features('your_audio_path.wav')
# The shape of audio_features will be (T, 39), where T is the number of time steps