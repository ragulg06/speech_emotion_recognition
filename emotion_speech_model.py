import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa
import numpy as np
import os

class EmotionSpeechModel:
    def __init__(self, model_path='emotion_speech_model.h5'):
        self.model = load_model(model_path)
        self.emotion_labels = {
            0: 'angry',
            1: 'disgust',
            2: 'fear',
            3: 'happy',
            4: 'neutral',
            5: 'sad'
        }
        
    def extract_features(self, audio_path, mfcc=True, chroma=True, mel=True):
        """Extract audio features including MFCCs"""
        try:
            sound, sr = librosa.load(audio_path, sr=22050, duration=3)
            
            # Pad or truncate to 3 seconds (66150 samples at 22050Hz)
            if len(sound) < 66150:
                sound = np.pad(sound, (0, 66150 - len(sound)), 'constant')
            else:
                sound = sound[:66150]
                
            # Extract features
            features = []
            if mfcc:
                mfccs = librosa.feature.mfcc(y=sound, sr=sr, n_mfcc=40)
                mfccs_processed = np.mean(mfccs.T, axis=0)
                features.extend(mfccs_processed)
                
            if chroma:
                chroma = librosa.feature.chroma_stft(y=sound, sr=sr)
                chroma_processed = np.mean(chroma.T, axis=0)
                features.extend(chroma_processed)
                
            if mel:
                mel = librosa.feature.melspectrogram(y=sound, sr=sr)
                mel_processed = np.mean(mel.T, axis=0)
                features.extend(mel_processed)
                
            return np.array(features)
        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            return None
    
    def predict_emotion(self, audio_path):
        """Predict emotion from audio file"""
        features = self.extract_features(audio_path)
        if features is None:
            return "Could not process audio file"
            
        # Reshape for model input (batch_size=1, timesteps=40, features=1)
        features = features[:40]  # Take first 40 MFCCs to match model input
        features = np.expand_dims(features, axis=0)
        features = np.expand_dims(features, axis=-1)
        
        # Predict
        predictions = self.model.predict(features)
        predicted_emotion = np.argmax(predictions)
        
        return self.emotion_labels[predicted_emotion], predictions[0]