from flask import Flask, render_template, request, jsonify, redirect, url_for
import os
from werkzeug.utils import secure_filename
from emotion_speech_model import EmotionSpeechModel
import sounddevice as sd
import numpy as np
import wave
from datetime import datetime

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'wav', 'mp3', 'ogg'}

# Initialize model
model = EmotionSpeechModel('emotion_speech_model.h5')

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables for recording
is_recording = False
recording_frames = []
sample_rate = 22050  # Matches model training sample rate

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get emotion prediction
        emotion, confidence = model.predict_emotion(filepath)
        
        # Format confidence scores
        confidences = {model.emotion_labels[i]: f"{confidence[i]*100:.2f}%" 
                      for i in range(len(confidence))}
        
        # Clean up
        os.remove(filepath)
        
        return render_template('index.html', 
                             emotion=emotion,
                             confidences=confidences,
                             audio_source=None)
    
    return redirect(request.url)

@app.route('/start_recording', methods=['POST'])
def start_recording():
    global is_recording, recording_frames
    
    if not is_recording:
        is_recording = True
        recording_frames = []
        
        def callback(indata, frames, time, status):
            if is_recording:
                recording_frames.append(indata.copy())
        
        # Start recording
        with sd.InputStream(callback=callback, 
                          channels=1,
                          samplerate=sample_rate,
                          dtype='float32'):
            while is_recording:
                sd.sleep(100)
        
        return jsonify({'status': 'recording started'})
    
    return jsonify({'status': 'already recording'})

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global is_recording, recording_frames
    
    if is_recording:
        is_recording = False
        
        # Save recording
        filename = f"recording_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        # Convert frames to numpy array
        audio_data = np.concatenate(recording_frames, axis=0)
        
        # Ensure we have exactly 3 seconds of audio (66150 samples)
        if len(audio_data) < 66150:
            audio_data = np.pad(audio_data, (0, 66150 - len(audio_data)), 'constant')
        else:
            audio_data = audio_data[:66150]
        
        # Save as WAV file
        with wave.open(filepath, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes((audio_data * 32767).astype(np.int16))
        
        # Get emotion prediction
        emotion, confidence = model.predict_emotion(filepath)
        
        # Format confidence scores
        confidences = {model.emotion_labels[i]: f"{confidence[i]*100:.2f}%" 
                      for i in range(len(confidence))}
        
        # Clean up
        os.remove(filepath)
        
        return jsonify({
            'status': 'recording stopped',
            'emotion': emotion,
            'confidences': confidences,
            'audio_source': None
        })
    
    return jsonify({'status': 'not recording'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)