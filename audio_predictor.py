# Importing Libraries
import os
import librosa
import numpy as np
import tensorflow as tf

# Paths and Configuration
MODEL_PATH = "saved_models/best_model_seed42.keras"
THRESHOLD_PATH = "saved_models/threshold_seed42.txt"
SAMPLE_RATE = 16000
SEQ_LENGTH = 32000  

# Loading Model and Threshold
model = tf.keras.models.load_model(MODEL_PATH)

with open(THRESHOLD_PATH, "r") as f:
    fake_threshold = float(f.read().strip())

# Audio Preprocessing
def preprocess_audio(file_path):

    # Loads an audio file and applies the same preprocessing used during model training.

    # Loading audio at 16 kHz and convert to mono
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)

    # Padding or trimming audio to fixed length of 32000 samples
    if len(audio) < SEQ_LENGTH:
        audio = np.pad(audio, (0, SEQ_LENGTH - len(audio)))
    else:
        audio = audio[:SEQ_LENGTH]

    # Convert audio to TensorFlow tensor
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Convert 1D waveform into 2D spectrogram using STFT
    stft = tf.signal.stft(audio_tensor, frame_length=255, frame_step=128)
    spectrogram = tf.abs(stft)

    # Applying log scaling
    spectrogram = tf.math.log(spectrogram + 1e-5)

    # Adding channel dimension
    spectrogram = tf.expand_dims(spectrogram, axis=-1)

    # Resizing spectrogram to match model input shape
    spectrogram = tf.image.resize(spectrogram, [128, 128])

    # Applying same standardization used in notebook
    spectrogram = tf.image.per_image_standardization(spectrogram)

    # Adding batch dimension: (1, 128, 128, 1)
    spectrogram = tf.expand_dims(spectrogram, axis=0)

    return spectrogram

# Prediction function
def predict_audio(file_path):

    spectrogram = preprocess_audio(file_path)

    prediction = model.predict(spectrogram, verbose=0)[0]

    fake_probability = float(prediction[0])
    real_probability = float(prediction[1])

    if fake_probability >= fake_threshold:
        predicted_label = "Deepfake (AI-Generated)"
        confidence = (fake_probability - fake_threshold) / (1 - fake_threshold)
    else:
        predicted_label = "Genuine (Human)"
        confidence = (fake_threshold - fake_probability) / fake_threshold


    print("Prediction:", predicted_label)
    print(f"Confidence Score: {confidence * 100:.6f}%")
    print(f"Fake Probability: {fake_probability * 100:.6f}%")
    print(f"Real Probability: {real_probability * 100:.6f}%")
    print(f"Threshold Used: {fake_threshold}")

audio_path = input("Enter audio file path: ").strip()
predict_audio(audio_path)