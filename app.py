# Importing Libraries
import os
import tempfile
import numpy as np
import tensorflow as tf
import librosa
import streamlit as st

# Paths and Configuration
MODEL_PATH = "saved_models/best_model_seed42.keras"
THRESHOLD_PATH = "saved_models/threshold_seed42.txt"
SAMPLE_RATE = 16000
SEQ_LENGTH = 32000   


# streamlit page configuration
st.set_page_config(page_title="Deepfake Audio Detection",layout="centered")
st.title("Deepfake Audio Detection")
st.write("Upload an audio file to classify it as Genuine Human speech or Deepfake AI-generated speech.")



# Loading Model and Threshold
# Cache the model so it is not loaded again whenever the app reruns
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

# Cache the threshold so it is not loaded again whenever the app reruns
@st.cache_data
def load_threshold():
    with open(THRESHOLD_PATH, "r") as f:
        threshold = float(f.read().strip())
    return threshold


model = load_model()
fake_threshold = load_threshold()

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

    # Converting audio to TensorFlow tensor
    audio_tensor = tf.convert_to_tensor(audio, dtype=tf.float32)

    # Converting 1D waveform into 2D spectrogram using STFT
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
def predict_audio(audio_path):
    spectrogram = preprocess_audio(audio_path)

    prediction = model.predict(spectrogram, verbose=0)[0]

    fake_probability = float(prediction[0])
    real_probability = float(prediction[1])

    if fake_probability >= fake_threshold:
        predicted_label = "Deepfake (AI-Generated)"
        confidence = (fake_probability - fake_threshold) / (1 - fake_threshold)
    else:
        predicted_label = "Genuine (Human)"
        confidence = (fake_probability - fake_threshold) / (fake_threshold)
    confidence = max(0.0, min(float(confidence), 1.0))
    return predicted_label, fake_probability, real_probability, confidence


# Uploading file using streamlit
uploaded_file = st.file_uploader(
    "Upload an audio file",
    type=["wav", "mp3", "flac", "ogg", "m4a"]
)


if uploaded_file is not None:

    # Allows user to play the uploaded file
    st.audio(uploaded_file)
    
    # Creating a temporary file name as librosa.load() requires the file path 
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(uploaded_file.read())
        temp_audio_path = temp_audio.name
    
    # Result Prediction
    if st.button("Detect Audio"):
        with st.spinner("Analyzing audio..."):
            predicted_label, fake_prob, real_prob, confidence = predict_audio(temp_audio_path)

        st.subheader("Prediction Result")

        if predicted_label == "Deepfake (AI-Generated)":
            st.error(f"Prediction: {predicted_label}")
        else:
            st.success(f"Prediction: {predicted_label}")

        st.write(f"**Fake Probability:** {fake_prob:.6f}")
        st.write(f"**Real Probability:** {real_prob:.6f}")
        st.write(f"**Confidence Score:** {confidence * 100:.2f}%")
        st.write(f"**Threshold Used:** {fake_threshold:.10f}")

        progress_value = max(0.0, min(float(confidence), 1.0))
        st.progress(progress_value)
        os.remove(temp_audio_path)


# Sidebar for Information
st.sidebar.header("Model Info")
st.sidebar.write("Model: Spectrogram CNN")
st.sidebar.write("Input: 2-second audio")
st.sidebar.write("Sample Rate: 16 kHz")
st.sidebar.write(f"Threshold: {fake_threshold:.10f}")

st.sidebar.header("Classes")
st.sidebar.write("0 → Deepfake / AI-Generated")
st.sidebar.write("1 → Genuine / Human")