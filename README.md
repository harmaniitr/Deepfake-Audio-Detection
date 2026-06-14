# Deepfake Audio Detection

## Project Overview

This project detects whether a given speech recording is **Genuine (Human)** or **Deepfake (AI-Generated)**. The system uses a spectrogram-based Convolutional Neural Network (CNN) for binary audio classification.

Raw audio is converted into fixed-length 2-second waveforms, transformed into STFT-based spectrograms, and then classified using the trained CNN model.

## Files Included

- `main.ipynb` - Training, validation, testing, and evaluation notebook
- `audio_predictor.py` - Python script for testing a single audio file
- `app.py` - Streamlit web application for audio upload and prediction
- `saved_models/best_model_seed42.keras` - Trained CNN model
- `saved_models/threshold_seed42.txt` - Selected decision threshold
- `performance_report.pdf` - Final evaluation metrics and result summary
- `methodology_report.pdf` - Preprocessing, feature extraction, and model architecture description

## Model Pipeline

The model follows this pipeline:

1. Load audio at **16 kHz**
2. Convert audio to a fixed length of **2 seconds**
3. Convert 1D waveform into 2D spectrogram using **STFT**
4. Resize spectrogram to **128 × 128**
5. Apply per-image standardization
6. Classify using a CNN model

## How to Run the Python Script

The Python script can be run either from VS Code by opening `audio_predictor.py` and clicking **Run Code**, or from the terminal using:

```bash
python audio_predictor.py
```
Then enter the audio file path when prompted.

Example:

```bash
Enter audio file path: sample_audio.wav
```

The script displays:

- Prediction
- Fake probability
- Real probability
- Confidence score
- Threshold used

## How to Run the Streamlit App

Run the Streamlit app from the project directory:

```bash
streamlit run app.py
```

Upload an audio file through the web interface and click **Detect Audio**.

The app displays:

- Predicted class
- Fake probability
- Real probability
- Confidence score
- Threshold used

## Output Classes

The model predicts one of the following classes:

- **Deepfake (AI-Generated)**
- **Genuine (Human)**

## Final Test Performance

| Metric | Value |
|---|---:|
| Accuracy | 92.83% |
| Equal Error Rate (EER) | 7.17% |
| Macro F1-Score | 92.83% |
| Weighted F1-Score | 92.83% |
| Fake Class Accuracy | 92.83% |
| Real Class Accuracy | 92.83% |

## Notes

The final prediction is made using the selected fake-class threshold rather than the default `0.5` threshold. The same preprocessing pipeline is used in the notebook, Python script, and Streamlit application to ensure consistent predictions.

A threshold-aware confidence score is used during inference to show how far the prediction is from the selected decision boundary.