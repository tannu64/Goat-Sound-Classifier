import streamlit as st
import joblib
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import io
import seaborn as sns

# --------------------------------------------------------------
# Helper: matches the features your model was trained on
# --------------------------------------------------------------
def compute_goat_features(audio_data, sr=16000):
    """
    Extract the same audio features used during training.
    Adjust if your model used a different set of features.
    """
    if sr != 16000:
        audio_data = librosa.resample(audio_data, orig_sr=sr, target_sr=16000)
        sr = 16000

    duration = librosa.get_duration(y=audio_data, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(audio_data)[0].mean()
    rms = librosa.feature.rms(y=audio_data)[0].mean()
    spec_centroid = librosa.feature.spectral_centroid(y=audio_data, sr=sr)[0].mean()
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio_data, sr=sr)[0].mean()
    spec_rolloff   = librosa.feature.spectral_rolloff(y=audio_data, sr=sr)[0].mean()

    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=13)
    mfcc_means = mfcc.mean(axis=1)

    features = {
        "duration": duration,
        "zcr": zcr,
        "rms": rms,
        "spec_centroid": spec_centroid,
        "spec_bandwidth": spec_bandwidth,
        "spec_rolloff": spec_rolloff
    }
    for i, val in enumerate(mfcc_means, start=1):
        features[f"mfcc_{i}"] = val
    
    return features

# --------------------------------------------------------------
# Streamlit App
# --------------------------------------------------------------
def main():
    st.title("Goat Sound Classifier")

    # 1) Load model and label encoder
    model = joblib.load("best_model_fold2.pkl")  # your trained model
    label_encoder = joblib.load("label_encoder.pkl")  # the LabelEncoder used in training

    # Show all possible label names
    all_labels = label_encoder.classes_
    st.write("**All possible goat-sound labels:**", all_labels)

    st.write("""
    **Instructions**:
    1. Upload a goat audio file (WAV/MP3/OGG).
    2. We'll display its waveform & spectrogram.
    3. We'll predict the goat sound category using the loaded model.
    """)

    uploaded_file = st.file_uploader("Upload Goat Audio", type=["wav", "mp3", "ogg"])
    
    if uploaded_file is not None:
        # Convert the uploaded file to a bytes buffer
        file_bytes = uploaded_file.read()
        audio_buffer = io.BytesIO(file_bytes)

        # Load with librosa
        try:
            y, sr = librosa.load(audio_buffer, sr=None, mono=True)
        except Exception as e:
            st.error(f"Error loading audio: {e}")
            return

        # Play the audio in Streamlit
        # For WAV: format="audio/wav"; for MP3: "audio/mp3", etc.
        st.audio(uploaded_file, format="audio/wav")

        # Plot waveform
        fig_wave, ax_wave = plt.subplots(figsize=(6, 3))
        librosa.display.waveshow(y, sr=sr, ax=ax_wave)
        ax_wave.set_title("Waveform")
        ax_wave.set_xlabel("Time (s)")
        ax_wave.set_ylabel("Amplitude")
        st.pyplot(fig_wave)

        # Plot spectrogram (with colorbar fix)
        fig_spec, ax_spec = plt.subplots(figsize=(6, 4))
        D = np.abs(librosa.stft(y))
        DB = librosa.amplitude_to_db(D, ref=np.max)
        img = librosa.display.specshow(DB, sr=sr, x_axis='time', y_axis='log', cmap='magma', ax=ax_spec)
        ax_spec.set_title("Spectrogram (Log Scale)")
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Log Frequency (Hz)")
        fig_spec.colorbar(img, ax=ax_spec, format="%+2.0f dB")
        st.pyplot(fig_spec)

        # Extract features
        feats_dict = compute_goat_features(y, sr=sr)
        # Sort keys to match model's expected feature order
        feature_vector = [feats_dict[k] for k in sorted(feats_dict.keys())]
        feature_array = np.array(feature_vector).reshape(1, -1)

        # Predict numeric label
        numeric_pred = model.predict(feature_array)
        # Decode to string label
        string_pred = label_encoder.inverse_transform(numeric_pred)

        st.success(f"**Predicted Goat Sound**: {string_pred[0]}")

        # Optionally "speak" or confirm classification
        st.write(f"This goat sound is classified as: **{string_pred[0]}**")

    else:
        st.info("Please upload an audio file to get started.")

if __name__ == "__main__":
    main()
