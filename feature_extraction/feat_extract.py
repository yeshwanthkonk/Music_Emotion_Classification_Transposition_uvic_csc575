import librosa
import numpy as np
from pydub import AudioSegment


def convert_mp3_to_wav(mp3_path,wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")

def extract_features(wav_path):
    y, sr = librosa.load(wav_path, sr=22050)
    mfcc_mean = np.mean(librosa.feature.mfcc(y=y, sr=sr), axis=1)
    chroma_mean = np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1)
    spectral_contrast_mean = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
    tempo = librosa.beat.beat_track(y=y, sr=sr)
    loudness_variation = np.std(y)

    return mfcc_mean, chroma_mean, spectral_contrast_mean, tempo, loudness_variation


def prepare_features_for_prediction(mfcc_mean, chroma_mean, spectral_contrast_mean, tempo_tuple, loudness_variation):
    # Unpack scalar tempo from tuple
    tempo_scalar = float(tempo_tuple[0][0]) if isinstance(tempo_tuple, tuple) else float(tempo_tuple)

    # Flatten all components
    flat_features = np.hstack([
        mfcc_mean.astype(np.float32),                 # (20,)
        chroma_mean.astype(np.float32),               # (12,)
        spectral_contrast_mean.astype(np.float32),    # (7,)
        [tempo_scalar, loudness_variation]            # (2,)
    ])

    assert flat_features.shape == (41,), f"Expected shape (41,), got {flat_features.shape}"
    return flat_features
