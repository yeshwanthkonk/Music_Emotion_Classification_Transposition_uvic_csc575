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