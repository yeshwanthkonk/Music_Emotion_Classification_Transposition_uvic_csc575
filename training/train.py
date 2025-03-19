import os

import torch
import numpy as np
import pretty_midi
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Sample EMOPIA MIDI file paths (replace with actual paths)
files_dir = "./midis/"
midi_files = os.listdir(files_dir)
labels = [file[1] for file in midi_files]  # 0: Happy (Q1), 1: Angry (Q2), 2: Sad (Q3), 3: Calm (Q4)


# Function to extract MIDI features
def extract_midi_features(midi_path):
    try:
        midi_data = pretty_midi.PrettyMIDI(midi_path)
        notes = []
        for instrument in midi_data.instruments:
            for note in instrument.notes:
                notes.append([note.pitch, note.start, note.end, note.velocity])

        if not notes:
            return np.zeros(4)

        notes = np.array(notes)
        mean_pitch = np.mean(notes[:, 0])
        duration = np.mean(notes[:, 2] - notes[:, 1])
        mean_velocity = np.mean(notes[:, 3])
        num_notes = len(notes)

        return np.array([mean_pitch, duration, mean_velocity, num_notes])
    except:
        return np.zeros(4)


# Extract features from MIDI files
features = np.array([extract_midi_features(files_dir+midi) for midi in midi_files])

# Preprocess data
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
X_train, X_test, y_train, y_test = train_test_split(features_scaled, labels, test_size=0.2, random_state=42,
                                                    stratify=labels)

# Train KNN Model for Multi-Class Classification
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# Predict and Evaluate
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(report)
print("Confusion Matrix:")
print(conf_matrix)
