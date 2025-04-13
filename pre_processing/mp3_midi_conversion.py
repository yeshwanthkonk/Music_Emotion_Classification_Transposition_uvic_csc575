# import librosa
# import pretty_midi
# from pydub import AudioSegment
# import numpy as np
# import os
#
# # Step 1: Convert MP3 to WAV
# def convert_mp3_to_wav(mp3_path, wav_path):
#     audio = AudioSegment.from_mp3(mp3_path)
#     audio.export(wav_path, format="wav")
#
# # Step 2: Extract pitch and timing
# def extract_notes(wav_path):
#     y, sr = librosa.load(wav_path)
#     onset_frames = librosa.onset.onset_detect(y=y, sr=sr)
#     onset_times = librosa.frames_to_time(onset_frames, sr=sr)
#     pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
#
#     notes = []
#     for frame in onset_frames:
#         index = magnitudes[:, frame].argmax()
#         pitch = pitches[index, frame]
#         if pitch > 0:
#             notes.append((onset_times[frame], pitch))
#     return notes
#
# # Step 3: Create MIDI
# def notes_to_midi(notes, output_midi_path):
#     midi = pretty_midi.PrettyMIDI()
#     instrument = pretty_midi.Instrument(program=0)
#     for time, pitch in notes:
#         note = pretty_midi.Note(
#             velocity=100,
#             pitch=int(librosa.hz_to_midi(pitch)),
#             start=time,
#             end=time + 0.5  # assuming fixed duration for simplicity
#         )
#         instrument.notes.append(note)
#     midi.instruments.append(instrument)
#     midi.write(output_midi_path)
#
# # Step 4: Full pipeline
# def mp3_to_midi(mp3_path, output_midi_path):
#     wav_path = "temp.wav"
#     convert_mp3_to_wav(mp3_path, wav_path)
#     notes = extract_notes(wav_path)
#     notes_to_midi(notes, output_midi_path)
#     os.remove(wav_path)
#
# # Example
input_folder = "C:/Yeshwanth/UVIC/CSC-575/Music_Emotion_Classification_Transposition_uvic_csc575/music_files"
#
#
# mp3_to_midi(os.path.join(input_folder, "1000.mp3"), "output_song.mid")

import os
import librosa
import pretty_midi
import numpy as np

# Load MP3
audio_path = os.path.join(input_folder, "1000.mp3")  # ← your MP3 file path here
y, sr = librosa.load(audio_path)

# Onset detection
onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
onset_times = librosa.frames_to_time(onset_frames, sr=sr)

# Pitch tracking (basic)
pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

# Create MIDI
midi = pretty_midi.PrettyMIDI()
instrument = pretty_midi.Instrument(program=0)  # 0 = Acoustic Grand Piano

for frame in onset_frames:
    pitch_slice = pitches[:, frame]
    index = pitch_slice.argmax()
    frequency = librosa.fft_frequencies(sr=sr)[index]

    if frequency > 0:
        midi_pitch = int(librosa.hz_to_midi(frequency))
        note = pretty_midi.Note(
            velocity=100,
            pitch=midi_pitch,
            start=librosa.frames_to_time(frame, sr=sr),
            end=librosa.frames_to_time(frame + 10, sr=sr)  # short fixed duration
        )
        instrument.notes.append(note)

midi.instruments.append(instrument)
midi.write('output.mid')
print("✅ MIDI file saved as 'output.mid'")
