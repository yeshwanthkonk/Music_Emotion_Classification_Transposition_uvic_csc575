import os
import librosa
import pretty_midi
from midi2audio import FluidSynth
from spleeter.separator import Separator
from pydub import AudioSegment

import multiprocessing

from transposition.key_trans_main import conversion

if __name__ == "__main__":
    multiprocessing.freeze_support()


input_folder = os.path.join(os.getcwd(), "music_files")

# === 1. SPLIT MP3 INTO STEMS ===
def split_stems(mp3_path, output_dir="stems_output"):
    separator = Separator('spleeter:2stems')
    separator.separate_to_file(mp3_path, output_dir)
    return os.path.join(output_dir, os.path.splitext(os.path.basename(mp3_path))[0])


# === 2. CONVERT STEM TO MIDI (ACCOMPANIMENT) ===
def convert_to_midi(wav_path, midi_path):
    y, sr = librosa.load(wav_path)
    onset_frames = librosa.onset.onset_detect(y=y, sr=sr, backtrack=True)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)

    midi = pretty_midi.PrettyMIDI()
    piano = pretty_midi.Instrument(program=0)

    for frame in onset_frames:
        pitch_slice = pitches[:, frame]
        index = pitch_slice.argmax()
        freq = librosa.fft_frequencies(sr=sr)[index]
        if freq > 0:
            midi_pitch = int(librosa.hz_to_midi(freq))
            start = librosa.frames_to_time(frame, sr=sr)
            end = start + 0.5
            note = pretty_midi.Note(velocity=100, pitch=midi_pitch, start=start, end=end)
            piano.notes.append(note)
    midi.instruments.append(piano)
    midi.write(midi_path)


# === 3. TRANSPOSE MIDI ===
def transpose_midi(input_midi, output_midi, semitones=-3):  # major → minor shift
    midi = pretty_midi.PrettyMIDI(input_midi)
    for instrument in midi.instruments:
        for note in instrument.notes:
            note.pitch += semitones
    midi.write(output_midi)


def render_midi_to_wav(midi_path, wav_output, soundfont_path):
    fs = FluidSynth(sound_font=soundfont_path)

    # Use absolute path for output to control where the file is saved
    abs_wav_path = os.path.abspath(wav_output)
    fs.midi_to_audio(midi_path, abs_wav_path)

    print(f"✅ WAV rendered at: {abs_wav_path}")
    return abs_wav_path  # Return the real path


# === 5. MIX VOCALS + NEW ACCOMPANIMENT ===
def mix_tracks(vocal_wav, transformed_wav, output_mp3):
    vocal = AudioSegment.from_wav(vocal_wav)
    transformed = AudioSegment.from_wav(transformed_wav)
    mixed = transformed.overlay(vocal)
    mixed.export(output_mp3, format="mp3")

temp_directory = os.path.join(os.getcwd(), "temp")
if not os.path.exists(temp_directory):
    os.makedirs(temp_directory)

original_mid = os.path.join(temp_directory, "original_midi.mid")
transformed_mid = os.path.join(temp_directory, "transformed_midi.mid")
new_instrument = os.path.join(temp_directory, "new_instrument.wav")
final_music = os.path.join(temp_directory, "final_output.mp3")

# === === RUN EVERYTHING === ===
# if __name__ == "__main__":
#     # everything from this point should be indented under this
#     mp3_input = os.path.join(input_folder, "1000.mp3")
#     soundfont = os.path.join(input_folder, "FluidR3_GM.sf2")  # Download and provide the path here
#
#     stem_path = split_stems(mp3_input)
#     vocals = os.path.join(stem_path, "vocals.wav")
#     accompaniment = os.path.join(stem_path, "accompaniment.wav")
#
#     convert_to_midi(accompaniment, original_mid)
#     transpose_midi(original_mid, transformed_mid, semitones=-3)
#     rendered_wav = render_midi_to_wav(transformed_mid, new_instrument, soundfont)
#     mix_tracks(vocals, rendered_wav, final_music)
#
#     print("✅ Emotionally transformed MP3 saved as: final_output.mp3")


soundfont = os.path.join(input_folder, "FluidR3_GM.sf2")

def transpose(mp3_input, predicted_emotion, target_emotion):
      # Download and provide the path here

    stem_path = split_stems(mp3_input)
    vocals = os.path.join(stem_path, "vocals.wav")
    accompaniment = os.path.join(stem_path, "accompaniment.wav")

    convert_to_midi(accompaniment, original_mid)
    conversion(original_mid, predicted_emotion, target_emotion, transformed_mid)
    rendered_wav = render_midi_to_wav(transformed_mid, new_instrument, soundfont)
    mix_tracks(vocals, rendered_wav, final_music)
    print("✅ Emotionally transformed MP3 saved as: final_output.mp3")
    return final_music
