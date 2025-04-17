import mido
from music21 import converter, key
from transposition.change_key import *

# sad calm angry happy

# input: call conversion() function, it needs the 'file_path' of the file in the backend(recommend)
# detected_emotion: it could be one of the 'sad' 'calm' 'angry' 'happy'
# target_emotion: it could be one of the 'sad' 'calm' 'angry' 'happy', but must be different from the detected emotion

# output: modified midi file stream. Now is the stream, but it can be saved into the disk. Please refer to test()

def conversion(file_path, detected_emotion, target_emotion, transformed_mid="output.mid"):
    midi_file = converter.parse(file_path)
    original_key = midi_file.analyze('key')
    modified_midi = None
    if target_emotion == 'calm':
        modified_midi = convert_to_calm(file_path, original_key, detected_emotion)
    elif target_emotion == 'sad':
        modified_midi = convert_to_sad(file_path, original_key, detected_emotion)
    elif target_emotion == 'angry':
        modified_midi = convert_to_angry(file_path, original_key, detected_emotion)
    elif target_emotion == 'happy':
        modified_midi = convert_to_happy(file_path, original_key, detected_emotion)
    else:
        print('error')
    modified_midi.save(transformed_mid)
    return modified_midi

def convert_to_calm(file_path, original_key, detected_emotion):
    key = str(original_key.tonic)
    mode = original_key.mode  # 'major' or 'minor'
    midi = mido.MidiFile(file_path)
    if detected_emotion == 'sad':
        if mode == 'major':
            # convert to Dorian； BPM +10~15%
            midi = shift2others(midi, 'dorian', key)
            midi = adjust_bpm(midi, factor=1.1)
        else:
            # convert to Dorian； BPM +10~15%
            midi = min2maj(midi, 'relative', key)
            midi = shift2others(midi, 'dorian', key)
            midi = adjust_bpm(midi, factor=1.1)
    elif detected_emotion == 'angry':
        if mode == 'major':
            # convert to Dorian； BPM -15~20%
            midi = shift2others(midi, 'dorian', key)
            midi = adjust_bpm(midi, factor=0.8)
        else:
            # convert to Dorian； BPM -15~20%
            midi = min2maj(midi, 'relative', key)
            midi = shift2others(midi, 'dorian', key)
            midi = adjust_bpm(midi, factor=0.8)
    elif detected_emotion == 'happy':
        if mode == 'major':
            # convert to Dorian； BPM -10~15%
            midi = shift2others(midi, 'dorian', key)
            midi = adjust_bpm(midi, factor=0.9)
        else:
            # convert to Dorian； BPM -10~15%
            midi = min2maj(midi, 'relative', key)
            midi = shift2others(midi, 'dorian', key)
            midi = adjust_bpm(midi, factor=0.9)
    else:
        print('error')
    return midi

def convert_to_sad(file_path, original_key, detected_emotion):
    key = str(original_key.tonic)
    mode = original_key.mode  # 'major' or 'minor'
    midi = mido.MidiFile(file_path)
    if detected_emotion == 'calm':
        if mode == 'major':
            # convert to Aeolian； BPM -15~25%
            midi = maj2min(midi, 'aeolian', key)
            midi = adjust_bpm(midi, factor=0.85)
        else:
            # convert to Phrygian； BPM -15~25%
            midi = min2maj(midi, 'relative', key)
            midi = shift2others(midi, 'phrygian', key)
            midi = adjust_bpm(midi, factor=0.85)
    elif detected_emotion == 'angry':
        if mode == 'major':
            # convert to Aeolian； BPM -10~20%
            midi = maj2min(midi, 'aeolian', key)
            midi = adjust_bpm(midi, factor=0.90)
            pass
        else:
            # convert to Phrygian； BPM -10~20%
            midi = min2maj(midi, 'relative', key)
            midi = shift2others(midi, 'phrygian', key)
            midi = adjust_bpm(midi, factor=0.90)
            pass
    elif detected_emotion == 'happy':
        if mode == 'major':
            # convert to Aeolian； BPM -15~25%
            midi = maj2min(midi, 'aeolian', key)
            midi = adjust_bpm(midi, factor=0.85)
        else:
            # convert to Phrygian； BPM -15~25%
            midi = min2maj(midi, 'relative', key)
            midi = shift2others(midi, 'phrygian', key)
            midi = adjust_bpm(midi, factor=0.85)
    else:
        print('error')
    return midi

def convert_to_angry(file_path, original_key, detected_emotion):
    key = str(original_key.tonic)
    mode = original_key.mode  # 'major' or 'minor'
    midi = mido.MidiFile(file_path)
    if detected_emotion == 'calm':
        if mode == 'major':
            # convert to Phrygian； BPM +25~40%
            midi = shift2others(midi, 'phrygian', key)
            midi = adjust_bpm(midi, factor=1.25)
        else:
            # convert to Harmonic minor； BPM +25~40%
            midi = maj2min(midi, 'harmonic', key)
            midi = adjust_bpm(midi, factor=1.25)

    elif detected_emotion == 'sad':
        if mode == 'major':
            # convert to Phrygian； BPM +20~35%
            midi = shift2others(midi, 'phrygian', key)
            midi = adjust_bpm(midi, factor=1.20)
        else:
            # convert to Harmonic minor； BPM +20~35%
            midi = maj2min(midi, 'harmonic', key)
            midi = adjust_bpm(midi, factor=1.20)
    elif detected_emotion == 'happy':
        if mode == 'major':
            # convert to Mixolydian； BPM +10~20%
            midi = shift2others(midi, 'mixolydian', key)
            midi = adjust_bpm(midi, factor=1.10)
        else:
            # convert to Ionian； BPM +10~20%
            midi = min2maj(midi, 'ionian', key)
            midi = adjust_bpm(midi, factor=1.10)
    else:
        print('error')
    return midi

def convert_to_happy(file_path, original_key, detected_emotion):
    key = str(original_key.tonic)
    mode = original_key.mode  # 'major' or 'minor'
    midi = mido.MidiFile(file_path)
    if detected_emotion == 'calm':
        if mode == 'major':
            # convert to Lydian； BPM +10~20%
            midi = shift2others(midi, 'lydian', key)
            midi = adjust_bpm(midi, factor=1.1)
        else:
            # convert to Ionian； BPM +10~20%
            midi = min2maj(midi, 'ionian', key)
            midi = adjust_bpm(midi, factor=1.1)
    elif detected_emotion == 'sad':
        if mode == 'major':
            # convert to Ionian； BPM +15~25%
            midi = shift2others(midi, 'lydian', key)
            midi = adjust_bpm(midi, factor=1.15)
        else:
            # convert to Ionian； BPM +15~25%
            midi = min2maj(midi, 'ionian', key)
            midi = adjust_bpm(midi, factor=1.15)
    elif detected_emotion == 'angry':
        if mode == 'major':
            # convert to Mixolydian； BPM +10~20%
            midi = shift2others(midi, 'mixolydian', key)
            midi = adjust_bpm(midi, factor=1.1)
        else:
            # convert to Ionian； BPM +10~20%
            midi = min2maj(midi, 'ionian', key)
            midi = adjust_bpm(midi, factor=1.1)
    else:
        print('error')
    return midi


def test():
    file = 'kachiusa.mid'
    # suppose this music is angry, and we wanna transform the emotion to happy:
    result = conversion(file, 'angry', 'happy')
    result.save("output.mid")
#test()