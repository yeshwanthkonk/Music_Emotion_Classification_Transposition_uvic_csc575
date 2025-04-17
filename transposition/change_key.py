import numpy as np
import mido

major_to_minor = {
    "C": "A", "G": "E", "D": "B", "A": "F#", "E": "C#", "B": "G#",
    "F#": "D#", "C#": "A#", "F": "D", "Bb": "G", "Eb": "C", "Ab": "F",
    "Db": "Bb", "Gb": "Eb", "Cb": "Ab"
}

minor_to_major = {v: k for k, v in major_to_minor.items()}

key_node = {
    "C": 0,
    "C#": 1, "Db": 1,
    "D": 2,
    "D#": 3, "Eb": 3,
    "E": 4,
    "F": 5,
    "F#": 6, "Gb": 6,
    "G": 7,
    "G#": 8, "Ab": 8,
    "A": 9,
    "A#": 10, "Bb": 10,
    "B": 11
}

KEY_TEMPLATES = {
    "C major": [0, 2, 4, 5, 7, 9, 11], "A minor": [9, 11, 0, 2, 4, 5, 7],
    "G major": [7, 9, 11, 0, 2, 4, 6], "E minor": [4, 6, 7, 9, 11, 0, 2],
    "D major": [2, 4, 6, 7, 9, 11, 1], "B minor": [11, 1, 2, 4, 6, 7, 9],
    "A major": [9, 11, 1, 2, 4, 6, 8], "F# minor": [6, 8, 9, 11, 1, 2, 4],
    "E major": [4, 6, 8, 9, 11, 1, 3], "C# minor": [1, 3, 4, 6, 8, 9, 11],
    "B major": [11, 1, 3, 4, 6, 8, 10], "G# minor": [8, 10, 11, 1, 3, 4, 6],
    "F major": [5, 7, 9, 10, 0, 2, 4], "D minor": [2, 4, 5, 7, 9, 10, 0],
    "Bb major": [10, 0, 2, 3, 5, 7, 9], "G minor": [7, 9, 10, 0, 2, 3, 5],
    "Eb major": [3, 5, 7, 8, 10, 0, 2], "C minor": [0, 2, 3, 5, 7, 8, 10],
    "Ab major": [8, 10, 0, 1, 3, 5, 7], "F minor": [5, 7, 8, 10, 0, 1, 3],
    "Db major": [1, 3, 5, 6, 8, 10, 0], "Bb minor": [10, 0, 1, 3, 5, 6, 8],
    "Gb major": [6, 8, 10, 11, 1, 3, 5], "Eb minor": [3, 5, 6, 8, 10, 11, 1]
}

def key_detection(midi):
    note_counts = np.zeros(12)
    for track in midi.tracks:
        for msg in track:
            if msg.type == 'note_on' and msg.velocity > 0:
                note = msg.note % 12
                note_counts[note] += 1

    best_key = None
    best_score = -1

    for key, template in KEY_TEMPLATES.items():
        score = sum(note_counts[n] for n in template)
        if score > best_score:
            best_score = score
            best_key = key
    return best_key

def maj2min(midi, convert_type, original_key):
    original_key_node = key_node[original_key]
    for track in midi.tracks:
        for msg in track:
            if convert_type == 'relative':
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    # 3, 6, 7
                    if scale == 4 or scale == 9 or scale == 11:
                        msg.note = max(0, min(127, msg.note - 4))
                    else:
                        msg.note = max(0, min(127, msg.note - 3))
            elif convert_type == 'harmonic':
                # 3, 6
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    if scale == 4 or scale == 9:
                        msg.note = max(0, min(127, msg.note - 4))
                    else:
                        msg.note = max(0, min(127, msg.note - 3))
            elif convert_type == 'melodic':
                print('wait...')
            elif convert_type == 'aeolian':
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    # 3, 6, 7
                    if scale == 4 or scale == 9 or scale == 11:
                        msg.note = max(0, min(127, msg.note - 1))
    # output_file = midi_file[:-4] + '_' + convert_type +'.mid'
    # midi.save(output_file)
    return midi

def min2maj(midi, convert_type, original_key):
    original_key_node = key_node[original_key]

    for track in midi.tracks:
        for msg in track:
            if convert_type == 'relative':
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    # 3, 6, 7
                    if scale == 3 or scale == 8 or scale == 10:
                        msg.note = max(0, min(127, msg.note + 4))
                    else:
                        msg.note = max(0, min(127, msg.note + 3))
            elif convert_type == 'ionian':
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    # 3, 6, 7
                    if scale == 3 or scale == 8 or scale == 10:
                        msg.note = max(0, min(127, msg.note + 1))
    # output_file = midi_file[:-4] + '_' + convert_type +'.mid'
    # midi.save(output_file)
    return midi

def shift2others(midi, convert_type, original_key):
    original_key_node = key_node[original_key]

    for track in midi.tracks:
        for msg in track:
            if convert_type == 'dorian':
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    # 3, 7
                    if scale == 4 or scale == 11:
                        msg.note = max(0, min(127, msg.note - 4))
                    else:
                        msg.note = max(0, min(127, msg.note - 3))
            elif convert_type == 'phrygian':
                # 2, 3, 6, 7
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    if scale == 2 or scale == 4 or scale == 9 or scale == 11:
                        msg.note = max(0, min(127, msg.note - 4))
                    else:
                        msg.note = max(0, min(127, msg.note - 3))
            elif convert_type == 'mixolydian':
                # 7
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    if scale == 11:
                        msg.note = max(0, min(127, msg.note - 4))
                    else:
                        msg.note = max(0, min(127, msg.note - 3))
            if convert_type == 'lydian':
                if msg.type in ['note_on', 'note_off']:
                    scale = (msg.note - original_key_node) % 12
                    # 4 rise up
                    if scale == 5:
                        msg.note = max(0, min(127, msg.note + 4))
                    else:
                        msg.note = max(0, min(127, msg.note + 3))
    return midi

def adjust_bpm(midi, factor=1.0):
    new_midi = mido.MidiFile()

    for track in midi.tracks:
        # new_track = mido.MidiTrack()
        for msg in track:
            if msg.type == 'set_tempo':
                msg.tempo = int(msg.tempo * (1 / factor))


    return midi


'''
# Testing code, ignore it
# shift_min2maj("kachiusa.mid",'relative','E')

print(key_detection(mido.MidiFile('kachiusa.mid')))
print(key_detection(mido.MidiFile('canon-3.mid')))
print(key_detection(mido.MidiFile('LiuYangHe.mid')))
# shift_maj2min("summer.mid",'relative','D')
midi_file = converter.parse("kachiusa.mid")


key_analysis = midi_file.analyze('key')
print(f"Detected key: {key_analysis}")

midi_file = converter.parse("canon-3.mid")
'''