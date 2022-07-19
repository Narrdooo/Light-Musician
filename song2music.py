import argparse
import librosa
import os
from tools import separate_vocal, transcribe_vocal, play_midi

def song2midi(args):
    song_path = args.path
    file_name = song_path.split('/')[-1]
    song_name = file_name[:file_name.find('.')]
    outdir = args.outdir
    midi_path = os.path.join(outdir, '%s.mid'%song_name)
    print("Selected song: %s"%song_name)
    print("Midi file will save to %s"%midi_path)
    if not os.path.exists(song_path):
        print("Song not Found!")
        return

    song, sr = librosa.core.load(song_path, sr=None, mono=True)
    if sr != 44100:
        song = librosa.core.resample(y=song, orig_sr=sr, target_sr=44100)
    vocal = separate_vocal(song)
    transcribe_vocal(vocal, midi_path)
    return midi_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path')
    parser.add_argument('--outdir')
    args = parser.parse_args()
    midi_path = song2midi(args)
    play_midi(midi_path)
