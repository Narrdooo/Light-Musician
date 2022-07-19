import os
import librosa
import numpy as np
from scipy import stats
import math
import sounddevice as sd
import matplotlib.pyplot as plt

from samples.preset import *


"""
[TODO]
- velocity
- play midi
- EQ
"""


class Instrument:
    def __init__(self, type=None, sample_dir=None) -> None:
        """
        type:           instrument type
        sample_dir:     directory that contains samples
        """

        self.type = type
        self.sample_dir = sample_dir
        self.notes = [None]*109
        self.sr = 44100

        if sample_dir is None:
            if type is None:
                print("No type selected and no sample provided...\nSetting the instrument as preset piano.")
                self.sample_dir = "./samples/piano"
            else:
                if type in preset_dict:
                    self.sample_dir = preset_dict[type]
                else:
                    raise ValueError("Sorry, this type of instrument is currently not supported.")

        # reading samples
        sample_notes = []
        for sample_path in os.listdir(self.sample_dir):
            wav, sr = librosa.core.load(os.path.join(self.sample_dir, sample_path), sr=self.sr, mono=True)
            # sample preprocessing
            # wav = librosa.util.normalize(wav)
            # idx = 0
            # while wav[idx] < 20:    idx += 1
            # wav = wav[idx:]
            # idx = len(wav) - 1
            # while wav[idx] < 20:    idx -= 1
            # wav = wav[:idx+1]

            f0, _, _ = librosa.pyin(wav, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'), sr=self.sr)
            f0 = stats.mode(f0)[0][0]
            pitch = int(math.log(f0*32/440, 2)*12+9)
            self.notes[pitch] = wav
            sample_notes.append(pitch)

        # generate note sounds
        sample_notes.sort()
        for note in range(21, sample_notes[0]):
            self.notes[note] = librosa.effects.pitch_shift(y=self.notes[sample_notes[0]], 
                                                            sr=self.sr, 
                                                            n_steps=note-sample_notes[0])
        for i in range(len(sample_notes)-1):
            note1, note2 = sample_notes[i], sample_notes[i+1]
            for note in range(note1+1, note2):
                selected = note1 if note-note1 < note2-note else note2
                self.notes[note] = librosa.effects.pitch_shift(y=self.notes[selected], 
                                                               sr=self.sr, 
                                                               n_steps=note-selected)
        for note in range(sample_notes[-1]+1, 109):
            self.notes[note] = librosa.effects.pitch_shift(y=self.notes[sample_notes[-1]], 
                                                            sr=self.sr, 
                                                            n_steps=note-sample_notes[-1])



    def play_note_seq(self, seq):
        """
        [input]
            seq:    [[note, onset, offset], [note, onset, offset], ...]
                    e.g.    [[22, 1.23, 5.34], ...]
        """
        wav = np.array([])
        for event in seq:
            note, onset, offset = event
            sub_wav = np.append(np.zeros(int(onset*self.sr)), 
                                self.play_note(note, offset-onset, play=False))
            if wav.size < int(offset*self.sr):
                wav = np.append(wav, np.zeros(int(offset*self.sr)-wav.size))
            wav += sub_wav
        sd.play(wav, self.sr)
        sd.wait()
        return wav

    

    def play_midi(self, midi):
        pass
            


    def play_note(self, note, length, velocity, play=True):
        """
        [input]
            note:       int, from 21 to 108
            length:     length of time for the note
            velocity:   velocity of the note
            play:       whether output the sound
        [output]
            wav_out:    the waveform of note with specific length 
        """
        wav = self.notes[note]
        len_wav = wav.size
        mat = librosa.stft(wav, n_fft=3072, hop_length=512, win_length=2048)
        ratio = len_wav / mat.shape[1]
        pad_idx = np.argmax(np.sum(abs(mat), axis=0))

        """
        audio phase
            (delay-)attack:           [0: pad_idx]
            (decay-keep-)release:     [pad_idx: ]
        """
        release = mat[:, pad_idx:]
        mat = mat[:, :pad_idx]
        len_release = release.shape[1]
        n_pads = int(self.sr * length / ratio) - pad_idx
        pad_r, pad_s = n_pads // len_release, n_pads % len_release
        for i in range(len_release):
            cnt = pad_r
            if pad_s:
                pad_s -= 1
                cnt += 1
            for _ in range(cnt):
                mat = np.append(mat, release[:, i: i+1], axis=1)
        wav_out = librosa.istft(mat, hop_length=512, win_length=2048)
        # plt.pcolormesh(abs(mat))
        # plt.show()
        if play:
            sd.play(wav_out, self.sr)
            sd.wait()
        return wav_out



if __name__ == '__main__':
    piano = Instrument(type='piano')
    for i in range(40, 80):
        piano.play_note(i, 2)