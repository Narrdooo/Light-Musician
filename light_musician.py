import librosa
import os
import time
import numpy as np
from spleeter.separator import Separator
from pygame_gui import gui

from vocal_transcription.inference import VocalTranscritiopn
from tools import separate, separate, transcribe_vocal
from vocal_transcription.utils import mkdir


class Light_Musician:
    def __init__(self, instrument=None) -> None:
        self.separator = Separator('spleeter:2stems')
        self.transcriber = VocalTranscritiopn('./vocal_transcription/checkpoints/2022-07-14-12-24-34/7999_iterations.pth')
        if not instrument:
            self.instrument = 'piano'
        else:
            self.instrument = instrument


    def convert(self, song, instrument=None):
        """
        song:          wavform (np.array) | song path (str)
        """
        if isinstance(song, str):
            if not os.path.exists(song):
                print("Aduio file not Found!")
                return
            file_name = song.split('/')[-1]
            file_name = file_name[:file_name.find('.')]
            song = self.read_audio(song)
        elif isinstance(song, np.ndarray):
            file_name = "numpy_array_wav"
        else:
            raise ValueError("You should input either wavform as numpy array or the path of audio file.")
        

        save_dir = "outputs/%s-%s"%(time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime()), file_name) 
        print("---Output will save to directory '%s'."%save_dir)
        mkdir(save_dir)
        
        vocal, _ = separate(self.separator, song, save_dir=save_dir)

        # output from spleeter is 44100 hz
        # change it to 16000 for intput to transcriber
        vocal = librosa.core.resample(vocal, orig_sr=44100, target_sr=16000)
        vocal_midi_path = os.path.join(save_dir, 'vocal.mid')

        # transribing vocal
        if instrument is None:  instrument = self.instrument
        instrument_id = self.instrument2id(instrument)
        transcribe_vocal(self.transcriber, vocal, vocal_midi_path, instrument_id)

        # start pygame gui
        accompaniment_path = os.path.join(save_dir, 'accompaniment.wav')
        #self.gui(vocal_midi_path, accompaniment_path)





    def read_audio(self, audio_path):
        audio, sr = librosa.core.load(audio_path, sr=None, mono=True)
        if sr != 44100:
            audio = librosa.core.resample(y=audio, orig_sr=sr, target_sr=44100)
        return audio


    def set_instrument(self, instrument):
        self.instrument = instrument


    def instrument2id(self, instrument):
        # return 57
        return 80


if __name__ == '__main__':
    light_musician = Light_Musician()
    light_musician.convert('./res/talk.m4a');
    