import librosa
import numpy as np
import soundfile
import os
import pygame


def transcribe_vocal(transcriber, vocal, midi_path, instrument):
    """
    transriber:     VocalTranscritiopn object
    vocal:          wavform (np.array) | audio path (str)
    midi_path:      path to save the output midi file
    """
    if isinstance(vocal, str):
        vocal, _ = librosa.core.load(vocal, sr=16000, mono=True)
    elif not isinstance(vocal, np.ndarray):
        raise ValueError("You should input either wavform as numpy array or the path of audio file.")
    print('---Begin vocal transcription...')
    transcriber.transcribe(vocal, midi_path, instrument)
    print('---Done vocal transcription!')



def separate(separator, audio, save_dir=None):
    """
    [input]
        separator:      spleeter.Separator object
        audio:          wavform (np.array) | audio path (str)
        save_dir:       directory for saving result, save nothing if save_dir is None
    [output]
        vocal:          waveform (numpy array)
        accompaniment:  waveform (numpy array)
    """
    if isinstance(audio, str):
        audio, _ = librosa.core.load(audio, sr=44100, mono=True)
    elif not isinstance(audio, np.ndarray):
        raise ValueError("You should input either wavform as numpy array or the path of audio file.")
    waveform = np.expand_dims(audio, axis=1)
    
    print("---Begin separating vocal and accompaniment...")
    prediction = separator.separate(waveform)
    accompaniment = librosa.core.to_mono(prediction["accompaniment"].T)
    accompaniment = np.clip(accompaniment, -1.0, 1.0)

    # vocal2piano
    vocal = librosa.core.to_mono(prediction["vocals"].T)
    vocal = np.clip(vocal, -1.0, 1.0)
    print("---Done separating vocal and accompaniment!")

    # saving results
    if save_dir:
        vocal_path = os.path.join(save_dir, 'vocal.wav')
        accompaniment_path = os.path.join(save_dir, 'accompaniment.wav')
        soundfile.write(vocal_path, vocal, 44100, subtype='PCM_16')
        soundfile.write(accompaniment_path, accompaniment, 44100, subtype='PCM_16')
        print("---save vocal and accompaniment to directory '%s'"%save_dir)

    return vocal, accompaniment



def play_midi(midi_path):
    """
    [input]
        midi_path:      path for midi file
        play:           whether to play the output audio
    [output]
        wav:            waveform (numpy array)
    """
    freq = 44100
    bitsize = -16
    channels = 2
    buffer = 1024
    pygame.mixer.init(freq, bitsize, channels, buffer)#初始化
    pygame.mixer.music.load(midi_path)

    pygame.mixer.music.set_volume(1)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
        # pos = pygame.mixer.music.get_pos()
        # command  = input('\r%d'%int(pos))
        # if command == 'r':   
        #     pygame.mixer.music.set_pos(pos+2)
        # elif command == 'l':
        #     pygame.mixer.music.set_pos(pos-2)
        # elif command == 'p':
        #     pygame.mixer.music.pause()
        # elif command.isnumeric():
        #     volume = float(command)
        #     if 0 <= volume and volume <= 1:
        #         pygame.mixer.music.set_volume(volume)
    


if __name__ == '__main__':
    play_midi("outputs/2022-05-11-15-13-25-晴天_周杰伦/vocal.mid")