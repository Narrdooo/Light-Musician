<img src="https://github.com/ronnnhui/Light-Musician/blob/master/logo.png?raw=true" width = 400>

> Light musician is a tool to convert song to its light version. With [Light Player](https://github.com/ronnnhui/Light-Player), vocals in a song can be convert to other instruments using vst plugin.

# Environments

- Python == 3.8
- Spleeter == 2.3.0
- Torch == 1.11.0+cu113

# Vocal transcription
Modify [Piano transcription](https://github.com/bytedance/piano_transcription) for vocal transcription.

## Dataset
MIR-ST500 is used to train the vocal transcription network.
see [here](https://github.com/ronnnhui/Light-Musician/tree/master/vocal_transcription/MIR-ST500_20210206) for more information.

# Usage
```python
from light_musician import Light_Musician

light_musician = Light_Musician()
light_musician.convert('晴天.mp4')
```

## Use Pygame as gui 
```python
from pygame_gui import gui

gui(vocal_midi_path, accompaniment_path)
```


## use [Light Player](https://github.com/ronnnhui/Light-Player)
Light player is a midi file player hosting vst plugin. It's developed on [Juce](https://github.com/juce-framework/JUCE) framework.<br>
See [here](https://github.com/ronnnhui/Light-Player) for more information.

<img src="https://github.com/ronnnhui/Light-Player/blob/master/sceenshot.png?raw=true" width = 600>

# Demos
See demons in [my video]() in Bilibili.
