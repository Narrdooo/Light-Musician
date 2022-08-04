<br>

<img src="https://github.com/ronnnhui/Light-Musician/blob/master/logo.png?raw=true" width = 400>

> Light musician is a tool to convert song to its light version. With [Light Player](https://github.com/ronnnhui/Light-Player), vocals in a song can be convert to other instruments using vst plugin.

# Environments

- Python == 3.8
- Spleeter == 2.3.0
- Torch == 1.11.0+cu113

# Vocal transcription
Modify [Piano transcription](https://github.com/bytedance/piano_transcription) for vocal transcription.<br>

## Pretrained model set-up
1. Download pretrained model here: https://pan.quark.cn/s/d8c198dca1ff
2. Put it as path "./test.pth"

## Dataset
MIR-ST500 is used to train the vocal transcription network.
see [here](https://github.com/ronnnhui/Light-Musician/tree/master/vocal_transcription/MIR-ST500_20210206) for more information.

# Usage
>If you have any trouble running the projects, feel free to open a new issue.
```python
from light_musician import Light_Musician

light_musician = Light_Musician()
light_musician.convert('晴天.mp4')
```

## Use Pygame as gui
By this way, u will be using audio engine in pygame.
Set up instrument u want by its id (check "./midi_instrument.py")

```python
from pygame_gui import gui

gui(vocal_midi_path, accompaniment_path)
```


## use [Light Player](https://github.com/ronnnhui/Light-Player)
Light player is a midi file player hosting vst plugin. It's developed on [Juce](https://github.com/juce-framework/JUCE) framework.<br>
See [here](https://github.com/ronnnhui/Light-Player) for more information.

<img src="https://github.com/ronnnhui/Light-Player/blob/master/sceenshot.png?raw=true" width = 600>

# Demos
See demons in [my video](https://www.bilibili.com/video/bv1BY4y1A7W3?vd_source=a9916b35ed8bc012bb6a374a036216cf) in Bilibili.
