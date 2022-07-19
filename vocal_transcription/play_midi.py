import pygame
import sys
freq = 44100
bitsize = -16
channels = 2
buffer = 1024
pygame.mixer.init(freq, bitsize, channels, buffer)#初始化
pygame.mixer.music.load('./mm.mid')
pygame.mixer.music.play()
pygame.display.set_mode([300,300])
while 1:
    for event in pygame.event.get():
        if event.type==pygame.QUIT:
            sys.exit()