import pygame
import cv2
import numpy as np
import sys

def gui(vocal_midi_path, accompaniment_path):
    """
    vocal_midi_path:        path of vocal melody in midi file
    accompaniment_path:     path of accompaniment file
    """
    pygame.init()
    width = 900
    height= 600
    fps = 60  # 设置刷新率，数字越大刷新率越高
    fcclock = pygame.time.Clock()
    play = True
    text_size = 20
    text_time = 30

    pygame.mixer.init()
    pygame.mixer.music.load(vocal_midi_path)
    accompaniment = pygame.mixer.Sound(accompaniment_path)
    pygame.mixer.set_num_channels(2)
    pygame.mixer.music.play()
    pygame.mixer.Channel(1).play(accompaniment)

    gui = pygame.display.set_mode([width, height])
    pygame.display.set_caption('light musician')
    circle_center = (314, 300)
    
    # background loading
    bg_1 = pygame.image.load('./res/pics/bg.jpg').convert()
    bg_2 = bg_1.copy()
    gui.blit(bg_2, (0, 0))
    pygame.draw.circle(bg_1, (0, 255, 0), circle_center, 187)
    bg_1.set_colorkey((0, 255, 0))
    
    # disk image loading
    disk_size = 380
    disk = cv2.imread('./res/pics/jennie.jpg')
    disk = cv2.cvtColor(disk, cv2.COLOR_BGR2RGB)
    ori_w = disk.shape[0]
    ori_h = disk.shape[1]
    ori_size = min(ori_w, ori_h)
    ratio = disk_size / ori_size
    disk_w = int(ori_w * ratio)
    disk_h = int(ori_h * ratio)
    disk = cv2.resize(disk, (disk_h, disk_w))
    angle = 0
    angle_speed = 0.3

    # begin running
    while True:
        pygame.display.update()
        change = False
        text = None
        for event in pygame.event.get():
            # user interaction
            if event.type == pygame.QUIT or \
                event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.mixer.Channel(1).stop()
                pygame.mixer.music.stop()
                pygame.mixer.quit()
                sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                play = not play
                change = True
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_UP:
                volume_1 = pygame.mixer.music.get_volume()
                volume_2 = pygame.mixer.Channel(1).get_volume()
                if volume_1 == 1 or volume_2 == 1:
                    font = pygame.font.Font('freesansbold.ttf', text_size)
                    text = font.render('already maximum volume!', 
                                        True, 
                                        (0, 0, 0), 
                                        (255, 255, 255))
                else:
                    pygame.mixer.music.set_volume(volume_1 + 0.1)
                    pygame.mixer.Channel(1).set_volume(volume_2 + 0.1)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN:
                volume_1 = pygame.mixer.music.get_volume()
                volume_2 = pygame.mixer.Channel(1).get_volume()
                if volume_1 == 0 or volume_2 == 0:
                    font = pygame.font.Font('freesansbold.ttf', text_size)
                    text = font.render('already minimum volume!', 
                                        True, 
                                        (0, 0, 0), 
                                        (255, 255, 255))
                else:
                    pygame.mixer.music.set_volume(volume_1 - 0.1)
                    pygame.mixer.Channel(1).set_volume(volume_2 - 0.1)
                    

        if play:
            # rotate
            cX = disk_h // 2
            cY = disk_w // 2
            M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
        
            # compute the new bounding dimensions of the image
            nW = int((disk_h * sin) + (disk_w * cos))
            nH = int((disk_h * cos) + (disk_w * sin))
            # adjust the rotation matrix to take into account translation
            M[0, 2] += (nW / 2) - cX
            M[1, 2] += (nH / 2) - cY
            # perform the actual rotation and return the image
            tmp  = cv2.warpAffine(disk, M, (nW, nH),borderValue=(255,255,255))
            tmp_w = tmp.shape[0]
            tmp_h = tmp.shape[1]
            sur_disk = pygame.Surface((tmp_w, tmp_h))
            pygame.surfarray.blit_array(sur_disk, tmp)
            pygame.draw.circle(sur_disk, (0, 255, 0), (tmp_w//2, tmp_h//2), 72)
            sur_disk.set_colorkey((0, 255, 0))
            disk_rect = sur_disk.get_rect(center=circle_center)
            gui.blit(sur_disk, disk_rect)
            gui.blit(bg_1, (0, 0))
            pygame.display.flip()  # 刷新窗口
            angle += angle_speed
            
        if change:
            if play:
                pygame.mixer.music.unpause()
                pygame.mixer.Channel(1).unpause()
            else:
                pygame.mixer.Channel(1).pause()
                pygame.mixer.music.pause()
        
        # if text is not None:
        #     gui.blit(text, (350, 280))
        #     pygame.display.flip()  # 刷新窗口
            
        fcclock.tick(fps)