import cv2
import numpy as np


def disk_pic_shaping(img_path):
    # disk picture loading
    disk = cv2.imread(img_path)
    width_j, height_j, _ = disk.shape
    # chopping
    size = min(width_j, height_j)
    start_width = (width_j - size) // 2
    start_height = (height_j - size) // 2
    disk = disk[start_width: start_width+size, start_height: start_height+size, :]
    # scaling
    size = 360
    disk = cv2.resize(disk, (size, size))

    # drawing outer circle
    circleOut = np.zeros((size, size, 3), np.uint8)
    circleOut = cv2.circle(circleOut, (size // 2, size // 2), size // 2, (1, 1, 1), -1, )
    disk *= circleOut
    
    # drawing inner circle
    disk = cv2.circle(disk, (size//2, size//2), 80, (0,0,0), -1)
    
    
    idx = len(img_path) - 1
    while idx >= 0 and img_path[idx] != '.':    
        idx -= 1 
    out_path = "%s_disk.png"%(img_path[:idx])
    print("modified picture will be saved to %s."%out_path)
    cv2.imwrite(out_path, disk)


if __name__ == '__main__':
    disk_pic_shaping('./res/pics/jennie.jpg')