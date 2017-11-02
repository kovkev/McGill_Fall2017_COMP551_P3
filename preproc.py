import numpy   as np
import scipy.misc # to visualize only\
from skimage import measure
import cv2

def openfile(file):
    lines = open(file, 'rb').readlines()
    return lines

def linetoImage(line):
    line = line.split(',')
    x = np.array(line).astype(float).astype(np.uint8)
    im = x.reshape(-1, 64)
    im[im<240] = 0
    im[im>=240] = 255
    return im

def findcharacters(image):
    _, contours, hierarchy = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    areas = [0] * len(contours)
    for x in range(len(contours)):
        areas[x] = cv2.contourArea(contours[x])
    threelargestareas = sorted(areas, reverse=True)[:3]

    for contour in contours:
        area = cv2.contourArea(contour)
        if area in threelargestareas:
            (x,y,w,h) = cv2.boundingRect(contour)
            cv2.rectangle(image, (x,y), (x+w,y+h), (255, 255, 255), 1)


lines = openfile('train_x.csv')
image = linetoImage(lines[28])
scipy.misc.imshow(image)

findcharacters(image)
scipy.misc.imshow(image)

#im2 = findobjects()
