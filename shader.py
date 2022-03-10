import cv2 as cv
import os.path
import numpy as np
import math
import sys
import time

#arguments
#input file name, output file name, scale
if (len(sys.argv) != 4):
    print("Use: Input Filename, Output Filename, Scale for Shading(int around value of 1)")
    exit()

if (os.path.isfile(sys.argv[1]) == False):
    print("Input File Not Found")
    exit()
inputimg = cv.imread(sys.argv[1])
inputhsv = cv.cvtColor(inputimg, cv.COLOR_BGR2HSV)

def hueToShadeVec(imghues):
    #input
    #slice of image hues
    #move hue towards purple for darker looking colour
    diff, direct = closerEndVec(imghues)

    #minimize distant by factor and then apply direction 
    funco = (np.sqrt(diff) * 2 * direct)
    newhue = levelerVec(np.copy(imghues) + funco)
    return newhue

def levelerVec(img):
    #input regular adjusted hues from hueToShadeVec and output values between 0 - 180
    #neccesary to have hue values wrap around range of 180
    huem = np.copy(img)
    undermask = huem < 0
    overmask = huem > 180
    np.putmask(huem,undermask,huem+180)
    np.putmask(huem,overmask,huem-180)
    return huem

def closerEndVec(img):
    #input
    #slice of image hues
    #output vector "diff", amount to move by along hue value
    #vector "dirs" indicating which direction the hues should go
    huem = np.copy(img)
    diff = np.copy(img)
    dirs = np.copy(img)
    
    between = np.logical_and(huem >= 49, huem < 139)
    np.putmask(diff,between,139 - huem)
    np.putmask(dirs,between,1)
    
    under2 = huem < 49
    np.putmask(diff,under2,huem + 49)
    
    over2 = np.logical_not(np.logical_or(between, under2))
    np.putmask(diff,over2,huem - 139)
    
    np.putmask(dirs,np.logical_not(between),-1)
    return diff, dirs

def newVSVec(img2,scale):
    #input
    #slice of image saturation and value
    #scale to move across vector field (around 1-2 usually)
    #output new value and saturation based on vector field equation
    img = np.copy(img2)
    s = img[:,:,0]
    v = img[:,:,1]
    news = s + ((v - 128)*0.4 * scale)
    newv = v - (20 * scale)
    img[:,:,0] = news
    img[:,:,1] = newv

    #constrain values between 0 and 255
    img = np.clip(img,a_min=0,a_max=255)
    return img

t0 = time.time()
img0hsv = np.copy(inputhsv)
#convert image from uints to ints
img0hsv = np.intc(img0hsv)
masker = img0hsv.sum(axis=(2)) == 0
img0hsv[:,:,0] = hueToShadeVec(img0hsv[:,:,0])
img0hsv[:,:,1:3] = newVSVec(img0hsv[:,:,1:3],float(sys.argv[3]))
img0hsv[masker] = [0,0,0]

imgorgb = cv.cvtColor(np.uint8(img0hsv), cv.COLOR_HSV2BGR)
cv.imwrite(sys.argv[2], imgorgb)

t1 = time.time()
print("Written in ", t1-t0, " seconds")