import cv2
import os
 
# create a funtion model 
def data(folder):
    images =[] # put image in an array
    for filename in os.listdir(folder)