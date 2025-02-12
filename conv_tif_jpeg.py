import cv2
import matplotlib.pyplot as plt
import os
import numpy as np

input_dir = ""
output_dir = ""

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

photos = os.listdir(input_dir)
for photo in photos:
    
    n = 1
    photoname = photo.split(".")
    savename = output_dir + "/" + photoname[0] + ".jpg"
    img = cv2.imread(input_dir + "/" + photo)
    cv2.imwrite(savename,img)
    if (n % 10 ==0):
        print(savename)
    n=n+1

print("Successed!")