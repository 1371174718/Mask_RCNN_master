import PIL.Image
import numpy as np
import cv2

num = np.zeros([2,2,3])
num[:,1,0] = 1
num[:,1,0] = 2
print(num)
num1 = np.logical_not(num[:,:,0])
print(num1)
num2 = num[:,:,1]*num1
print(num2)