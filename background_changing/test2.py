import cv2
import numpy as np
src = cv2.imread("../data/frame/step1/frame00000.jpg")
cv2.imshow("obj", src)
width= src.shape[1]
height = src.shape[0]

img = cv2.imread("../data/background_img/yellow.jpg")
img = cv2.resize(img, (width, height))

img = np.float32(img)*1.0/255
alpha = np.zeros((width, height), np.float)
src = np.float32(src)*1.0/255
channels = cv2.split(src)
img_channels = cv2.split(img)

#Vlahos algorithm
#alpha = 1 – a1*(I_GREEN – a2*I_BLUE)
const_a1 = 0.1
const_a2 = 0.5
alpha = 1 - const_a1*(channels[1] - const_a2*channels[0])

alpha, _ = cv2.threshold(alpha, 1,1, cv2.THRESH_TRUNC)
alpha, _ = cv2.threshold(-1*alpha, 0, 0, cv2.THRESH_TRUNC)
alpha = -1 *alpha

for i in range(0,3):
    cv2.multiply(alpha, channels[i], channels[i])
    cv2.multiply(1 - alpha, img_channels[i], img_channels[i])


res_font = cv2.merge(channels)
res_back = cv2.merge(img_channels);

res = res_font + res_back
np.uint8(res*255)
res = cv2.resize(res, (1500, 800))

cv2.imshow("res",res)
cv2.waitKey(0)



