import numpy
import cv2
import matplotlib.pyplot as plt

## TODO 3.1
## Load Image
## Show it on screen
file = 'img.png'  ## path to the image
'''
Your code here
'''
img = cv2.imread(file, cv2.IMREAD_UNCHANGED)

###############################################
## TODO 3.2
## Resize Image by a factor of 0.5
## Show it on screen
## Save as small.jpg
'''
Your code here
'''
scale = 0.5
dim = (int(img.shape[0] * scale), int(img.shape[1] * scale))
resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imwrite('small.png', resize_img)
###############################################
## TODO 3.3
## Create and save 3 single-channel images from small image
## one image each channel (r, g, b)
## Display the channel-images on screen
'''
Your code here
'''
# opencv BRG

B = img.copy()
R = img.copy()
G = img.copy()

B[:, :, 1] = 0
B[:, :, 2] = 0

R[:, :, 0] = 0
R[:, :, 1] = 0

G[:, :, 0] = 0
G[:, :, 2] = 0

cv2.imshow('blue', B)
cv2.waitKey(0)
cv2.imshow('red', R)
cv2.waitKey(0)
cv2.imshow('green', G)
cv2.waitKey(0)


fig = plt.figure()
fig.add_subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(R, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('red')

fig.add_subplot(1, 3, 2)
plt.imshow(cv2.cvtColor(G, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('green')

fig.add_subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(B, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('blue')

plt.show()
###############################################
