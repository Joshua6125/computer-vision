import cv2
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(128)-64
X, Y = np.meshgrid(x,x)
f = (100*(np.hypot(X,Y)<32)).astype(np.uint8)

plt.clf()
plt.imshow(f);
plt.gray()

sift = cv2.xfeatures2d.SIFT_create()
kps, dscs = sift.detectAndCompute(f, mask=None)

ax = plt.gca()

for kp in kps:
    ax.add_artist(plt.Circle((kp.pt), kp.size/2, color='green', fill=False))

plt.show()
