import cv2
import matplotlib.pyplot as plt
import numpy as np

def plot_corners(corners, **kwargs):
    plt.plot(corners[:,0], corners[:,1], 'o', **kwargs)

im1_orig = cv2.imread('data/IMG_20220502_174707.jpg')
im2_orig = cv2.imread('data/IMG_20220502_174709.jpg')
im1 = cv2.cvtColor(im1_orig, cv2.COLOR_BGR2GRAY)
im2 = cv2.cvtColor(im2_orig, cv2.COLOR_BGR2GRAY)
im1 = cv2.equalizeHist(im1)
im2 = cv2.equalizeHist(im2)

corners = cv2.goodFeaturesToTrack(im1, 4000, qualityLevel=0.005, minDistance=100)
corners = corners[:,0,:]
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.0001)
corners = cv2.cornerSubPix(im1, corners, (5, 5), (-1, -1), criteria)

corners2,mask,dist = cv2.calcOpticalFlowPyrLK(im1, im2, corners, None, maxLevel=2)
mask = np.reshape(mask, -1).astype(bool)

corners = corners[mask]
corners2 = corners2[mask]

H,mask = cv2.findHomography(corners2, corners)

result = cv2.warpPerspective(im2, H, im2.shape[::-1])

im2_transformed = cv2.warpPerspective(im2_orig, H, (im2_orig.shape[1], im2_orig.shape[0]))
cv2.imwrite('data/IMG_20220502_174709_aligned.jpg', im2_transformed)
