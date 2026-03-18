import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the two images
filename1 =('D:\image1.jpg')
filename2 =('D:\image2.jpg')
img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)

# Convert the images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Detect SIFT features and compute their descriptors
##########de mporousa na xrhsimopoihsw thn sift library opote evala orb
##########sift = cv2.xfeatures2d.SIFT_create()
##########kp1, des1 = sift.detectAndCompute(gray1, None)
##########kp2, des2 = sift.detectAndCompute(gray2, None)

orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(gray1, None)
kp2, des2 = orb.detectAndCompute(gray2, None)

# Draw the SIFT features on the images
img1_with_features = cv2.drawKeypoints(img1, kp1, None)
img2_with_features = cv2.drawKeypoints(img2, kp2, None)

# Show the SIFT features
plt.imshow(img1_with_features)
plt.show()
plt.imshow(img2_with_features)
plt.show()

# Find matches between the two images using BFMatcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
good_matches = sorted(matches, key = lambda x:x.distance)

img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:50], None, flags=2)
plt.imshow(img_matches)
plt.show()

# Finding outliers with the RANSAC algorithm
src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
H,_ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# Apply panorama correction
width = img2.shape[1] + img1.shape[1]
height = img2.shape[0] + img1.shape[0]
H_inv = np.linalg.inv(H)
result = cv2.warpPerspective(img2, H_inv, (width, height))
plt.imshow(result)
plt.show()
result[0:img1.shape[0], 0:img1.shape[1]] = img1

plt.figure(figsize=(20,10))
plt.imshow(result)
plt.axis('off')
plt.show()