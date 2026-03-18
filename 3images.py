import cv2
import numpy as np
import matplotlib.pyplot as plt

def stitch_images(img1, img2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features and compute their descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Find matches between the two images using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = sorted(matches, key = lambda x:x.distance)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # Finding outliers and Homography with the RANSAC algorithm
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Apply panorama correction
    width = img2.shape[1] + img1.shape[1] 
    height = img2.shape[0] + img1.shape[0] 
    H_inv = np.linalg.inv(H)
    test = cv2.warpPerspective(img2, H_inv, (width, height))
    result = cv2.warpPerspective(img2, H_inv, (width, height))
    result[0:img1.shape[0], 0:img1.shape[1]] = img1
    return result

def stitchfinal_images(img1, img2):
    # Convert the images to grayscale
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # Detect SIFT features and compute their descriptors
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    # Find matches between the two images using BFMatcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    good_matches = sorted(matches, key = lambda x:x.distance)

    # Finding outliers and Homography with the RANSAC algorithm
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    H, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # Apply panorama correction
    width = (img2.shape[1] + img1.shape[1]) 
    height = (img2.shape[0] + img1.shape[0]) 
    H_inv = np.linalg.inv(H)
    result = cv2.warpPerspective(img2, H_inv, (width, height))
    result[0:img1.shape[0]//2,0:img1.shape[1]//2] = img1[0:img1.shape[0]//2,0:img1.shape[1]//2]
    return result

# Load the three images
filename1 =('D:\image1.jpg')
filename2 =('D:\image2.jpg')
filename3 =('D:\image3.jpg')
img1 = cv2.imread(filename1)
img2 = cv2.imread(filename2)
img3 = cv2.imread(filename3)

# Stitch the first two images
result1 = stitch_images(img1, img2)

# Stitch the last two images
result2 = stitch_images(img2, img3)

# Stitch the final result with the third image
result = stitchfinal_images(result1, result2)

# Show the final result
plt.figure(figsize=(40,20))
plt.imshow(result)
plt.axis('off')
plt.show()

#de katafera na kanw to blending