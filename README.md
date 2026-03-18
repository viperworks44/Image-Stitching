Image Stitching (2 & 3 Images, python)

This project demonstrates image stitching using OpenCV for both 2-image and 3-image cases.
Images are loaded and converted to grayscale for processing.
ORB is used to detect keypoints and compute descriptors.
Feature matching is performed using a Brute-Force matcher (Hamming distance).
Matches are sorted to retain the best correspondences.
RANSAC is applied to estimate homography and remove outliers.
For 2 images, one image is warped and stitched directly with the other.
For 3 images, stitching is done sequentially to form a larger panorama.
warpPerspective is used to align images properly.
Matplotlib is used to display the final stitched results.
