#!/usr/bin/python

# Standard imports
import cv2
import numpy as np;

# Read image
im = cv2.imread("2.png")
#im = cv2.medianBlur(im,5)
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()


params.minDistBetweenBlobs=0

params.filterByColor=False
params.blobColor=255




# Change thresholds
params.minThreshold = 0
params.maxThreshold=10000


# Filter by Area.
params.filterByArea = True
params.minArea =120
params.maxArea=2000

# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.1



# Filter by Convexity
params.filterByConvexity = False
params.minConvexity = 0.87

# Filter by Inertia
params.filterByInertia = False
params.minInertiaRatio = 0.01

# Create a detector with the parameters
detector = cv2.SimpleBlobDetector_create(params)


# Detect blobs.
keypoints = detector.detect(im)

# Draw detected blobs as red circles.
# cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
# the size of the circle corresponds to the size of blob
im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

# Show blobs
cv2.imshow("KeypointsLight", im_with_keypoints)

params.maxThreshold=10000
params.filterByColor=True
params.blobColor=0

detector = cv2.SimpleBlobDetector_create(params)
keypointsD = detector.detect(im)

im_with_keypointsD = cv2.drawKeypoints(im, keypointsD, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
cv2.imshow("KeypointsDark", im_with_keypointsD)
# Show blobs

print ('End')
print('End3 ')

cv2.waitKey(0)