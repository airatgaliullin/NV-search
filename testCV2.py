# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 21:29:37 2017

@author: Airtat
based on https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
"""
#Hello!

#56

import cv2
import numpy as np
from skimage import measure
from imutils import contours
import imutils

def circles(name, image2, picture_name):
    imgC=cv2.imread(picture_name)
    params = cv2.SimpleBlobDetector_Params()
    params.minDistBetweenBlobs=0
    params.filterByColor=False
    params.blobColor=255
    
    params.minThreshold = 0
    params.maxThreshold=1000
    
    params.filterByArea = False
    params.minArea =100
    params.maxArea=100000
   
    params.filterByCircularity = False
    params.minCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    detector = cv2.SimpleBlobDetector_create(params)
   
    keypoints = detector.detect(image2)
    # print keypoints[0].size
    # print keypoints[1].size
    # print keypoints[2].size

    
    im_with_keypoints = cv2.drawKeypoints(image2, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    print 'number of NVs', len(keypoints)
    cv2.imshow(name, im_with_keypoints)


def mincircles(name, image1,picture_name,target_radius):
# find the contours in the mask, then sort them from left to
# right
    cv2.imshow('image1',image1)
    

    x_coordinates=[]
    y_coordinates=[]

    imgC=cv2.imread(picture_name)
    cnts= cv2.findContours(image1.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if imutils.is_cv2() else cnts[1]
    if len(cnts)>0:
        cnts = contours.sort_contours(cnts)[0]
    print 'total number of blobls', len(cnts)
# loop over the contours
    for (i, c) in enumerate(cnts):
# draw the bright spot on the im
        (x, y, w, h) = cv2.boundingRect(c)
        ((cX, cY), radius) = cv2.minEnclosingCircle(c)
        print 'radius_1',radius
        if int(radius)>0.5*target_radius and  int(radius)<3*target_radius:
            pixel=image1[int(cY)][int(cX)]
            print 'radius_2',radius
            if pixel==255:
                    cv2.circle(imgC, (int(cX), int(cY)), int(radius),(0, 0, 255), 0)
                    x_coordinates.append(cX)
                    y_coordinates.append(cY)
       # cv2.putText(image1, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
 
# show the output image
    
    cv2.imshow(name, imgC)
    return x_coordinates,y_coordinates    



picture='sofwinres.png'
img = cv2.imread(picture,0)
num_of_pix=len(img[0])
image_size=1.7 #scan area size in um

nv_size=0.5
target_radius=num_of_pix/image_size*nv_size*0.5
print target_radius


imgC=cv2.imread(picture)
#img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gblur = cv2.GaussianBlur(img, (95, 95), 0)

mblur= cv2.medianBlur(img,5)

print 'mean',mblur.mean()

cv2.imshow('gblur', gblur)






#cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
threshold_range=[int(mblur.mean())]


for i in threshold_range:  
    thresh = cv2.threshold(gblur, i, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
# perform a connected component analysis on the thresholded
# image, then initialize a mask to store only the "large" components
    # circles(str(i)+'before',thresh)    


    labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")
    mask2=np.zeros(thresh.shape, dtype="uint8")
    for label in np.unique(labels):
        if label==0:
            continue
        
	# otherwise, construct the label mask and count the
	# number of pixels 
        labelMask = np.zeros(thresh.shape, dtype="uint8")
        
        labelMask[labels == label] = 255
        numPixels = cv2.countNonZero(labelMask)
	

    # if the number of pixels in the component is sufficiently
	# large, then add it to our mask of "large blobs"
        if numPixels>3:
            mask=cv2.add(mask, labelMask)

        # if numPixels>4000:
        #     mask2=cv2.add(mask, labelMask)
    #cv2.imshow('mask',mask)
    # cv2.imshow(str(i)+'before',mask)
    

# method that counts the counters
    X,Y=mincircles(str(i)+'afterMINI',mask,picture,target_radius)
    # for k in xrange (mask.shape[0]): #traverses through height of the image
    #         for j in xrange (mask.shape[1]): #traverses through width of the image
    #                A=0

    # print k
    # print j
    # circles(str(i)+'after',mask,picture)

# cv2.imshow('detected circlesErode',thresh)




cv2.waitKey(0)
cv2.destroyAllWindows()