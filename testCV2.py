# -*- coding: utf-8 -*-
"""
Created on Sat Sep 09 21:29:37 2017


based on https://www.pyimagesearch.com/2016/10/31/detecting-multiple-bright-spots-in-an-image-with-python-and-opencv/
"""
#Hello!

#56



import cv2
import numpy as np
from skimage import measure
from imutils import contours
import imutils
import os


class Image_processor:

    def __init__(self):
        self.threshold_coefficient=1

    def config(self,search_range,**kw):
        if search_range==5:
            self.threshold_coefficient=1.1
        else:
            self.threshold_coefficient=1.3





    def crop_image(self, folder,filename):


        img=cv2.imread(os.path.join(folder,filename))
        

        gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        invert_gray=cv2.bitwise_not(gray)
        _,thresh = cv2.threshold(invert_gray,1,255,cv2.THRESH_BINARY)
        contours,hierarchy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cnt = contours[0]
        x,y,w,h = cv2.boundingRect(cnt)
        crop = img[y:y+h,x:x+w]
        #cv2.imwrite(os.path.join(folder,'cropped.png'),crop)
        del img
        del gray
        return crop


        

    def find_blobs(self,folder,filename,cropped_image,search_range,z):
        
        x_c,y_c=self.get_center(filename)


        #picture=os.path.join(folder,'cropped.png')
        img=cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

        #img=cv2.imread(os.path.join(folder,'cropped.png'),0)
        num_of_pix=len(img[0])

        nv_size=1
        target_radius=num_of_pix/search_range*nv_size*0.5



        gblur = cv2.GaussianBlur(img, (175, 175), 0)
        threshold_range=[int(gblur.mean()*self.threshold_coefficient)]


        cv2.imshow('gblur',gblur)
        for i in threshold_range:
            thresh = cv2.threshold(gblur, i, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.erode(thresh, None, iterations=2)
            thresh = cv2.dilate(thresh, None, iterations=4)
            cv2.imshow('threshold',thresh)



            labels = measure.label(thresh, neighbors=8, background=255)
            mask = np.zeros(thresh.shape, dtype="uint8")
            mask2 = np.zeros(thresh.shape, dtype="uint8")

            for label in np.unique(labels):
                if label==0:
                    continue
                
                labelMask = np.zeros(thresh.shape, dtype="uint8")
                labelMask[labels == label] = 255
                numPixels = cv2.countNonZero(labelMask)

                #print 'numPixels', numPixels
                #cv2.imshow('labelMask'+str(numPixels),labelMask)


                if numPixels>target_radius**2*0.01 and numPixels<target_radius**2*8:
                    mask=cv2.add(mask, labelMask)
                mask2=cv2.add(mask2, labelMask)

                

                
            cv2.imshow('mask2',mask2)
            cv2.imshow('mask',mask)
            X_pixel,Y_pixel,imgC=self.mincircles(mask,cropped_image,target_radius)
            X,Y=self.coordinate_transform(X_pixel,Y_pixel,x_c,y_c,search_range,num_of_pix)
        
        
        name='z='+str(z)+'; '+'x_c='+str(x_c)+'; '+'y_c='+str(y_c)+'; '+'search_range='+str(search_range)+'um'
        #cv2.imshow(name, imgC)
        cv2.imwrite(os.path.join(folder,name+'.png'),imgC )
        # small_imgC=cv2.resize(imgC,(0,0),fx=float(scan_size)/zoom,fy=float(scan_size)/zoom)
        del img
        del gblur
        del thresh
        del mask
        del mask2
        del imgC

        return X,Y


    def mincircles(self,image1,cropped_image,target_radius):
        imgC=cropped_image
        x_pixel_coordinates=[]
        y_pixel_coordinates=[]

        cnts= cv2.findContours(image1.copy(), cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if imutils.is_cv2() else cnts[1]

        if len(cnts)>0:
            cnts = contours.sort_contours(cnts)[0]



# loop over the contours
# draw the bright spot on the im
        for (i, c) in enumerate(cnts):
            (x, y, w, h) = cv2.boundingRect(c)
            ((cX, cY), radius) = cv2.minEnclosingCircle(c)
            #print 'ratio', int(radius)/target_radius
            if int(radius)>0.15*target_radius and  int(radius)<1.5*target_radius:
                pixel=image1[int(cY)][int(cX)]
                if pixel==255:
                    cv2.circle(imgC, (int(cX), int(cY)), int(radius),(0, 0, 255),5, 0)
                    x_pixel_coordinates.append(cX)
                    y_pixel_coordinates.append(cY)
                    
        return x_pixel_coordinates,y_pixel_coordinates,imgC  


    def coordinate_transform(self,X_pixel,Y_pixel,x_c,y_c,search_range,num_of_pix):
        X=np.zeros(len(X_pixel))
        Y=np.zeros(len(X_pixel))
        for i in xrange(len(X_pixel)):
            X[i]=x_c+(X_pixel[i]-num_of_pix/2)*search_range/num_of_pix
            Y[i]=y_c+(-Y_pixel[i]+num_of_pix/2)*search_range/num_of_pix
        return X,Y


    def merge_images(self,image1,image2):
        print image1
        img1=cv2.imread(image1)
        img2=cv2.imread(image2)
        vis=np.concatenate((img1,img2), axis=1)
        small=cv2.resize(vis,(0,0),fx=0.5,fy=0.5)
        cv2.imshow('small',small)



    def get_center(self,filename):
        center_coordinates=[int(s) for s in filename if s.isdigit()]
        x_search=filename.split('y_c')[0]
        y_search=filename.split('y_c')[1]
        x_minus=x_search.count('-')
        y_minus=y_search.count('-')



        center_coordinates_x=[int(s) for s in x_search if s.isdigit()]
        x_c=(10*center_coordinates_x[0]*(len(center_coordinates_x)-1)+center_coordinates_x[len(center_coordinates_x)-1])*(-1)**x_minus
       
        center_coordinates_y=[int(s) for s in y_search if s.isdigit()]
        y_c=(10*center_coordinates_y[0]*(len(center_coordinates_y)-1)+center_coordinates_y[len(center_coordinates_y)-1])*(-1)**y_minus




        return x_c,y_c















# cv2.waitKey(0)
# cv2.destroyAllWindows()



        







  



