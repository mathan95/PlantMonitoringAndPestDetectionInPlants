# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 10:25:07 2018

@author: MathanP
"""

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.image as mpimg
import cv2
from keras.applications import VGG19
#from keras.preprocessing.image import load_img
#from keras.preprocessing.image import img_to_array
from keras.applications.vgg19 import preprocess_input

###############################################################
#green-removal part#
##############################################################

img=mpimg.imread("7.jpg")  #input to the green removal. If already processed image then
A=img/255                   #If it is an already processed image then no need to divide 
plt.figure(1)

ish=A.shape

X=np.reshape(A,(ish[0]*ish[1],ish[2]))

centroids=np.array([0,1,0])
errlist1=np.zeros((X.shape[0],1),np.float64)

for p in range(0,X.shape[0],1):
    mat=X[p,:]
    mat3=mat-centroids
    err=np.sum(np.multiply(mat3,mat3))
    errlist1[p]=err
for l in range(0,10,1):
    index_min=np.argmin(errlist1)
    if (errlist1[index_min]>0.7):
        break
    centroids=centroids+X[index_min,:]
    errlist1[index_min]=1
centroids=centroids/10
errlist=np.zeros((X.shape[0],1),np.float64)

                    ###########################################################################
                    #Introduction of another loop to make it a little bit adaptive#
                    ###########################################################################
for p in range(0,X.shape[0],1):     
    mat=X[p,:]
    mat3=mat-centroids
    err=np.sum(np.multiply(mat3,mat3))
    errlist[p]=err
    
threshold=0.2
errlist=np.greater(errlist,threshold)
X=np.multiply(X,errlist)
X_out=np.reshape(X,(ish[0],ish[1],ish[2]))


#plt.imshow(X_out)
#plt.figure()
#plt.subplot(1,2,2),plt.imshow(mpimg.imread('7.jpg'))

##################################################################
#connected-component part#
##################################################################




####################################################################################################
#THIS PART IS TO REMOVE THE NOISE COMPONENTS WHICH HAS LESS THAN PIXEL NUMBER OF 100
#First this is done only to remove noise out and not to get the pest as they may have seperated to diffrent comps
#
#If we dilate without removing noises then they can form as a component
##################################################################################################


X_out2=np.uint8(X_out*255)

img2=cv2.cvtColor(X_out2,cv2.COLOR_BGR2GRAY)

kernel = np.ones((10,10), np.uint8)

ret, thresh = cv2.threshold(img2,0,1,cv2.THRESH_BINARY)


# You need to choose 4 or 8 for connectivity type
connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The fourth cell is the centroid matrix
centroids = output[3]

list_new=[]
for i in range(0,num_labels,1):
    list1=np.equal(labels,i)            #list with interested connected components.
    if (np.sum(np.sum(list1))>100):
        list_new.append(i)
        
#######################################################################
#For removing the noises out of the picture#
#This noise removed image will be used later to 
######################################################################
#plt.imshow(X_out)
        
shape=np.shape(img2)
for i in range(0,shape[0]):
    for j in range(0,shape[1]):
        if (labels[i,j] in list_new[1:])!= True:
            X_out[i,j,:]=0
#plt.figure()
#plt.imshow(X_out)
            
#################################################################################################
#In this morphological operation are done and then connected components are used in order to 
#localize the pests correctly
################################################################################################
X_out2=np.uint8(X_out*255)

img2=cv2.cvtColor(X_out2,cv2.COLOR_BGR2GRAY)

kernel = np.ones((10,10), np.uint8)

ret, thresh = cv2.threshold(img2,0,1,cv2.THRESH_BINARY)

                        ##################################################################
                        #using dilation to first remove any holes that seperate a same component mistakenly#
                        #then using the erosion to seperate the mistakenly joined parts#
                        #################################################################

thresh = cv2.dilate(thresh, kernel, iterations=2)
thresh=cv2.erode(thresh,kernel,iterations=3)
#thresh = cv2.dilate(thresh, kernel, iterations=1)

#plt.figure()
#plt.imshow(thresh)


# You need to choose 4 or 8 for connectivity type
connectivity = 4  
# Perform the operation
output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
# Get the results
# The first cell is the number of labels
num_labels = output[0]
# The second cell is the label matrix
labels = output[1]
# The fourth cell is the centroid matrix
centroids = output[3]

list_new=[]
for i in range(0,num_labels,1):
    list1=np.equal(labels,i)            #list with interested connected components.
    if (np.sum(np.sum(list1))>600):
        list_new.append(i)


#plt.subplot(1,2,1),plt.imshow(X_out)
#plt.subplot(1,2,2),plt.imshow(mpimg.imread('7.jpg'))

n=len(list_new)
shape=np.shape(img2)

#################################################################
#localiztion-part##
#################################################################

print(list_new)
list1=np.zeros((n,4),np.int)
list1[:,0]=shape[0]
list1[:,2]=shape[1]



for i in range(0,shape[0]-1,1):
    for j in range(0,shape[1]-1,1):
        if ((labels[i,j] in list_new)==True and (labels[i,j]!=0)):
            b=np.argmax(np.equal(list_new,labels[i,j]))
            if list1[b,0]>i:
                list1[b,0]=i
            elif list1[b,1]<i:
                list1[b,1]=i
            if list1[b,2]>j:
                list1[b,2]=j
            elif list1[b,3]<j:
                list1[b,3]=j

#########################################################################
#for plotting each connected component#
#for j in range(1,n,1):
#    filter1=np.zeros((shape[0],shape[1],3),np.float64)
#    filter1[:,:,0]=np.equal(labels,list_new[j])
#    filter1[:,:,1]=np.equal(labels,list_new[j])
#    filter1[:,:,2]=np.equal(labels,list_new[j])
#    plt.figure(j+1)
#    imb=np.multiply(X_out/255,filter1)
#    im=img/255
#    #plt.imshow(im[list1[j,0]:list1[j,1],list1[j,2]:list1[j,3],:])
#    plt.imshow(im[max(0,list1[j,0]-50):min(shape[0],list1[j,1]+50),max(0,list1[j,2]-50):min(shape[1],list1[j,3]+50),:])
    #plt.imshow(imb)
 ######################################################################

   
print(list1)   #for printing the localization results


######################################################################

######################################################################
#neural-network part#
#####################################################################

model = VGG19()
input_images=np.zeros((n-1,224,224,3),np.float64)
print(n)
X_out=X_out*255
#for j in range(1,n,1):
#    im=img
#    #im=img
#    #plt.imshow(im[list1[j,0]:list1[j,1],list1[j,2]:list1[j,3],:])
#    #plt.imshow(im[max(0,list1[j,0]-50):min(shape[0],list1[j,1]+50),max(0,list1[j,2]-50):min(shape[1],list1[j,3]+50),:])
#    img_test=X_out[max(0,list1[j,0]-100):min(shape[0],list1[j,1]+100),max(0,list1[j,2]-100):min(shape[1],list1[j,3]+100),:]
#    # load an image from file
#    #image = load_img('7.jpg', target_size=(224, 224))
#    
#    image=cv2.resize(img_test,(224,224))
#    
#    # convert the image pixels to a numpy array
#    #plt.imshow(image)
#    
#    #image = img_to_array(image)
#    # reshape data for the model
#    #image = image.reshape(( image.shape[0],image.shape[1], image.shape[2]))
#    # prepare the image for the VGG model
#    input_images[j-1,:,:,:]=image

###############################################################################
#This filtered image is not a good input for the CNN
###############################################################################
#f1=np.zeros((shape[0],shape[1],3))
#f1[:,:,1]=thresh
#f1[:,:,2]=thresh
#f1[:,:,0]=thresh
#im=np.multiply(img,f1)
###############################################################################
    
for j in range(1,n,1):
    #im=img
    #plt.imshow(im[list1[j,0]:list1[j,1],list1[j,2]:list1[j,3],:])
    #plt.imshow(im[max(0,list1[j,0]-50):min(shape[0],list1[j,1]+50),max(0,list1[j,2]-50):min(shape[1],list1[j,3]+50),:])
    img_test=X_out[max(0,list1[j,0]-25):min(shape[0],list1[j,1]+25),max(0,list1[j,2]-25):min(shape[1],list1[j,3]+25),:]
    #image = load_img('7.jpg', target_size=(224, 224))
    v=np.shape(img_test)
    # load an image from file
    ###############################################################################################
    #
    
    if ((v[0]<222)and(v[1]<222)):
        img_test2=np.zeros((224,224,3),np.float64)
        w=np.int(np.floor((224-v[0])/2))
        x=np.int(np.floor((224-v[1])/2))
        img_test2[w:w+v[0],x:x+v[1],:]=img_test
    elif (v[0]<222):
        img_test2=np.zeros((224,v[1],3),np.float64)
        w=np.int(np.floor((224-v[0])/2))
        img_test2[w:w+v[0],:,:]=img_test
    elif (v[1]<222):
        img_test2=np.zeros((v[0],224,3),np.float64)
        x=np.int(np.floor((224-v[1])/2))
        img_test2[:,x:x+v[1],:]=img_test
    else:
        img_test2=img_test
        
    image=cv2.resize(img_test2,(224,224))
    
    # convert the image pixels to a numpy array
    #plt.imshow(image)
    
    #image = img_to_array(image)
    # reshape data for the model
    #image = image.reshape(( image.shape[0],image.shape[1], image.shape[2]))
    # prepare the image for the VGG model
    input_images[j-1,:,:,:]=image

print(np.shape(input_images))

input_images1 = preprocess_input(input_images)
    # predict the probability across all output classes
yhat = model.predict(input_images1)

#X_out=X_out/255
print(np.shape(yhat))    
for k in range(1,n,1): 
    probability=np.sum(yhat[k-1,300:323])
    if (probability>0.25):
        plt.figure(k)
        im=img/255
        plt.imshow(im[max(0,list1[k,0]-25):min(shape[0],list1[k,1]+25),max(0,list1[k,2]-25):min(shape[1],list1[k,3]+25),:])