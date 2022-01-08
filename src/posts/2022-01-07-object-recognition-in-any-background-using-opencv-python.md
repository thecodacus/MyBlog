---
layout: blog
title: Object Recognition In Any Background Using OpenCV Python
date: 2017-03-26T16:35:45.601Z
category: computer vision
featuredImage: /images/uploads/detected-with-yolo-schreibtisch-mit-objekten.jpeg
redirect_from:
    - /2017/03/26/object-recognition-using-opencv-python/
---

In my previous posts we learnt how to use classifiers to do [Face Detection](https://web.archive.org/web/20201125043415/https://thecodacus.com/opencv-python-face-detection/) and how to create a [dataset](https://web.archive.org/web/20201125043415/https://thecodacus.com/opencv-face-recognition-python-part1/) to [train](https://web.archive.org/web/20201125043415/https://thecodacus.com/face-recognition-opencv-train-recognizer/) a and [use](https://web.archive.org/web/20201125043415/https://thecodacus.com/face-recognition-loading-recognizer/) it for Face Recognition, in this post we are will looking at how to do Object Recognition to recognize an object in an image ( for example a book), using SIFT/SURF Feature extractor and Flann based KNN matcher,

Many of you already asked me for a tutorial on this, So here it is

## Lets Do Object Recognition

So before we start, Lets create a new folder in our project folder, I am naming it as “Object Rec” Inside this we are going to save all our stuff about object recognition,

Now create a folder inside that name it “TrainingData”, this is where we are going ti save out training image which we are going to recognize in the live webcam

Now we need a sample image which we will be going to track or recognise

![image of raspberry pi box for Object recognition](/images/uploads/20200210_180258-scaled.jpeg)

Raspberry Pi Box

So i am using this as my training image, after you get your training image that you want to track, place that file and rename it to “TrainImg.jpg”.

# Lets Start Coding

So we are ready with the setup, Now lets open your favourite python editor, and jump straight to object recognition code

First lets insert the libraries which is just the numpy and the cv2 library

```python
import cv2
import numpy as np
```

After adding this we need to add the SIFT/SURF feature extractor, which will extract some distinct features from images as key points for our object recognition

and we will also need a feature feature matcher which will match the features from the sample/ training image with the current image from the webcam,

```python
detector=cv2.SIFT()
FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})
```

In the above code we initialized the feature extractor SIFT as detector and feature matcher flann.

Now lets load the training image from the folder that we created earlier, and extract its features first,

```python
trainImg=cv2.imread("TrainingData/TrainImg.jpg",0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)
```

In the above code we used cv2.imread() to load the image which we saved earlier, and next we used the feature extractor to detect features and stored them in two variables one is trainKP which is the list of key points / coordinates of the features, and other in trainDesc which is list of descriptions of the corresponding key points

We will need these to to find visually similar objects in our live video,

Now lets initialize the camera of the VideoCapture object

```python
cam=cv2.VideoCapture(0)
```

## Start The Main LOOP!!

Now that we are done with all the preparation, we can start the main loop to start doing the main work

```python
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)
```

In the above code, we first captured a frame from the camera, then converted it to gray scale, then we extracted the features like we did in the training image,after that we used the **_flann_** feature matcher to match the features in both images, and stored the matches results in **_matches_** variable

here flann is using knn to match the features with k=2, so we will get 2 neighbors

After this we have to filter the matches to avoid false matches

```python
goodMatch=[]
    for m,n in matches:
        if(m.distance&lt;0.75*n.distance):
            goodMatch.append(m)
```

In the above code we created an empty list named **_goodMatch_**  and the we are checking distance from the most neatest neighbor m and the next neatest neighbor n, and we are considering the match is a good match if the distance from point **_“m”_** is less the 70% of the distance on point “**_n_**” and appending that point to **_“goodMatch”_**

We also need to make sure that we have enough feature matches to call these a match, for that we are going to set a threshold “**_MIN_MATCH_COUNT_**” and if the number of matches are greater than then value then only we are going to consider them as match

```python
    MIN_MATCH_COUNT=30
    if(len(goodMatch)&gt;=MIN_MATCH_COUNT):
        tp=[]
        qp=[]

        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)

        tp,qp=np.float32((tp,qp))

        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)

        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
    else:
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
```

So in the above code we first check if number of matched features are more than the minimum number of threshold then we are going to do further operation,

Now we created two empty to get the coordinates of the matched features from the training image as well as from the query image, and converted that to numpy lists

the we used cv2.findHomography(tp,qp,cv2.RANSAC,3.0) to find transformation constant to translate points from training points to query image points,

now we want to draw border around the object so we want to get the coordinates of the border corners from the training image, which are (0,0), (0,h-1), (w-1,h-1),(w-1,0) where h,w is the height and width of the training image

Now using the transformation constant “H” that we got from earlier we will translate the coordinates from training image to query image,

finally we are using “cv2.polylines()” to draw the borders in the query image

Lastly if the number of features are less that the minimum match counts then we are going to print it in the screen in the else part

Now lets display the image, and close the window if loop ends

```python
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
```

## Complete Object Recognition Code

```python
import cv2
import numpy as np
MIN_MATCH_COUNT=30

detector=cv2.SIFT()

FLANN_INDEX_KDITREE=0
flannParam=dict(algorithm=FLANN_INDEX_KDITREE,tree=5)
flann=cv2.FlannBasedMatcher(flannParam,{})

trainImg=cv2.imread("TrainingData/TrainImg.jpeg",0)
trainKP,trainDesc=detector.detectAndCompute(trainImg,None)

cam=cv2.VideoCapture(0)
while True:
    ret, QueryImgBGR=cam.read()
    QueryImg=cv2.cvtColor(QueryImgBGR,cv2.COLOR_BGR2GRAY)
    queryKP,queryDesc=detector.detectAndCompute(QueryImg,None)
    matches=flann.knnMatch(queryDesc,trainDesc,k=2)

    goodMatch=[]
    for m,n in matches:
        if(m.distance&lt;0.75*n.distance):
            goodMatch.append(m)
    if(len(goodMatch)&gt;MIN_MATCH_COUNT):
        tp=[]
        qp=[]
        for m in goodMatch:
            tp.append(trainKP[m.trainIdx].pt)
            qp.append(queryKP[m.queryIdx].pt)
        tp,qp=np.float32((tp,qp))
        H,status=cv2.findHomography(tp,qp,cv2.RANSAC,3.0)
        h,w=trainImg.shape
        trainBorder=np.float32([[[0,0],[0,h-1],[w-1,h-1],[w-1,0]]])
        queryBorder=cv2.perspectiveTransform(trainBorder,H)
        cv2.polylines(QueryImgBGR,[np.int32(queryBorder)],True,(0,255,0),5)
    else:
        print "Not Enough match found- %d/%d"%(len(goodMatch),MIN_MATCH_COUNT)
    cv2.imshow('result',QueryImgBGR)
    if cv2.waitKey(10)==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
```

## Complete Video Guide

<iframe width="750" height="480"  style="width:100%;" src="https://www.youtube.com/embed/vwEdmG0q8UI" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

### Updates: Github links:

For this code: you can visit:[ https://github.com/thecodacus/object-recognition-sift-surf](https://github.com/thecodacus/object-recognition-sift-surf)
