---
layout: blog
title: OpenCV Object Tracking by Colour Detection in Python
date: 2017-08-13T17:50:14.907Z
category: computer vision
featuredImage: /images/uploads/f8b90e80-d3f0-11e9-98fa-dd887a89f392.jpeg
---
Hi everyone, we have already seen lots of advanced detection and recognition techniques, but sometime its just better with old school colour detection techniques for multiple object tracking.

So today we will be doing simple colour detection to detect some green objects and mark them in live camera view.

##  Libraries In Use

we will be using only 2 libraries in this tutorial

* OpenCV
* Numpy

## Lets Start Coding

**We will start by importing the libraries first**

```python
import cv2
import numpy as np
```

 

Now to detect color we need to know what is color in pixels of an image. Images are made of tiny dots of pixels each having a color and we can define those colors in terms of HSV -> Hue, Saturation, Value.

The hue of a pixel is an angle from 0 to 359 the value of each angle decides the color of the pixel the order of the color is same but i reverse as the order in rainbow order from red to violet and again back to red

The Saturation is basically how saturated the color is, and the Value is how bright or dark the color is

So the range of these are as follows

* Hue is mapped – >0º-359º as \[0-179]
* Saturation is mapped ->  0%-100% as \[0-255]
* Value is 0-255 (there is no mapping)

So what does that mean.. It means for hue if we select for example 20 it will take it as 40º in terms of degree,\
And for saturation 255 means 100% saturate and 0 means 0% saturate

Enough Talking lets jump back to code. We need to tell out program that we only want green color object to be detected rest of the colors we are not interested in. to do that we need to decide a rage for HSV value for Green (as there are lots of variation of green color)

```python
lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])
```

So we declared these limits for the hsv values of each pixels. Now we will create a new binary image of same size a original image, we will call it mask and we’ll make sure only those pixels that are in this hsv range will be allowed to be in the mask. that way only green objects will be in the mask

Okay before doing that lets initialize our camera object

```python
cam= cv2.VideoCapture(0)
```

and create a font for the text we will be printing in the screen

```python
ont=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)
```

Now lets Start The Main Processing

first we will read a frame from the camera

```python
ret, img=cam.read()
```

we will resize it to make it a small fixed size for faster processing

```python
img=cv2.resize(img,(340,220))
```

Now we will convert this image to hsv format

```python
imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
```

after this we will be creating the filter which will create the mask for green color

```python
mask=cv2.inRange(imgHSV,lowerBound,upperBound)
```

now lets see how the mask looks

```python
cv2.imshow("mask",mask)
cv2.imshow("cam",img)
cv2.waitKey(10)
```

[![OpenCV multiple object tracking Color detection ](https://web.archive.org/web/20201125055058im_/https://i1.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-13-at-7.59.45-PM-230x300.png?resize=314%2C410)](https://web.archive.org/web/20201125055058/https://thecodacus.com/opencv-object-tracking-colour-detection-python/screen-shot-2017-08-13-at-7-59-45-pm/)

Raw mask output

 

## Filtering the Mask

As we can see the output is quite great but we have some false positives in the mask. Those are the noises which are not good for object tracking. We have to clean to make out tracker work otherwise we will be seeing object marked in random places.

to do that we need to do some morphological operation called opening and closing

opening will remove all the dots randomly popping here and there and closing will close the small holes that are present in the actual object

so before doing that we need a 2d matrix called kernal which is basically to control the effects of opening and closing

```python
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)
```

now lets see the results again

```python
cv2.imshow("maskClose",maskClose)
cv2.imshow("maskOpen",maskOpen)
cv2.waitKey(10)
```

![OpenCV multiple object tracking Color detection ](https://web.archive.org/web/20201125055058im_/https://i1.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-13-at-8.11.35-PM-300x213.png?resize=414%2C294)![OpenCV multiple object tracking Color detection ](https://web.archive.org/web/20201125055058im_/https://i2.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-13-at-8.11.58-PM-300x213.png?resize=410%2C290)![OpenCV multiple object tracking Color detection ](https://web.archive.org/web/20201125055058im_/https://i2.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-13-at-8.11.41-PM-300x213.png?resize=406%2C288)![OpenCV multiple object tracking Color detection ](https://web.archive.org/web/20201125055058im_/https://i1.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-13-at-8.11.50-PM-300x211.png?resize=407%2C286)

Tada! the result in the maskClose is the final form after cleaning all the noise now we know exactly where the object is so we can draw a contours from this mask

```python
maskFinal=maskClose
conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
cv2.drawContours(img,conts,-1,(255,0,0),3)
```

and now the variable ***conts***  is a list of contours (in this case only one contour is present but if multiple objects are there it will contain all the contours). we will loop through all the contours and put a rectangle over it and we will mark them with a number for object tracking

```python
for i in range(len(conts)):
    x,y,w,h=cv2.boundingRect(conts[i])
    cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
    cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
```

### And this is the output

[![OpenCV multiple object tracking Color detection ](https://web.archive.org/web/20201125055058im_/https://i0.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-13-at-8.31.24-PM-300x215.png?resize=397%2C285)](https://web.archive.org/web/20201125055058/https://thecodacus.com/opencv-object-tracking-colour-detection-python/screen-shot-2017-08-13-at-8-31-24-pm/)

## Thats it. we are almost done  !!

Now we will put all the processing part inside the loop and all the constant part outside the loop

the loop will look like

```python
while True:
    ret, img=cam.read()
    img=cv2.resize(img,(340,220))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

    cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)
```

 

## The Complete Code For Multiple Object Tracking

```python
import cv2
import numpy as np

lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])

cam= cv2.VideoCapture(0)
kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))

font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX,2,0.5,0,3,1)

while True:
    ret, img=cam.read()
    img=cv2.resize(img,(340,220))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    cv2.drawContours(img,conts,-1,(255,0,0),3)
    for i in range(len(conts)):
        x,y,w,h=cv2.boundingRect(conts[i])
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255), 2)
        cv2.cv.PutText(cv2.cv.fromarray(img), str(i+1),(x,y+h),font,(0,255,255))
    cv2.imshow("maskClose",maskClose)
    cv2.imshow("maskOpen",maskOpen)
    cv2.imshow("mask",mask)
    cv2.imshow("cam",img)
    cv2.waitKey(10)
```

 

## The Complete Video Tutorial For Multiple Object Tracking

<iframe width="600" height="400"  style="width:100%;" src="https://www.youtube.com/embed/efWITgemKvs" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>