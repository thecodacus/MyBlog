---
layout: blog
title: Gesture Recognition Virtual Mouse Using OpenCV
date: 2017-08-17T13:47:51.545Z
category: computer vision
featuredImage: /images/uploads/guesture-recognition-technology-blog-feature-image.jpeg
redirect_from:
    - /2017/08/16/gesture-recognition-virtual-mouse-using-opencv-python/
---

In My Last OpenCV tutorial, I wrote a program to detect green objects and track them. In this post, I am going to show you how we can extend that idea to do some more things like gesture recognition. We will apply that to create a virtual mouse.

I will be using that code as a base of this program and will work on top of it. So if you haven’t read the previous tutorial you can check it [here ](https://web.archive.org/web/20210128044644/https://thecodacus.com/opencv-object-tracking-colour-detection-python/)

## Libraries In Use

The external libraries that we will be using:

-   OpenCV
-   NumPy
-   wx
-   pynput

## Let's Take a Look Back At The Colour Detection Code

This was the color detection code. If you don’t know how it's working [check that post first](https://web.archive.org/web/20210128044644/https://thecodacus.com/opencv-object-tracking-colour-detection-python/)

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

## Let's Create a Virtual Mouse with Gesture Recognition

Okay let's start modifying the above code for gesture recognition

#### Import libraries

```python
import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx
```

These are the libraries that we will be using. **_pynput_** to control mouse movements and clicking and **_wx_** to get the display resolution of the monitor

#### Global variables Setup

now that we already have all the libraries let's set up all the variables and objects

```python
mouse=Controller()
app=wx.App(False)
(sx,sy)=wx.GetDisplaySize()
(camx,camy)=(320,240)
```

we will need these variables and objects, mouse object is for mouse movements and to get the screen resolution we need a **_wx_** app then we can use them **_`wx.GetDisplaySize()`_** to get the screen resolution.

lastly, we are setting some variables **_camx_**, **_camy_** to set the captured image resolution. we will be using it later in the image resize function

## Let's Start The Main Loop

```python
while True:
    ret, img=cam.read()
    img=cv2.resize(img,(camx,camy))

    #convert BGR to HSV
    imgHSV= cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    # create the Mask
    mask=cv2.inRange(imgHSV,lowerBound,upperBound)
    #morphology
    maskOpen=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernelOpen)
    maskClose=cv2.morphologyEx(maskOpen,cv2.MORPH_CLOSE,kernelClose)

    maskFinal=maskClose
    conts,h=cv2.findContours(maskFinal.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
```

The above code is the portion of the loop we wrote in our color detection program. We don’t need to modify the loop till this point. We will be adding our code after this.

```python
while True:
    :
    :

    if(len(conts)==2):
        # logic for the open gesture, move mouse without click
        ....
        ....
    elif(len(conts)==1):
        # logic for close gesture
        ....
        ....
    cv2.imshow("cam",img)
    cv2.waitKey(5)
```

Above is the structure of our extended code. after getting the contours in conts variable we will check if there are contours of 2 objects present in the frame we will move the mouse but we won't perform any click operation

similarly, if there is only one object contour present we will move the mouse as well as we will perform click operations

#### Implement The Open Gesture Operation

[![virtual mouse gesture recognition ](https://web.archive.org/web/20210128044644im_/https://i1.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-15-at-9.17.50-PM-300x243.png?resize=300%2C243)](https://web.archive.org/web/20210128044644/https://thecodacus.com/gesture-recognition-virtual-mouse-using-opencv-python/screen-shot-2017-08-15-at-9-17-50-pm/)

Open Gesture

To Implement the open gesture we need to do some calculations to find some coordinates.  See the below image to get the idea

We have to first calculate the center of both detected green objects which we can easily do by taking the average of the bounding boxes maximum and minimum points. now we got 2 coordinates from the center of the 2 objects we will find the average of that and we will get the redpoint shown in the image.. okay let's do this

```python
while True:
    :
    :
    if(len(conts)==2):
        # logic for the open gesture, move mouse without click
        x1,y1,w1,h1=cv2.boundingRect(conts[0])
        x2,y2,w2,h2=cv2.boundingRect(conts[1])
        # drawing rectangle over the objects
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        #centre coordinate of first object
        cx1=x1+w1/2
        cy1=y1+h1/2
        # centre coordinate of the 2nd object
        cx2=x2+w2/2
        cy2=y2+h2/2
        # centre coordinate of the line connection both points
        cx=(cx1+cx2)/2
        cy=(cy1+cy2)/2
        # Drawing the line
        cv2.line(img, (cx1,cy1),(cx2,cy2),(255,0,0),2)
        # Drawing the point (red dot)
        cv2.circle(img, (cx,cy),2,(0,0,255),2)
```

So the above code is the result of what I just explained earlier and with this, we have the coordinate to position our mouse cursor

Now we need to position our mouse cursor according to the calculated coordinate okay let's do that

```python
while True:
    :
    :
    if(len(conts)==2):
        :
        :
        mouse.release(Button.left)
        mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
        mouse.position=mouseLoc
        while mouse.position!=mouseLoc:
            pass
```

In the above code first, we are doing a mouse release to ensure the mouse left button is not pressed. Then we are converting the detected coordinate from camera resolution to the actual screen resolution. After that, we set the location as the **mouse.position**. but to move the mouse it will take time for the curser so we have to wait till the curser reaches that point. So we started a loop and we are not doing anything there we are just waiting will the current mouse location is the same as the assigned mouse location, that is for the open gesture

#### Implement Close Gesture/ Clicking

Now let's implement the close gesture where we will be clicking the object and dragging it

```python
while True:
    :
    :
    if(len(conts)==2):
        :
        :
    elif(len(conts)==1):
        x,y,w,h=cv2.boundingRect(conts[0])
        #drawing the rectangle
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cx=x+w/2
        cy=y+h/2
        cv2.circle(img,(cx,cy),(w+h)/4,(0,0,255),2)

        mouse.press(Button.left)
        mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
        mouse.position=mouseLoc
        while mouse.position!=mouseLoc:
            pass
```

Python

The above code is similar to the open gesture, but the difference is we only have one object here so we only need to calculate the center of it. And that will be where we will position our mouse pointer. Also, we are performing a mouse press operation instead of a mouse release operation. The rest of the part is the same as the earlier one.

This is the result :

![virtual mouse gesture recognition ](https://web.archive.org/web/20210128044644im_/https://i0.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-15-at-11.09.17-PM-300x247.png?resize=370%2C305)

Close Gesture

## Some Fine Tuning

We are almost done. The code is almost perfect except. we won't be able to drag anything. because in close gesture we are continuously performing **_mouse.press_** operation which will result in continuous multiple clicks while dragging.

To solve this problem what we can do is, we will be putting a flag called “**_pinchFlag_**” and we will set that **_1_** once we perform a click operation. and we won't perform mouse press operation anymore until the flag is **_0_** again

so the code will look like this

```python
pinchFlag=0# setting initial value
while True:
    :
    :
    if(len(conts)==2):
        :
        :
        if(pinchFlag==1): #perform only if pinch is on
            pinchFlag=0 # setting pinch flag off
            mouse.release(Button.left)
        mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
        mouse.position=mouseLoc
        while mouse.position!=mouseLoc:
            pass
    elif(len(conts)==1):
        :
        :
        if(pinchFlag==0): #perform only if pinch is off
            pinchFlag=1 # setting pinch flag on
            mouse.press(Button.left)
        mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
        mouse.position=mouseLoc
        while mouse.position!=mouseLoc:
            pass
```

## Final Code For Virtual Mouse with Gesture Recognition

That pretty much it. I recorded a video tutorial for this which will fix some more common problems you can check that out below

but for now, the final code looks like this

```python
import cv2
import numpy as np
from pynput.mouse import Button, Controller
import wx
mouse=Controller()

app=wx.App(False)
(sx,sy)=wx.GetDisplaySize()
(camx,camy)=(320,240)

lowerBound=np.array([33,80,40])
upperBound=np.array([102,255,255])

cam= cv2.VideoCapture(0)

kernelOpen=np.ones((5,5))
kernelClose=np.ones((20,20))
pinchFlag=0

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

    if(len(conts)==2):
        if(pinchFlag==1):
            pinchFlag=0
            mouse.release(Button.left)
        x1,y1,w1,h1=cv2.boundingRect(conts[0])
        x2,y2,w2,h2=cv2.boundingRect(conts[1])
        cv2.rectangle(img,(x1,y1),(x1+w1,y1+h1),(255,0,0),2)
        cv2.rectangle(img,(x2,y2),(x2+w2,y2+h2),(255,0,0),2)
        cx1=x1+w1/2
        cy1=y1+h1/2
        cx2=x2+w2/2
        cy2=y2+h2/2
        cx=(cx1+cx2)/2
        cy=(cy1+cy2)/2
        cv2.line(img, (cx1,cy1),(cx2,cy2),(255,0,0),2)
        cv2.circle(img, (cx,cy),2,(0,0,255),2)
        mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
        mouse.position=mouseLoc
        while mouse.position!=mouseLoc:
            pass
    elif(len(conts)==1):
        x,y,w,h=cv2.boundingRect(conts[0])
        if(pinchFlag==0):
            pinchFlag=1
            mouse.press(Button.left)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        cx=x+w/2
        cy=y+h/2
        cv2.circle(img,(cx,cy),(w+h)/4,(0,0,255),2)
        mouseLoc=(sx-(cx*sx/camx), cy*sy/camy)
        mouse.position=mouseLoc
        while mouse.position!=mouseLoc:
            pass
    cv2.imshow("cam",img)
    cv2.waitKey(5)
```

## Virtual Mouse with Gesture Recognition Video Tutor

[Simple Gesture Recognition To Create Virtual Mouse | using OpenCV and Python (Tutorial) Part 1](https://web.archive.org/web/20210128044644if_/https://www.youtube.com/embed/DTkvaYRX8o0?feature=oembed)
