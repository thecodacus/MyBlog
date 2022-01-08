---
layout: blog
title: How to use IP Webcam with opencv as a wireless camera
date: 2017-07-31T17:34:38.320Z
category: computer vision
featuredImage: /images/uploads/ip-webcam-for-pc.png
redirect_from:
    - /2017/07/31/ip-webcam-opencv-wireless-camera/
---

Hi guys, If you are interested in creating robots, or embedded systems like me then you must be wondering if there is an way to use your android phone as an wireless camera (IP Webcam ) for your opencv code embedded in a SBC like Raspberry Pi,

You might have already know about IP Webcam apps for smart phones, but for those who don’t know yet, its an application which streams live images from your mobile camera over local wifi network, and you can see it through web browser.

Okay so now that we all are on same page, lets talk about how we can grab that image streaming on local network in to opencv.

## Prerequisites

-   IP webCam ([from here](https://web.archive.org/web/20201125054759/https://play.google.com/store/apps/details?id=com.pas.webcam&hl=en))
-   Android Phone

## Setting up IP Webcam

After installing IP webcam in your phone, open it up and you will see an IP address in the screen with a port number like below.[](https://web.archive.org/web/20201125054759/https://i2.wp.com/142.93.251.188/wp-content/uploads/2017/07/IP-webcam-android-3.png)

![IP Webcam with OpenCV﻿](/images/uploads/unnamed.webp)

Note that down somewhere, now type that in your browser. it will open up the IP webcam dashboard, which will look like below

![IP Webcam with OpenCV﻿](/images/uploads/unnamed-1-.webp)

Now we need to get the url to get image frames from this dashboard. To do that go to javascript option and we will see the live camera feed from the phone into the browser. just Right click and select **_“Copy Image Address”_**

[](https://web.archive.org/web/20201125054759/https://i2.wp.com/142.93.251.188/wp-content/uploads/2017/07/Screen-Shot-2017-07-30-at-10.30.21-PM.png)

Now paste that link somewhere like a notepad we will be needing that in our code

## Lets Code

So now we are done with the setup all we need to do now is write a python code to grab a frame from the url that we noted down in our last step and display that in an infinite loop

### lets import the libraries

```python
import urllib
import cv2
import numpy as np
```

so imported the **_cv2_** for opencv and **_numpy_** for matrix manipulation and **_urllib_** for opening url and reading values from urls

now we need to read the image from the url that we noted down earlier

```python
url='http://192.168.0.103:8080/shot.jpg'
```

now its time to construct the loop

```python
while True:
    imgResp=urllib.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    cv2.imshow('test',img)
    if ord('q')==cv2.waitKey(10):
        exit(0)
```

### So what just happened up there

Inside the while loop, we are opening the url using **_“urllib.urlopen(url)”_** this will give us an response in a variable “imgResp”,

`imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)`\
In this line we are reading the values from the url response and converting it to a byte array and ultimately converting it to a numpy array and stored it in a numpy variable imgNp. This is our image. in encoded formate.

`img=cv2.imdecode(imgNp,-1)`\
In this line we decode the encoded image and store it in another variable called img. and thats out final image. Now after that we can do out image processing in that image or what ever we want and then finally display that image we an **_imshow-waitKey_** combination

```python
cv2.imshow('test',img)
if ord('q')==cv2.waitKey(10):
    exit(0)
```

## Finally all put together

```python
import urllib
import cv2
import numpy as np

url='http://192.168.0.103:8080/shot.jpg'

while True:
    imgResp=urllib.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)

    # all the opencv processing is done here
    cv2.imshow('test',img)
    if ord('q')==cv2.waitKey(10):
        exit(0)
```

## Complete IP Webcam  OpenCV Video Tutorial

<iframe width="600" height="400" style="width:100%;" src="https://www.youtube.com/embed/2xcUzXataIk" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
