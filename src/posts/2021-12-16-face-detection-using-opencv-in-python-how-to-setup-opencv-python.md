---
layout: blog
title: Face Detection Using OpenCV In Python | How To Setup OpenCV Python
date: 2017-01-30T14:13:54.773Z
category: computer vision
featuredImage: /images/uploads/7cd53d36d121d839da9600ca055b01db.gif
redirect_from:
    - /2017/01/30/opencv-python-face-detection/
---

Opencv is the most popular computer vision library, and today we are going to learn how to set up opencv, how to access your webcam and, how easily we can write a face detection program with just a few lines of code.

Hi there, this is me Anirban, writing my first blog on face detection, I am a crazy person, and if you are like me you might like the kind of stuff that is in my blog.

In my posts, I will give you a detailed description of the project that we are going to work on as well as a video tutorial of the same at the end of the post.

Before starting, I am assuming you already have a good knowledge of python, and if you are not confident,

[Here](https://web.archive.org/web/20200803144552/https://amzn.to/2kUK9WQ) are some suggested ebooks for python, to build your background in python first, this will help you to understand things later on.

Additionally [here](https://web.archive.org/web/20200803144552/https://amzn.to/2mtfCf1) is a suggested ebook to get started with OpenCV,

So without wasting any more time let's move to the main stuff.\
\
To setup OpenCV in python environment, you will need these things ready ( match the versions to follow along with this tutorial),

-   [Python 2.x](https://web.archive.org/web/20200803144552/https://www.python.org/downloads/)
-   [OpenCV 2.x](https://web.archive.org/web/20200803144552/http://opencv.org/downloads.html)
-   Numpy library (later will download it using pip)

First thing first download python and install it in its default location (i.e c:/python27)\
after you have installed it download the OpenCV and extract it, go to **_“opencv/Build/python/2.7/x86”_** folder and copy **_“cv2.pyd”_** file to **_“c:/python27/Lib/site-packages/”_** folder.

And now we are ready to use OpenCV in python. just one single problem is there, Opencv uses NumPy library for its images so we have to install NumPy library too, let's do that

Go to Start and type **_“cmd”_** you will see the command prompt icon right-click on it and select **_“run as administrator”_** this will bring us to the cmd window.

Now type\
**_“cd c:/python27/scripts/”_**\
hit enter then type\
**_“pip install numpy”_**

This will install the NumPy library in your python packages

## Now We Are Ready To Do Some Coding

Go to Start and search **_“IDLE”_** and open it.

To use OpenCV we need to import the opencv library first,

```python
import cv2
```

After that, we need to import the numpy library

```python
import numpy as np
```

so now we can use opencv and numpy in our code

# Let's Do Face detection

Now that everything is set up and running let's write a code to detect faces from the webcam.

This is a kind of hello world program for opencv\
The method that we are going to use in this program is a cascade classifier, which can be loaded with a pre-trained xml file, these xml files are hard to train but luckily we don't have to worry opencv already has many of pretrained classifiers ready for face detection.

To use the face detection classifier we need to copy the classifier xml file from the **_“\[opencv extracted folder]_/sources/data/haarcascades/”**, and then copy the file **_haarcascade_frontalface_default.xml_** to your project folder (the same location where you will save the python program)

Now that's done we can proceed further and load the classifier now

```python
detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

let add the video capture object now

```python
cap=cv2.VideoCapture(0)
```

In the above line VideoCapture has one argument which is device id, for an inbuilt webcam it's usually ‘0’, and if you have some other webcam you can change that number to see that is your webcam’s Id

let's test the camera now

```python
ret,img=cap.read()
cv2.imshow('windowname',img)
cv2.waitKey(0)
```

looks like its working fine

in the above code we read the image from the video capture object using **_cap.read()_** method, it returns one status variable which is just **_True/False,_** and the captured frame then we used **_imshow()_** method to display the image, here the first argument is the window name and the second argument is the image that we want to display, the third line we used **_waitKey(10)_** is used for a delay of 10 milliseconds, it is important for the **_imshow()_** method to work properly

Before using the face detector we need to convert the captured image to grayscale

```python
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
```

Now let's apply the face detector to detect faces in our captured image

```python
faces = detector.detectMultiScale(gray, 1.3, 5)
```

the above line will get the (x, y) and (height, width) of all the faces present in the captured image in a list. So now we have to loop through all the faces and draw a rectangle there

```python
for (x,y,w,h) in faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
```

I think it's clear what this code is doing, let me explain the rectangle() the first argument is the input image in which we are going to draw the rectangles, second is the x,y coordinate of the face, then the height and weight, after that we are specifying the color of the line which is in the form of (blue, green, red) and you can adjust the value of each color, the range is 0-255, in this case, it's a green line, and the last argument is the line thickness.

Now that we have marked the faces with green rectangles we can display them

```python
cv2.imshow('frame',img)
cv2.waitKey(0)
```

## Almost Done!!

now to detect faces from the webcam live, we need to create a loop that will get the images from the webcam frame by frame and detect the faces and show them in a window. so if we arrange the above code in a loop it will look like this

```python
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

You can see that I changed the **_“waitKey”_** because it also returns the value of the key pressed in the keyboard so we are comparing it with the key ‘q’ if it's true then we are breaking the loop

after the program ends we need to release the video capture object and destroy all the windows

```python
cap.release()
cv2.destroyAllWindows()
```

## The Complete Face Detection Code In One Piece

```python
import numpy as np
import cv2

detector= cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## The Complete Video Tutorial

[OpenCV Face Detection | How to setup OpenCV with python and Write a face detection program](https://web.archive.org/web/20200803144552if_/https://www.youtube.com/embed/1Jz24sVsLE4?feature=oembed)

# Summary

In this post, you discovered how to setup opencv in python, and wrote your very own face detection program specifically, you learned the most basic steps in opencv including:

-   How to use a classifier.
-   How to read images from the camera
-   How to Draw on images
-   How to display images
-   Mainly you learned face detection in real-time

Do you have any questions about opencv or this tutorial?

Ask your question in the comments and I will do my best to answer.
