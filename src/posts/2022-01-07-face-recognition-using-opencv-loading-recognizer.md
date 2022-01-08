---
layout: blog
title: Face Recognition Using OpenCV | Loading Recognizer
date: 2017-02-21T16:27:04.754Z
category: computer vision
featuredImage: /images/uploads/dormakaba-blog-post-pictures-_-1024-x-683-83.webp
redirect_from:
    - /2017/02/21/face-recognition-loading-recognizer/
---

In my previous post we learnt to train a recognizer using a dataset, in this post we are loading recognizer to see how we can use that recognizer to recognize faces.

If you are following my previous posts then you already have the trained recognizer with you inside a folder named “trainner” and “trainner.yml” file inside it. Now we are going to use that training data to recognize some faces we previously trained .

### Lets Start By Importing The Libraries

```python
import cv2
import numpy as np
```

Yes that’s it, thats all we need for this projects.

### Now Loading Recognizer

next we create a recognizer object using opencv library and load the training data (before that just sve your script in the same location where your “trainner” folder is located)

```python
recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
```

Now we will create a cascade classifier using haar cascade for face detection, assuming u have the cascade file in the same location,

```python
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);
```

Now we will create the video capture object

```python
cam = cv2.VideoCapture(0)
```

Python

Next we need a “font” that’s because we are going to write the name of that person in the image so we need a font for the text

```python
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
```

Okay so the first parameter is the font name, 2nd and 3rd is the horizontal and the vertical scale,4rth is shear (like italic), 5th is thickness of line, 6th is line type

So we have all setup

### Lets Start the main Loop

Lets start the main loop and do the following basic steps

-   Starts capturing frames from the camera object
-   Convert it to Gray Scale
-   Detect and extract faces from the images
-   Use the recognizer to recognize the Id of the user
-   Put predicted Id/Name and Rectangle on detected face

```python
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray,1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x-50,y-50),(x+w+50,y+h+50),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
    cv2.imshow('im',im)
    if cv2.waitKey(10) &amp; 0xFF==ord('q'):
        break
```

So its pretty similar to the face detection code the only difference is the following lines

```python
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
```

in the above two line the recognizer is predicting the user Id and confidence of the prediction respectively\
in the next line we are writing the User ID in the screen below the face, which is (x, y+h) coordinate

## Just Little Finishing Touch (For Unknown Faces)

Now with this we are pretty much done we can add some more finishing touch like its showing user Id instead of the name,\
and it cant handle unknown faces,

So to add this additional features we can do the following,

```python
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf&lt;50):
            if(Id==1):
                Id="Anirban"
            elif(Id==2):
                Id="Obama"
        else:
            Id="Unknown"
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
```

## Now Some Cleanup

```python
cam.release()
cv2.destroyAllWindows()
```

Now that everything is done, we need to close the camera and the windows. and we are done!!!!

And this is the results

![](/images/uploads/face-recognition-300x276.jpeg)

## The Complete Face Recognition Code In One Piece

Now as Promised

```python
import cv2
import numpy as np

recognizer = cv2.createLBPHFaceRecognizer()
recognizer.load('trainner/trainner.yml')
cascadePath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath);


cam = cv2.VideoCapture(0)
font = cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_SIMPLEX, 1, 1, 0, 1, 1)
while True:
    ret, im =cam.read()
    gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=faceCascade.detectMultiScale(gray, 1.2,5)
    for(x,y,w,h) in faces:
        cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
        Id, conf = recognizer.predict(gray[y:y+h,x:x+w])
        if(conf&lt;50):
            if(Id==1):
                Id="Anirban"
            elif(Id==2):
                Id="Sam"
        else:
            Id="Unknown"
        cv2.cv.PutText(cv2.cv.fromarray(im),str(Id), (x,y+h),font, 255)
    cv2.imshow('im',im)
    if cv2.waitKey(10) &amp; 0xFF==ord('q'):
        break
cam.release()
cv2.destroyAllWindows()
```

## Complete Video Tutorial

Feel Free to Subscribe my blog and my youtube channel

<iframe width="600" height="400" style="width:100%;" src="https://www.youtube.com/embed/oqMTdjcrAGk" frameborder="0" allow="accelerometer; autoplay; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Updates: Github links:

For this code: you can visit: <https://github.com/thcodacus/Face-Recognition>

nazmi69 has done a good job converting the code for python 3.x and opencv 3.0\
available at <https://github.com/nazmi69/Face-Recognition>
