---
layout: blog
title: Face Recognition – OpenCV Python | Dataset Generator
date: 2017-01-30T15:47:46.247Z
category: machine learning
featuredImage: /images/uploads/shutterstock_504120676.jpeg
---
In my last post we learnt how to setup opencv and python and wrote this code to detect faces in the frame. Now lets take it to the next level, lets create a face recognition program, which not only detect face but also recognize the person and tag that person in the frame

## Lets Do Face Recognition

To make a face recognition program, first we need to train the recognizer with dataset of previously captured faces along with its ID, for example we have two person then first person will have ID 1 and 2nd person will have ID 2,  so that all the images of person one in the dataset will have ID 1 and all the images of the 2nd person in the dataset will have ID 2, then we will use those dataset images to train the recognizer to predict the 1 of an newly presented face from the live video frame

So lets break the program into 3 major part:

1. Dataset Creator
2. Trainer
3. Detector

In this post we are going to see how to create a program to ganerate dataset for our face recognition program

## Dataset Generator[](https://web.archive.org/web/20201028091519/https://thecodacus.com/opencv-face-recognition-in-python-part-1/)

Lets create the dataset generator script, open your python ***IDLE*** and create a new file and save it in your project folder and make sure you also have the ***haarcascade_frontalface_default.xml*** file in the same folderJust like in the previous post we will need to do the following first:

* cv2 library (opencv library)
* create a video capture object
* cascadeClassifier object

So here it is in form of python code

```python
import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
```

Our dataset generator is going to capture few sample faces of one person from the live video frame\
and assign a ID to it and it will save those samples in a folder which we are going to create now and we will name it ***dataSet***

So create a folder named ***dataSet*** in the same location where you have saved your .py scriptWe are going to follow this naming convention for the sample images to make sure they dont mixed up with other person’s image samples\
**\
*User.\[ID].\[SampleNumber].jpg***for example if the user id is 2 and its 10th sample from the sample list then the file name will be\
**\
*User.2.10.jpg***

Why this format?? well we can easily get which user’s face it is from its file name while loading the image for the training the recognizer

OK, now lets get the user id from the shell as input, and initialize a counter variable to store the sample number

```python
Id=raw_input('enter your id: ')
sampleNum=0
```

Now let start the main loop, we will take 20 samples from the video feed and will save it in the ***dataSet*** folder that we created previously

```python
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

We wrote the above code to detect face in my earlier post,

So we are going to modify this code to make the dataset generator for our face recognizer program

```python
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/user."+Id+'.'+str(sampleNum)+".jpg",gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

So we added this two lines there to get the sample number and save the face in jpg format with our naming convention

and for those who dont know what we captured the face, its this ***“gray\[y:y+h,x:x+w]”*** part where x,y is the top left coordinate of the face rectangle and h,w is the height and the weight of the face in terms of pixels

but this code will take samples vary rapidly like 20 samples in a second.. but we dont want that, we want to capture faces from different angles and for that it needs to be slow.

for that we need to increase the delay between the frames\
and we need to break the loop after it took 20 samples so we need to change few more things in the above code

```python
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w]) #

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
```

There we go, now it will wait for 100 between frames which will give you time to move your face to get a different angle and it will close after taking 20 samples

So our main loop is done now we just have to release the camera and close the windows

```python
cam.release()
cv2.destroyAllWindows()
```

### Lets Test It[](https://web.archive.org/web/20201028091519/https://thecodacus.com/opencv-face-recognition-in-python-part-1/)

If we run this code now then we will see that it will capture faces from the live video and will save it in the dataSet folder

![opencv Detected Face image output](/images/uploads/image2.jpeg)



![Dataset generated](/images/uploads/image3.jpeg)

looks good… Now we have our dataset we can now train the recognizer to learn the faces from this dataset\
**\
*In the next post we will create the trainer portion of the code***

## NOW THE COMPLETE CODE IN ONE PIECE[¶](https://web.archive.org/web/20201028091519/https://thecodacus.com/opencv-face-recognition-in-python-part-1/)

```python
import cv2
cam = cv2.VideoCapture(0)
detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Id=raw_input('enter your id')
sampleNum=0
while(True):
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        
        #incrementing sample number 
        sampleNum=sampleNum+1
        #saving the captured face in the dataset folder
        cv2.imwrite("dataSet/User."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])

        cv2.imshow('frame',img)
    #wait for 100 miliseconds 
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break
    # break if the sample number is morethan 20
    elif sampleNum>20:
        break
cam.release()
cv2.destroyAllWindows()
```

## VIDEO TUTORIAL OF THIS POST

<iframe width="600" height="400" style="width:100%;" src="https://www.youtube.com/embed/4W5M-YaJtIA" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

### Updates: Github links:

For this code: you can visit: <https://github.com/thecodacus/Face-Recognition>

nazmi69 has done a good job converting the code for python 3.x and opencv 3.0\
available at <https://github.com/nazmi69/Face-Recognition>