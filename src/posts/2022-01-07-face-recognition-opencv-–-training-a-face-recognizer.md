---
layout: blog
title: Face Recognition OpenCV – Training A Face Recognizer
date: 2017-02-04T16:17:39.495Z
category: computer vision
featuredImage: /images/uploads/mercadona-facial-recognition-mallorca.jpeg
---
To perform face recognition we need to train a face recognizer, using a pre labeled dataset, In my previous post we created a labeled dataset for our face recognition system, now its time to use that dataset to train a face recognizer using opencv python,

# Lets Train A Face Recognizer

First create a python “trainner.py” file in the same folder where we saved out dataset generator script in the previous post, and then create a folder in the same directory name it “trainner”, this is the folder where we are going to save our recognizer after training.

So the folder structure till now should be something like this\
![](https://web.archive.org/web/20201028074328im_/https://i0.wp.com/142.93.251.188/wp-content/uploads/2017/01/Capture.png?resize=227%2C142)\
Those who don’t know how we got the dataset folder and script , it was created in my previous post you should check that first.

Before we start actual coding we need a new library called [pillow](https://web.archive.org/web/20201028074328/https://python-pillow.org/),\
open your **cmd** (***run as administrator*** )and type following command to navigate to your python pip directory:\
***“cd c:/python27/scripts/”***now type the following to install pillow:\
“**pip install pillow**”\
this will install the latest version of pillow in your python library

## Lets Start Coding

So now we rare ready to code the trainner,\
Just like all the other opencv script we need:

import the opencv / ***cv2*** library,\
we will need the ***os*** to accress the file list in out dataset folder,\
we also need to import the ***numpy*** library,\
and we need to import the pillow / ***PIL*** library we installed before,

It will look something like this:

```python
import cv2,os
import numpy as np
from PIL import Image
```

Now we need to initialize the recognizer and the face detector

```python
recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");
```

### Load The Training Data

Ok, now we will are going to create a function which will grab the training images from the dataset folder, and will also get the corresponding Ids from its file name, (remember we formatted the filename to be like ***User.id.samplenumber*** in our previous script)

So I am going to name this function ***“getImagesAndLabels”***  we need the path of the dataset folder so we will provide the folder path as argument. So the function will be like this

```python
def getImagesAndLabels(path):
```

So now inside this function we are going to do the following

* Load the training images from dataset folder
* capture the faces and Id from the training images
* Put them In a List of Ids and FaceSamples  and return it

To load the image we need to create the paths of the image

```python
  imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
```

this will get the path of each images in the folder.\
now we need to create two lists for faces and Ids to store the faces and Ids

```python
  faceSamples=[]
  Ids=[]
```

Now we will loop the images using the image path and will load those images and Ids, we will add that in your lists

```python
    for imagePath in imagePaths:
        pilImage=Image.open(imagePath).convert('L')
        imageNp=np.array(pilImage,'uint8')
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        faces=detector.detectMultiScale(imageNp)
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
```

In the above code we used used “Image.open(imagePath).convert(‘L’)” is loading the image and converting it to gray scale, but now its a PIL image we need to convert it to numpy array.\
for that we are converting it to numpy array “imageNP=np.array(pilImage,’uint8′)”.\
To get the Id we split the image path and took the first from the last part (which is “-1” in python) and that is the name of the imagefile. now here is the trick, we saved the file name in our previous program like this “***User.Id.SampleNumber***” so if we split this using “**.**” the we will get 3 token in a list “User”, “Id”, “SampleNumber”\
so to get the Id we will choone 1st index (index starts from 0)

So we get:

```python
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
```

Now we are using the detector to extract the faces and append them in the faceSamples list with the Id

which looks like:

```python
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
```

So we are done now we just have to return that value

```python
return faceSamples,Ids
```

#### Now the entire function will look like this

```python
def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids
```

### Almost Done!!

We are almost finished, now we just have to call that function and feed the data to the recognizer to train

```python
faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
```

#### Thats it!!

Now if we run this code it will create a “***trainner.yml***” file inside the trainner folder,\
We will use this file in our next post to actually recognize the faces that we trained the face recognizer to recognize,

### The Complete Code

```python
import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')

```

 

### Now The Complete Video Tutorial

<iframe width="600" height="400" style="width:100%;" src="https://www.youtube.com/embed/T-yWORkWvNs" frameborder="0" allow="autoplay; encrypted-media" allowfullscreen></iframe>

 

### Updates:

To make sure the trainner doesn’t take any file other that the jpg files we will add an if condition in the method

```python
import cv2,os
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer()
detector= cv2.CascadeClassifier("haarcascade_frontalface_default.xml");

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #create empth face list
    faceSamples=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:

        # Updates in Code
        # ignore if the file does not have jpg extension :
        if(os.path.split(imagePath)[-1].split(".")[-1]!='jpg'):
            continue

        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces=detector.detectMultiScale(imageNp)
        #If a face is there then append that in the list as well as Id of it
        for (x,y,w,h) in faces:
            faceSamples.append(imageNp[y:y+h,x:x+w])
            Ids.append(Id)
    return faceSamples,Ids


faces,Ids = getImagesAndLabels('dataSet')
recognizer.train(faces, np.array(Ids))
recognizer.save('trainner/trainner.yml')
```

### Github links:

For this code: you can visit: <https://github.com/thecodacus/Face-Recognition>

nazmi69 has done a good job converting the code for python 3.x and opencv 3.0\
available at <https://github.com/nazmi69/Face-Recognition>