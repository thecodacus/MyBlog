---
layout: blog
title: Prepare Real life Data Set To Train Your Tensorflow Model
date: 2017-08-30T19:05:50.730Z
category: machine learning
featuredImage: /images/uploads/8d2d943a-dogcat.png
---
In the last few tutorial, we learned [what is a neural network](https://web.archive.org/web/20201028091951/https://thecodacus.com/neural-network-scratch-python-no-libraries/), and[ how to write your own network in python from scratch](https://web.archive.org/web/20201028091951/https://thecodacus.com/neural-network-scratch-python-no-libraries/). We learned[ how to use Tensorflow](https://web.archive.org/web/20201028091951/https://thecodacus.com/tensorflow-tutorial-ground-zero-start/) to quickly create a neural network and train it easily. Then we learned [how to use Tensorboard](https://web.archive.org/web/20201028091951/https://thecodacus.com/tensorboard-tutorial-visualize-networks-graphically/) to visualize the network for debugging and see real-time results. Now we are equipped with the basic knowledge, we can start building models for learning from real life data. But before that we need data. And we have to get the data in a format which we can feed into the network. In this tutorial, we will learn how to prepare data set for our network from real data.

## Let’s Get SomeData First

In this tutorial, we will be using a data set provided by Kaggle dog vs cat competition. which is publicly available in Kaggle’s website  We will prepare that data set so that we can feed it to the network.

You can get the dataset here -> https://www.kaggle.com/c/dogs-vs-cats/data

the [train.zip](https://web.archive.org/web/20201028091951/https://www.kaggle.com/c/dogs-vs-cats/data) is a compressed folder containing 25000 images of dogs and cats. The file name is in this format ***\[dog/cat].\[sample number].jpg*** 

So by reading the filename, we can get if it’s a dog or cat. Let’s download this zip and extract it into our working directory where we are saving out python scripts.

Now we can read all the images and store them in a python list. and need it to the network one by one. But, the size of the zip it self is 500mb and after extracting it if to load the entire dataset into your program, it will occupy too much ram probably in GB, and we don’t even need all the images at the same time. So we will create a Class which will get few of those images in batch, let’s say 20 images and then after we train our network on

So we will create a Class which will collect few of those images and generate a mini batch, let’s say 20 images and then after we will train our network on it.  Then the generator we collect next set of images from the folder and create another mini batch. this will continue until we are finished with all the images in the folder

## Okay, Let’s Create a Dataset Generator

Okay assuming we have the “train.zip” file extracted in your current working directory and named “train”, Let’s create the Dataset Generator

Let’s add the libraries first

```python
import cv2 # to load the images
import numpy as np # to do matrix mnupulations 
from os.path import isfile, join # to manupulate file paths
from os import listdir # get list of all the files in a directory
from random import shuffle # shuffle the data (file paths)
```

 

So we are using opencv to load the image, we can also use PIL to load the image but I think opencv would be much easier to use. and we also want to do some image manipulations so we will use opencv for that.

#### Saparateing The Data

The data we get from the Kaggle dataset it’s a mixed data. means all the images of dogs and cats are in the same folder. Now let’s separate them into two separate folders.

We are going to write a function which will do that for us.

```python
def seperateData(data_dir):
    for filename in listdir(data_dir):
        if isfile(join(data_dir, filename)):
            tokens = filename.split('.')
            if tokens[-1] == 'jpg':
                image_path = join(data_dir, filename)
                if not os.path.exists(join(data_dir, tokens[0])):
                    os.makedirs(join(data_dir, tokens[0]))
                copyfile(image_path, join(join(data_dir, tokens[0]), filename))
                os.remove(image_path)
```

 

This Function will create 2 folders named cat and dog in the data directory and move the files in their corresponding folder.

As its one time job and we may not have to use it in future, again and again, we are making this function, not as a part of our dataset generator class.

next, let’s create the ***DataSetGenerator*** class.

 

```python
class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_labels = self.get_data_labels()
        self.data_info = self.get_data_paths()

    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels

    def get_data_paths(self):
        data_paths = []
        for label in self.data_labels:
            img_lists=[]
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'jpg':
                    image_path=join(path, filename)
                    img_lists.append(image_path)
            shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths
```

 

In the above code, we created a class named ***DataSetGenerator,***  we in the initializer we are taking the dataset directory path as an argument to list all the folders present in the dataset directory, then creating a list of the file paths in those individual directories using ***get_data_labels*** and ***get_data_paths*** method written below.

Now let’s see what inside ***get_data_labels*** and **get_data_paths** methods. Inside the ***get_data_labels*** method, we used “***listdir”*** function to get a list of names for all the items in the data directory. So, in this case, our list will have two item “cat”, “dog”. These are the classes of our dataset.

Now inside the ***get_data_paths*** method, we again used the function ***“listdir”***  to get all the files and folders available in dataset directory. and looping then in a for loop. then we are checking each path if it’s pointing to a file or a folder. If it’s a file then we are splitting the path string with a ***dot ->***“.”  and checking the last token. So If it’s a jpg file the last token will be “jpg”. this is just to take the image files and to ignore any other files that are present in that directory like temp files.

Once we confirmed its a JPG file we added the entire path along with the data directory plus the class directory as the image path

So we took the image file path and appended it in a list ***img_lists***. And we added that list to the main data_path list

Now the data_path list should contain two lists one with all the lists of image paths for dog and one with list of all the image paths for cat

So we got all the image file paths and the corresponding labels, Now what? How to get the images?, We will be using the Python’s concept of ***generator***, and the concept of ***yield*** to create the mini-batches on the fly and delete them after we are done training our network on it. So it will help us the avoid loading the entire dataset into memory and running out of ram.

## Let’s Create A Mini-Batch Generator

Let’s see how we can make a generator to generate the mini batches for us

```python
class DataSetGenerator:
    :
    :
    def get_mini_batches(self, batch_size=10, image_size=(200, 200), allchannel=True):
        images = []
        labels = []
        empty=False
        counter=0
        each_batch_size=int(batch_size/len(self.data_info))
        while True:
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels),dtype=int)
                label[i] = 1
                if len(self.data_info[i]) &lt; counter+1:
                    empty=True
                    continue
                empty=False
                img = cv2.imread(self.data_info[i][counter])
                img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)
                labels.append(label)
            counter+=1

            if empty:
                break
            # if the iterator is multiple of batch size return the mini batch
            if (counter)%each_batch_size == 0:
                yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
                del images
                del labels
                images=[]
                labels=[]
```

 

In this ***get_mini_batches*** method, we are creating the generator. We are taking ***batch_size*** , output ***image_size*** and a flag ***allchannel*** to check whether we want to use all the 3 channel RGB or just gray scale image

We first divided the ***batch_size*** in the batch_size for each class, so that we take an equal amount of samples from each class. and a ***counter*** for the current iteration for each class.

Then we created two empty lists images and labels, and started an infinite while loop, we will break this loop when none of the classes have any images left to train. Inside the while loop we initiate an inner for loop to get images from each and every class one by one

So before loading any images, we have to check if that class has any image_path left in the list, so we checked if the length of that list of that particular class is less than our ***counter*** value. if it is then we considered it as empty and continuing to the next class. if it’s not we are setting the empty flag as False and loading the image using cv2.imread() method.

we want all our sample images to be the same size, So we need to resize them to a square image without changing their aspect ratio. we do that we are going to use another function named ***resizeAndPad,*** that we will implement later using OpenCV.

For the allchannel flag if it’s false then we are converting the image to gray scale using cv2.cvtColor() method,

Finally after the for loop we got a one set of images appended to images list and labels list we increased the counter by one for next set of image.

Next, we are checking if the value of empty is true that means all the lists are done loading so we will break the while loop. and if it’s not then we will go to the next step.

Now, each time the counter is a multiple of the each_batch_size  which is the batch size for an individual class, we will yield the results and after the yield, we will delete them to release the memory and continue to append the rest of the data in the lists for next call.

 

## Creating The Image Resizer

We used the image resizer method named ***resizeAndPad()** in* our the previous method, we will need to define that our class.

here it is

 

```python
class DataSetGenerator:
    :
    :
    def resizeAndPad(self, img, size):
        h, w = img.shape[:2]

        sh, sw = size
        # interpolation method
        if h &gt; sh or w &gt; sw:  # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h

        # padding
        if aspect &gt; 1: # horizontal image
            new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.uint8)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        elif aspect &lt; 1: # vertical image
            new_shape = list(img.shape)
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape,dtype=np.uint8)
            w_offset = int((h-w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        else:
            new_img = img.copy()
        # scale and pad
        scaled_img = cv2.resize(new_img, size, interpolation=interp)
        return scaled_img
```

Python

So here first we checked if we are shrinking the image or enlarging. Cuz the cv2.INTER_AREA method is better for shrinking the image where as  ***cv2.INTER_CUBIC***  is better for enlarging the image.

Next, we are checking if it’s a horizontal image or a vertical image. and padding the image with zeros so that it becomes a square image.

Then we applied the ***cv2.resize***  method to scale the image according to the given size.

## We are done!!

That It we are done now we can use this class to generate mini batches for our tensorflow model.\
If we run the ***separate***  method with argument ***“./train”*** where it’s the directory where dog vs cat training images are stored

it will separate the images of dogs and cat into two separate folders, and we only need that for one time.

## The complete code to Prepare Data Set From Real Life Data

Here is the complete code for this tutorial.

```python
import cv2
import numpy as np
from os.path import isfile, join
from os import listdir
from random import shuffle
from shutil import copyfile
import os
import pickle


def seperateData(data_dir):
    for filename in listdir(data_dir):
        if isfile(join(data_dir, filename)):
            tokens = filename.split('.')
            if tokens[-1] == 'jpg':
                image_path = join(data_dir, filename)
                if not os.path.exists(join(data_dir, tokens[0])):
                    os.makedirs(join(data_dir, tokens[0]))
                copyfile(image_path, join(join(data_dir, tokens[0]), filename))
                os.remove(image_path)



class DataSetGenerator:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.data_labels = self.get_data_labels()
        self.data_info = self.get_data_paths()

    def get_data_labels(self):
        data_labels = []
        for filename in listdir(self.data_dir):
            if not isfile(join(self.data_dir, filename)):
                data_labels.append(filename)
        return data_labels

    def get_data_paths(self):
        data_paths = []
        for label in self.data_labels:
            img_lists=[]
            path = join(self.data_dir, label)
            for filename in listdir(path):
                tokens = filename.split('.')
                if tokens[-1] == 'jpg':
                    image_path=join(path, filename)
                    img_lists.append(image_path)
            shuffle(img_lists)
            data_paths.append(img_lists)
        return data_paths

    # to save the labels its optional incase you want to restore the names from the ids 
    # and you forgot the names or the order it was generated 
    def save_labels(self, path):
        pickle.dump(self.data_labels, open(path,"wb"))

    def get_mini_batches(self, batch_size=10, image_size=(200, 200), allchannel=True):
        images = []
        labels = []
        empty=False
        counter=0
        each_batch_size=int(batch_size/len(self.data_info))
        while True:
            for i in range(len(self.data_labels)):
                label = np.zeros(len(self.data_labels),dtype=int)
                label[i] = 1
                if len(self.data_info[i]) &lt; counter+1:
                    empty=True
                    continue
                empty=False
                img = cv2.imread(self.data_info[i][counter])
                img = self.resizeAndPad(img, image_size)
                if not allchannel:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    img = np.reshape(img, (img.shape[0], img.shape[1], 1))
                images.append(img)
                labels.append(label)
            counter+=1

            if empty:
                break
            # if the iterator is multiple of batch size return the mini batch
            if (counter)%each_batch_size == 0:
                yield np.array(images,dtype=np.uint8), np.array(labels,dtype=np.uint8)
                del images
                del labels
                images=[]
                labels=[]

    def resizeAndPad(self, img, size):
        h, w = img.shape[:2]

        sh, sw = size
        # interpolation method
        if h &gt; sh or w &gt; sw:  # shrinking image
            interp = cv2.INTER_AREA
        else: # stretching image
            interp = cv2.INTER_CUBIC

        # aspect ratio of image
        aspect = w/h

        # padding
        if aspect &gt; 1: # horizontal image
            new_shape = list(img.shape)
            new_shape[0] = w
            new_shape[1] = w
            new_shape = tuple(new_shape)
            new_img=np.zeros(new_shape, dtype=np.uint8)
            h_offset=int((w-h)/2)
            new_img[h_offset:h_offset+h, :, :] = img.copy()

        elif aspect &lt; 1: # vertical image
            new_shape = list(img.shape)
            new_shape[0] = h
            new_shape[1] = h
            new_shape = tuple(new_shape)
            new_img = np.zeros(new_shape,dtype=np.uint8)
            w_offset = int((h-w) / 2)
            new_img[:, w_offset:w_offset + w, :] = img.copy()
        else:
            new_img = img.copy()
        # scale and pad
        scaled_img = cv2.resize(new_img, size, interpolation=interp)
        return scaled_img


if __name__=="__main__":
    seperateData("./train")
```

- - -

##### [***In the next post, we will see how can we use this to train our*** tensorflow](https://web.archive.org/web/20201028091951/https://thecodacus.com/cnn-image-classifier-using-tensorflow/)***[ model. ](https://web.archive.org/web/20201028091951/https://thecodacus.com/cnn-image-classifier-using-tensorflow/)Thank you very much for reading***