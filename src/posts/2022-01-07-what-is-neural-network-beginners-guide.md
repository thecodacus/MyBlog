---
layout: blog
title: What Is Neural Network | Beginners Guide
date: 2017-06-18T17:20:05.591Z
category: machine learning
featuredImage: /images/uploads/what-is-an-artificial-neural-networks.jpeg
redirect_from:
    - /2017/06/18/neural-network-beginners-guide/
---

Hi everyone, in this post I will be introducing you about neural network, how it works and how you can create your own neural network, Many of you already seen or heard what amazing things people are doing using neural networks, many of you already know the theory also, but struggling in practical implementations.

In one of my [YouTube tutorials](https://web.archive.org/web/20201125060732/https://www.youtube.com/watch?v=O8BXapqL5lA&feature=youtu.be) I showed how you can implement a neural network from scratch without using any modules or external machine learning libraries. So check that if you want a deep level understanding of neural network.

But In this post I am going to discuss on how to instantly build a neural network model and get started with it almost with no efforts.

So let’s get started

# Initial Setup

We are going to use a python library called “**_[Keras](https://web.archive.org/web/20201125060732/https://keras.io/)_**” which is a wrapper class for neural network library “**_theano/tensorfow”._**  In this tutorial we are going to use [theano](https://web.archive.org/web/20201125060732/http://deeplearning.net/software/theano/) as backend library for Keras, Which is available for Win/Linux/Mac so that everyone can follow along.

### Lets set it up

Open your terminal (cmd for windows), and type **_“pip install theano”_**  install the theano python package, and then hit **_“pip install keras”_**  and now we have all the packages we need for this tutorial.

### Some Things To Check

Before proceeding we have to make sure that keras is indeed trying to use theano as back-end not tensorflow as we have installed theano only

to check that we need to check a file which is present at\
**_<user>/.keras/keras.json_**

for windows user “.keras” folder is visible and can be opened using directly using windows explorer

for linux/mac user use terminal to open the file  type **_“nano ~/.keras/keras.json”_**  this will directly open the file in the terminal using nano editor.\
make sure the file look following

```json
{
	"image_dim_ordering": "th",
	"epsilon": 1e-7,
	"floatx": "float32",
	"backend": "theano"
}
```

That’s all for the setup. We can now start coding our very won neural network.

## Design First

First lets talk about our design of the network, In this tutorial we are going to keep things simpler. So we are creating a super simple design, We will make more complex designs in our later tutorials.

It’s always better to have some visual understanding of what we are going to do. So the design looks as in the below image.

![Neural Network design](/images/uploads/architecture.jpeg)

As you can see we are going to have 3 layers 1st one is called the input layer and last one is called the output layer. what ever in between is called hidden layer(s). In this case we are going to create input layer with 2 input neurons and one hidden layer with 5 neurons and 1 output neuron in output layer.

## Lets Code A Neural Network

We will start by importing keras libraries. To build the network he need two modules, 1st is **_model_** which will contain the network structure, We will be using **_“Sequencial”_** which is the most common. and **_“layers”_**  which will be the building block of the network. in this simple tutorial we are going to use Dense layer which is the basic layers that we have discussed previously in the diagram. And we need numpy also.

```python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
```

So it looks like this

### Lets Define The Model

Now that we have all the components, we need to define the model structure as discussed in the design section.

we will be having 1 input layer one hidden layer and one output layer, So it looks like this.

```python
model = Sequential()
```

In the above code we first initialized a Sequential model, called **_“model”_**   Now we just have to attach layers like a layers in a sandwich, We start with adding the first hidden layer and there we also define the number of input neurons and number of neurons in the layer.

```python
model.add(Dense(5, input_dim=2, activation='tanh'))
```

We don’t have to create the input layer, The model will create it automatically. So we Start with first hidden layer. It’s a **_Dense_** layer and we just need to specify the input dimension (number of input neuron) of the input layer.  the hidden layer itself contain 5 neurons, and we need an activation function for each layer (check out my neural network from scratch for more detailed overview), we are using ‘tanh’ function as activation function.

```python
model.add(Dense(1, activation='tanh'))
```

Now we will added the output layer. we don’t have to worry about input dimension in this as it’s not the first layer in the sequence. and this is the output layer and contain only one output, with “tanh” as activation function.

So the Model will look like this

```python
model = Sequential()
model.add(Dense(5, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='tanh'))
```

### Compile The Model

What do I mean by compile the model. Actually in the above steps we just defined the structure of the model. next we will compile the model where the library will actually build the network according to the instruction we gave, as most optimised form as possible

```python
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
```

### Lets Create the Dataset

In this tutorial we will be training the neural network to behave like an XOR gate so our dataset will contain 2 binary input bits and 1 binary output bit.

-   1,1 -> 0
-   1,0 -> 1
-   0,1 -> 1
-   0,0 -> 0

Above is the input and output set. lets prepare them in form of data that our model can understand,

So we have to create our dataset’s dimension in the following format

-   input shape =(number of total samples, number of inputs )
-   output shape =(number of total samples, number of outputs )

which is in our case it will be

-   input shape= (4,2)
-   output shape= (4,1)

### The Code Will Look Like This

```python
# Preparing Inputs
input=[]
input.append([0,0])
input.append([0,1])
input.append([1,0])
input.append([1,1])

#converting Input variable from list to numpy array
input=np.asarray(input)

#Preparing corresponding Outputs
output=[]
output.append([0])
output.append([1])
output.append([1])
output.append([0])

#Converting list to numpy array

output=np.asarray(output)
```

### Training The Model

Now that we have prepared the training set, we can start training the Model

```python
model.fit(input,output,shuffle=True, nb_epoch=200, batch_size=4)
```

In the above code **_“nb_epoch”_**  indicates how many times we want to run the total set again, “**_batch_size”_** is the number of samples to train in one shot. rest of the parameters are self-explanatory. Now we can test out results.

### Lets Test The Model

We have trained the model, So its time to test it,

```python
#Testing with input 1,1 / Expected Output  is 0
input=[[1,1]]
input=np.asarray(input)
output=model.predict(input)
print output[0]
```

# Complete Neural Network Code

Below is the complete code for this tutorial, Have fun and please comment if you have any doubts, I will be happy to answer.

```python
from keras.models import Sequential
from keras.layers import Dense
import numpy as np

model = Sequential()
model.add(Dense(5, input_dim=2, activation='tanh'))
model.add(Dense(1, activation='tanh'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Preparing Inputs
inputSet=[]
inputSet.append([0,0])
inputSet.append([0,1])
inputSet.append([1,0])
inputSet.append([1,1])

#converting Input variable from list to numpy array
inputSet=np.asarray(inputSet)

#Preparing corresponding Outputs
output=[]
output.append([0])
output.append([1])
output.append([1])
output.append([0])

#Converting list to numpy array
output=np.asarray(output)

#training the model
model.fit(inputSet,output,shuffle=True, nb_epoch=200, batch_size=4)

#Testing with input 1,1 / Expected Output is 0
inputSet=[[1,1]]
inputSet=np.asarray(inputSet)
output=model.predict(inputSet)
print "Output of 1,1"
print (output[0]);
```

As you can see the output is -0.00012 which is very close to 0. So it’s fairly acceptable.

---

#### Thanks for reading this article feel free to comment and share your thoughts/doubts
