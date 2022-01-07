---
layout: blog
title: TensorFlow Tutorial Ground Zero | How To Start
date: 2017-08-16T18:52:06.821Z
category: machine learning
featuredImage: /images/uploads/maxresdefault.jpeg
---
Hey Everyone in my last post I showed how we can write a simple neural network program in python from scratch, just to have a better understanding of how they actually work under the hood. If you haven’t checked it yet please [click here](https://web.archive.org/web/20201125055035/https://thecodacus.com/neural-network-scratch-python-no-libraries/) to check that out first. Okay so in this post I prepared a complete basics yet well-informed tensorflow tutorial, where we will see how we can use Tensorflow to write the network we did in our previous post. So let’s get started.

*Info: Join This newly created [Slack Community](https://web.archive.org/web/20201125055035/https://join.slack.com/t/codacus/shared_invite/MjI4MDU5NzEzNzE3LTE1MDI5NzIzNjEtZGZlNDMxNjczYg) for everyone to discuss about ML Computer Vision and Artificial Intelligence* 

## Setting up Tensorflow

Tensorflow website has a complete tutorial guide to install tensorflow. which is available for linux, macOS and Windows

check this link for that ***[www.tensorflow.org/install/](https://web.archive.org/web/20201125055035/https://www.tensorflow.org/install/)***

In all my previous post I tried to write my posts in a way that it can be done in any OS like Linux Mac or Windows. but unfortunately you might face some issue with windows so I would suggest either ***dual-boot to ubuntu***  or ***use Virtual box to create virtual machine and load ubuntu in it*** ,

***Update: I have created A video to install Tensorflow. and this will work on all the platforms*** 

<iframe width="600" height="400" style="width:100%;" src="https://www.youtube.com/embed/gWfVwnOyG78" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>

## Hello Codacus

To test that we have successfully installed tensorflow lets run this code to see if its working perfectly

```python
import tensorflow as tf
hello = tf.constant('Hello, Codacus!')
sess = tf.Session()
print(sess.run(hello))
```

If there is no error, you should get ***“Hello, Codacus”*** in the output

 

## TensorFlow Basic Concepts

There are the main concept in tensorflow that we need to understand.

* placeholders
* variables
* operations
* session

these are the three things we will be dealing with when working with tensorflow. lets discuss about these one by one individually

### Placeholders

Placeholders are the terminals/data point through which we will be feeding data into the network we will build. Its like gate point for our input and output data.

lets create some placeholders

```python
inputs=tf.placeholder('float',[None,2],name='Input')
targets=tf.placeholder('float',name='Target')
```

So we created two placeholders the data type that will feed to the placeholder is ***float*** , 2nd parameter is the shape of input. In tensorflow data flows through the network in form of matrix.

so lets say we have 3 input neurons.

then our input data should be a list of 3 values for example \[1,2,3]

Now lets say we have 4 sets of inputs for example

```python
[1,2,3]
[2,4,3]
[2,8,6]
[5,4,3]
```

and we want to feed all the three inputs in a batch the shape on the input will be ***\[4,3]***

Now lets say we don’t know how many input set we are going to feed at the same time. it can be 1 it can be 100 sets at a time. So we specify that with ***None***.

if we set the input shape of the placeholder as \[None,3] then it will be able to take any sets of data in a single shot. that is what we did in out earlier code.

In the second case I didn’t select any shape, in this case it will accept any shape. but there is a chance of getting error in runtime if the data the network is expecting has a different shape than the data we provided in the placeholder.

The third parameter I used is ***name***  it is not so important for the network to work. but it will be useful to give a name to every node of the network. This will help up to identify the nodes when we will be checking the network diagram in ***Tensor board*** 

### Variables

Variables in tensorflow are different from regular variables, unlike placeholders ***we will never set any specific values in these, nor use it for storing any data***. We will be using it to make connections between layers. and ***tensorflow itself will use these variables to tune the network*** during training. So these are variable for tensorflow to play with. These are often called trainable parameters.

Lets create some Variables

```python
weight1=tf.Variable(tf.random_normal(shape=[2,3],stddev=0.02),name="Weight1")
biases1=tf.Variable(tf.random_normal(shape=[3],stddev=0.02),name="Biases1")
```

So we have two variables here. the first parameter in here is the values we want it initialize it with. in this case we initialized the first one as 2×3 matrix with random values with variation of +/- 0.02. for the shape the ***first one is the number of input connection*** that the layer will receive. and the ***2nd one is the number of output connection*** that the layer will produce for the next layer.

here is an example



![Neural Network](/images/uploads/architecture.jpeg)

For the bias I only selected 3 as shape because for the first bias neurons there is no input and it has 3 output connections.

 

### Operations

Operations are the part where data get transferred from one layer to the next layer. using the weight variables we create which is basically the connection weights between neurons of two consecutive layers

In our previous post we calculated the output of each neuron by summing the weighed output of all the previous layer neuron. we were doing that for each neuron in the layer using a for loop. But there is a better way of doing that. for example we have 2 input neuron and 3 hidden neuron. So there will be total 2×3 number of connections in between. we are presenting them in a 2d matrix

[![](https://web.archive.org/web/20201125055035im_/https://i0.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-17-at-12.21.27-AM.png?resize=653%2C173)](https://web.archive.org/web/20201125055035/https://thecodacus.com/tensorflow-tutorial-ground-zero-start/screen-shot-2017-08-17-at-12-21-27-am/)\
If we do matrix multiplication we will get  the below results

[![](https://web.archive.org/web/20201125055035im_/https://i1.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-17-at-12.31.50-AM-300x136.png?resize=300%2C136)](https://web.archive.org/web/20201125055035/https://thecodacus.com/tensorflow-tutorial-ground-zero-start/screen-shot-2017-08-17-at-12-31-50-am/)so these are the same calculation we did in our previous post for each output using a loop to calculate the sum. Now to add the bias we can directly sum the bias with the results

[![](https://web.archive.org/web/20201125055035im_/https://i2.wp.com/142.93.251.188/wp-content/uploads/2017/08/Screen-Shot-2017-08-17-at-12.35.57-AM-300x126.png?resize=300%2C126)](https://web.archive.org/web/20201125055035/https://thecodacus.com/tensorflow-tutorial-ground-zero-start/screen-shot-2017-08-17-at-12-35-57-am/)

this is the final form of the output with bias.. okay let’s do this in tensorflow. To do this we need to do operations that is matrix multiplications

```python
hLayer=tf.matmul(inputs,weight1)
hLayer=hLayer+biases1
```

So here is two matrix operations first one is matrix multiplications of inputs and the weights1 matrix which will produce the output without bias. and in the next line we did matrix addition which basically performed an element by element additions.

Now this is what we call operations in tensorflow. It’s actually flowing the data from one layer.

one last thing is still pending that is applying the activation functions in the layer outputs. so lets apply that

```python
hLayer=tf.sigmoid(hLayer, name='hActivation')
```

Okay so we completed till hidden layer lets create the output layer also and finish the network

```python
weight2=tf.Variable(tf.random_normal(shape=[3,1],stddev=0.02),name="Weight2")
biases2=tf.Variable(tf.random_normal(shape=[1],stddev=0.02),name="Biases2")
```

As we can see the output layer has only 1 neuron and the previous hidden layer had 3 neurons. So the weight matrix has 3 input and 1 output thus the shape is \[3,1]. and the bias has only one neuron to connect to thus bias shape in \[1]

lets create the output layer

```python
output=tf.matmul(hLayer,weight2)
output=output+biases2
output=tf.sigmoid(output, name='outActivation')
```

with this our network is complete.

##### Optimization

We still have’t done anything to calculate the error or any optimization methods to train the network

```python
#cost=tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=targets)
# update  tf.nn.softmax_cross_entropy_with_logits is for classification problems only
# we will be using tf.squared_difference()
cost=tf.squared_difference(targets, output)
cost=tf.reduce_mean(cost)
optimizer=tf.train.AdamOptimizer().minimize(cost)
```

The method ***softmax_cross_entropy_with_logits*** will take the output of the neuron and target placeholder where we will put out expected outputs. then it will calculate the error for each and every neuron in the layer.

***Update:*** I changes the cost function to ***tf.squared_difference()***  which is similar to the same squared difference, and best suitable for this problem which is not a classification problem,  ***tf.squared_difference***  takes two input fitsr is the target value and second is the predicted value.

next we took that error in a variable ***cost*** and put operated ***reduce_mean***  function on it and stored it in ***cost*** variable again. the ***reduce_mean***  is taking that array of errors and calculating the average of them and returning a single value for the error.

finally we are creating the optimizer to called ***AdamOptimizer*** to minimize the error ***cost .*** It’s basically do the same back propagation using the cost as starting error and propagate that error backwards while fixing the weights a little bit every time. Almost similar that we did in our own neural network in the last post.

### Session

Till now what we did is, we described tensorflow what is the structure of the network and how data will flow. but the network is still not running and no operation has been done yet. So to run the network we need to start a ***tensorflow session.***  any actual calculation will happen inside a session what ever tensorflow operation we perform will actually happen after we start the session and run it.

So before starting the session lets create the data set first. I am going to make it work as another binary gate XOR gate

of course we can do much complicated stuff but for the sake of simplicity we are going to make it work like an XOR gate. So lets generate the input outputs for this

```python
#generating inputs
import numpy as np

inp=[[0,0],[0,1],[1,0],[1,1]]
out=[[0],[1],[1],[0]]

inp=np.array(inp)
out=np.array(out)
```

So before that we need our inputs. and we need it in numpy array format so we converted it to numpy array.

Now lets start the session

```python
epochs=4000 # number of time we want to repeat

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        error,_ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)
```

So here we started a session and named it ***sess .*** After we setup the session the first thing we need to do is we need to initialize all the tensorflow variables that we created using  ***tf.global_variables_initializer().run() .*** 

Now we started a for loop to repeat our training. now inside the loop we are using ***sess.run()*** . The first parameter in this is a list of operations that we want to do. In this case I want to know the error for each training and I also want to Optimize the network. So I put cost and the optimizer variable in the list.

Te second parameter is the ***feed_dict*** . In this argument we will tell tensorflow from where it will take the data for all its ***placeholders***. So we have 2 ***placeholders*** in our network ***inputs*** and ***targets*** we have to tell tensorflow from where it should take its data. We put that information in the ***feed_dict*** parameter.

for ***inputs*** we set ***inp*** . and for ***targets*** we set ***out .*** Therefore we set ***feed_dict ={inputs:inp , targets:out}***

and now the output of ***sess.run()*** is the output of the individual operation that we put inside the list. that is the ***cost*** and the ***optimizer***. the ***cost*** returns the average ***error*** and the ***optimizer*** doesn’t return any value so we set it as ***_*** to ignore it .

and finally we are printing those values.

## Lets Test the model

We have trained the model now lets test its performance

```python
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        error,_ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)
    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        inp=[[a,b]]
        inp=np.array(inp)
        prediction=sess.run([output],feed_dict={inputs: inp})
        print(prediction)
```

 

So here we go. after training we can test the network by running the output node and supplying data to the inputs placeholder  and finally printing it

## Saving The Session for later

One thing to remember is all the variable that were tuned in the session is only remain in that state till the session is active. one we close the session the state of the network is lost. To avoid this we need the save the session. and reload it every time we want to use it again.

to save the session we can use tf.train.Saver() to create a saver object

```python
saver = tf.train.Saver()
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(epochs):
        error,_ =sess.run([cost,optimizer],feed_dict={inputs: inp,targets:out})
        print(i,error)
    saver.save(sess, "model.ckpt") # saving the session with file name "model.ckpt"
```

At the end of the for loop after the training we are saving the session with a file name “model.ckpt”. so now we can load it for later use.

 

## Restoring/Loading the session

Once we have a saved session we can now restore it easily by calling the restore method of Saver class object

```python
saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, "model.ckpt")
    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        inp=[[a,b]]
        inp=np.array(inp)
        prediction=sess.run([output],feed_dict={inputs: inp})
        print(prediction)
```

So This is how we can restore a previously trained model or later use. Thanks for reading I hope this post we help you to understand the basics of tensorflow.

In the next post we will discuss how to use Tensorboard to visualize the network we design in a graphical representation and see the data flowing in real-time