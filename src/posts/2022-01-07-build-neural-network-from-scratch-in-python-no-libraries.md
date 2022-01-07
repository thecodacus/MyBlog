---
layout: blog
title: Build Neural Network From Scratch in Python (no libraries)
date: 2017-08-17T18:00:28.237Z
category: machine learning
featuredImage: /images/uploads/file-20201210-18-elk4m.jpeg
---
Hello, my dear readers, In this post I am going to show you how you can write your own neural network ***without the help of any libraries*** yes we are not going to use any libraries and by that I mean any external libraries like tensorflow or theano. I will still be using math library and numpy. but that will be very minimum

What will be the benefit of this? well, it will be more efficient to use a neural network library if you want to do some serious work and you don’t want to put your mind write a NN. but if you want to learn how the neural network is working under the hood then you might what to write your own small code to see its insights

#### Before starting Let us Have a Look What is Neural Network

<iframe width="600" height="400" style="width:100%;" src="https://www.youtube.com/embed/hENZrj0qxqg" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>



## The Building Blocks

Before jumping into the code lets look at the structure of a simple neural network and decide the building blocks

![neural network structure](/images/uploads/architecture.jpeg)

Image borrowed from ritchieng.com

Above is a simple neural network with three input neuron, three hidden neurons and one output neuron.

each and every neuron is connected to all the neurons in its previous layer. and these connections have weights some are strong some are weak. and forming a network

So looking at this out basic building blocks will be

* Connection -> will keep information between the connection of two neuron
* Neuron -> will get signals from connected neurons and produce an output
* Network -> will create a network of the neurons and flow data in the layers

## Let’s Code a Neural Network From Scratch

okay then without wasting any more time lets start the coding.  we will need two libraries, and we will only use them ones

```python
import math
import numpy as np
```

 

### Now let’s create Connection class

```python
class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0
```

 

That’s it!!.. the connection object will only hold information about the connected neuron, weight of the connection and the delta weight for momentum.

 

### Neuron Class

```python
class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)
```

 

so here we created a neuron class and we defined the initializer with a list of dendrons (connections), error, gradient to calculate error distribution of the connected neurons and the output. it is taking a layer as its input which will be the previous layer. for input and bias neurons the previous layer is None.

Once a neuron is created it has to create connections to its previous layer neurons. which is stored in the dendrons list. and now we have some static variables ***eta, Alpha*** which will be the learning rate and momentum factor of the neurons

Now we need some basic functions for our Neuron class

```python
class Neuron:
    :
    :
    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output
```

the method ***addError*** will be used to accumulate error sent from all the neurons in the next layer during backpropagation. ***sigmoid*** is our activation function and ***dSigmoid*** is the derivation of the sigmoid function it will be used to calculate the gradient of the neuron during backpropagation. others are simply getters and setters.

let’s do the feedforward for a single neuron

```python
class Neuron:
    :
    :
    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)
```

 

In the feedforward method, we are simply checking if there is any previously connected neurons are present. if not then it’s an input or bias neuron and does not need to do and feedforward. But if there is any neuron it’s getting the output of the connected neurons one by one and multiplying it with the connection’s weight which in this case ***dendron.weight.*** summing them up and passing it through the activation function ***sigmoid*** and that the output.

Let define the back Propagation method for a single neuron

```python
class Neuron:
    :
    :
    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output)
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight
            dendron.weight = dendron.weight + dendron.dWeight
            dendron.connectedNeuron.addError(dendron.weight * self.gradient)
        self.error = 0
```

 

here in this method, we are using the ***error x d.(output)** *  to calculate the gradient. The gradient will decide the direction we need to alter the weight of a connection and output of the connected neuron will decide how much we need to alter the weight to minimize the error

so the change in weight for a single connection will be

```
δweight= η x gradient x output of connected neuron + α x previous δweight
```

here eta (η) is the learning rate which will decide how fast the network will update its weight and alpha (α) is the momentum rate which will give the weights a momentum so that it will continue to move in a particular direction ignoring small fluctuations. which basically helps the neuron to avoid local minima

that were the backpropagation for the single connection we have to do it for all the connections. So we loop through all the dendrons and calculated all the change in weights and applied that change to the current weight

We also need to send the error back to the connected neuron so that it can also fix its connection and backpropagate that error.

But we are not using setError to set the error for the connected neuron because there might be other neurons in the same layer which is also connected to the same neuron so if we set the error then what ever error was set by any other neuron will be overwritten. so we need to add the error to the existing error the connected neuron already has. therefore we are using addError Method to accumulate error from all the neurons which are also connected to it.

at last, after we finished adjusting the dendron weights and back propagated the error we need set the error to zero.

#### That’s the complete neuron class

```python
class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output);
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight;
            dendron.weight = dendron.weight + dendron.dWeight;
            dendron.connectedNeuron.addError(dendron.weight * self.gradient);
        self.error = 0;
```

 

## The Network Class

Now lets create the ***Network*** class

```python
class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None)) # bias neuron
            layer[-1].setOutput(1) # setting output of bias neuron as 1
            self.layers.append(layer)
```

 

So this is the initializer of the Network class. it’s taking an argument as input which is the network ***topology,***  this will be a list of numbers. For example, a topology = \[2,5,1] represents there are 3 layers in the network. First second and third layer containing 2,5,1 neurons respectively.  the first layer is always the input layer and last is always the output layer rest of them are hidden layers

So in the neuron class, we created an internal list called layers and we filled it with neurons according to the number of neurons specified in the topology

If its the first layer, then there is no previous layer thus the neurons were initialized with None parameters ***Neuron(None).***  at the end of the layer, we are adding a bias neuron with None parameter (no previous layer for bias)  and setting its output as 1

at last appending the ***layer*** the ***self.layers*** list

Now we need some helper methods in the Network class

```python
class Network:
    :
    :
    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

```

 

#### Setting the input

the first one is to set the inputs. The first layer of the network is the input layer thus its looping through all the input values and setting the output of the corresponding input neuron with the corresponding input value

#### Calculating overall Error of network

![formula for rms error](/images/uploads/imgb0001.png)

[![](https://web.archive.org/web/20201112015008im_/https://i1.wp.com/142.93.251.188/wp-content/uploads/2017/08/imgb0001-300x135.png?resize=184%2C83)](https://web.archive.org/web/20201112015008/https://thecodacus.com/neural-network-scratch-python-no-libraries/imgb0001/)next is measuring the error.  The error function is calculating the difference in output and the target and squaring it and summing all the squared error from all the output neurons and performing a square root. Which is basically a root mean square of all the output node errors.

#### Feed Forward The Network

```python
class Network:
    :
    :
    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword();
```

The feed forward method nor the network class is simple. We just have to call ***thefeedForword*** method  each and every neuron in the network from input to output layer in order

#### Back Propagate The Network

```python
class Network:
    :
    :
    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]: #reverse the order
            for neuron in layer:
                neuron.backPropagate()
```

For the back propagation. its same as feed forward but from output to input we have to call the back propagation of each and every neuron. but just like we need to set input before feed forward here we need to set the error for each and every output neuron. for other neurons, the next layer sends the error.

So for the output neuron, we have to set the error manually for each neuron in the layer. The error for each neuron will be target minus the output of that neuron

now we need to call the ***backPropagate***  of each and every neuron in the network from the output layer to the input layer in order

#### Get Results from the network

```python
class Network:
    :
    :
    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop() # removing the bias neuron
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o &gt; 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()# removing the bias neuron
        return output
```

 

We are getting the results in a list from the last layer in the network and popping out the last output which is the output of a bias neuron

and optionally I created another method  ***getThResults*** which is basically the thresholded version of the output on 0.5 as threshold

#### Lets Test The Network

```python
def main():
    topology = []
    topology.append(2)
    topology.append(3)
    topology.append(2)
    net = Network(topology)
    Neuron.eta = 0.09
    Neuron.alpha = 0.015
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
    while True:
        err = 0
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForword()
            net.backPropagate(outputs[i])
            err = err + net.getError(outputs[i])
        print "error: ", err
        if err &lt; 0.01:
            break

    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        net.setInput([a, b])
        net.feedForword()
        print net.getThResults() 


if __name__ == '__main__':
    main()
```

To test the network we are going to train to be a ***half adder.***  which is as follows

[![](https://web.archive.org/web/20201112015008im_/https://i0.wp.com/142.93.251.188/wp-content/uploads/2017/08/half-adder2-300x106.jpg?resize=492%2C174)](https://web.archive.org/web/20201112015008/https://thecodacus.com/neural-network-scratch-python-no-libraries/half-adder2/)

Where A, B is the input and Sum, Carry-out is the output

In the main function we defined a network topology to be \[2,3,2], we set the learning rate to be0.09 and alpha to be 0.015. we prepared the inputs and corresponding outputs.

In the while we are training the network for each input which is ***setInput-> feedForward-> backPropagate*** until the error is less than a threshold value

In the next while loop, we are taking input from the console and predicting the output using the network which is

***setInput-> feedForward->getThResults***

## We just created a Neural Network From Scratch

That’s it we created a neural network from scratch. Congrats you have the wisdom now.

the complete code will look like this

```python
import math
import numpy as np


class Connection:
    def __init__(self, connectedNeuron):
        self.connectedNeuron = connectedNeuron
        self.weight = np.random.normal()
        self.dWeight = 0.0


class Neuron:
    eta = 0.001
    alpha = 0.01

    def __init__(self, layer):
        self.dendrons = []
        self.error = 0.0
        self.gradient = 0.0
        self.output = 0.0
        if layer is None:
            pass
        else:
            for neuron in layer:
                con = Connection(neuron)
                self.dendrons.append(con)

    def addError(self, err):
        self.error = self.error + err

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x * 1.0))

    def dSigmoid(self, x):
        return x * (1.0 - x)

    def setError(self, err):
        self.error = err

    def setOutput(self, output):
        self.output = output

    def getOutput(self):
        return self.output

    def feedForword(self):
        sumOutput = 0
        if len(self.dendrons) == 0:
            return
        for dendron in self.dendrons:
            sumOutput = sumOutput + dendron.connectedNeuron.getOutput() * dendron.weight
        self.output = self.sigmoid(sumOutput)

    def backPropagate(self):
        self.gradient = self.error * self.dSigmoid(self.output);
        for dendron in self.dendrons:
            dendron.dWeight = Neuron.eta * (
            dendron.connectedNeuron.output * self.gradient) + self.alpha * dendron.dWeight;
            dendron.weight = dendron.weight + dendron.dWeight;
            dendron.connectedNeuron.addError(dendron.weight * self.gradient);
        self.error = 0;


class Network:
    def __init__(self, topology):
        self.layers = []
        for numNeuron in topology:
            layer = []
            for i in range(numNeuron):
                if (len(self.layers) == 0):
                    layer.append(Neuron(None))
                else:
                    layer.append(Neuron(self.layers[-1]))
            layer.append(Neuron(None))
            layer[-1].setOutput(1)
            self.layers.append(layer)

    def setInput(self, inputs):
        for i in range(len(inputs)):
            self.layers[0][i].setOutput(inputs[i])

    def feedForword(self):
        for layer in self.layers[1:]:
            for neuron in layer:
                neuron.feedForword();

    def backPropagate(self, target):
        for i in range(len(target)):
            self.layers[-1][i].setError(target[i] - self.layers[-1][i].getOutput())
        for layer in self.layers[::-1]:
            for neuron in layer:
                neuron.backPropagate()

    def getError(self, target):
        err = 0
        for i in range(len(target)):
            e = (target[i] - self.layers[-1][i].getOutput())
            err = err + e ** 2
        err = err / len(target)
        err = math.sqrt(err)
        return err

    def getResults(self):
        output = []
        for neuron in self.layers[-1]:
            output.append(neuron.getOutput())
        output.pop()
        return output

    def getThResults(self):
        output = []
        for neuron in self.layers[-1]:
            o = neuron.getOutput()
            if (o &gt; 0.5):
                o = 1
            else:
                o = 0
            output.append(o)
        output.pop()
        return output


def main():
    topology = []
    topology.append(2)
    topology.append(3)
    topology.append(2)
    net = Network(topology)
    Neuron.eta = 0.09
    Neuron.alpha = 0.015
    while True:

        err = 0
        inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
        outputs = [[0, 0], [1, 0], [1, 0], [0, 1]]
        for i in range(len(inputs)):
            net.setInput(inputs[i])
            net.feedForword()
            net.backPropagate(outputs[i])
            err = err + net.getError(outputs[i])
        print "error: ", err
        if err &lt; 0.01:
            break

    while True:
        a = input("type 1st input :")
        b = input("type 2nd input :")
        net.setInput([a, b])
        net.feedForword()
        print net.getThResults()


if __name__ == '__main__':
    main()
```

## Video Tutorial For Neural Network From Scratch



<iframe width="600" height="400" style="width:100%;" src="https://www.youtube.com/embed/O8BXapqL5lA" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>