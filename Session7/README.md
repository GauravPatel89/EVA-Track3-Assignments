# Session7 : ENAS Implementation for CIFAR10 Dataset


## Assignment

1. Check [this](https://arxiv.org/pdf/1409.4842.pdf) paper out. They mention on page 6, that the RF is 224x224. Show calculations of Receptive Field for Network described in paper

2. Design following network (CIFAR10):
    <p align="center">
      <img width="800" height="300" src="https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/Session7/enasdiscoverednetwork.png">
    </p>  
The lines you see are the skip-connections. You need to add them.Train for 100 Epochs (add BN and ReLU after every layer).
        

## Assignment 7B: 
**Session7B.ipynb**
-- This is a google colab file containing implementation of ENAS model for CIFAR-10 dataset.
  The google colab file can be accessed [here](https://colab.research.google.com/drive/1Oo4-D9gF2sDjdXQhh48dMF-iWljQUe1Q)
  

## Assignment 7A: 
### Receptive Field calculations for GoogLeNet network

Following is the calculation for Receptive field of GoogLeNet model explained in https://arxiv.org/pdf/1409.4842.pdf. 
Considering that in academics networks start processing from size of 56x56 with preceding block common. We are also considering the network from the layer at which it start processing 56x56 image. We arrive at Receptive field of 225x225.

----
**Glossary**

j = jump

s = stride

k = kernel size

r<sub>Out</sub> = Receptive field Out

r<sub>In</sub> = Receptive field In

----
**Equations**

**j<sub>Out</sub> = j<sub>In</sub> x s**

**r<sub>Out</sub> = r<sub>In</sub> + (k-1) x j<sub>In</sub>**

-----
**Layer1: Conv 3x3 + 1(S)**

k,p,s = 3,1,1

n<sub>In</sub> = 56

r<sub>In</sub> = 1

j<sub>In</sub> = 1

j<sub>Out</sub> = 1 x 1 = 1

r<sub>Out</sub> = 1 + (3-1) x 1 = 3

-----
**Layer2: MaxPool 3x3 + 2(S)**

k,p,s = 3,0,2

r<sub>In</sub> = 3

j<sub>In</sub> = 1

j<sub>Out</sub> = 1 x 2 = 2

r<sub>Out</sub> = 3 + (3-1) x 1 = 5


-----
**Layer3: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 5

j<sub>In</sub> = 2

j<sub>Out</sub> = 2 * 1 = 2

r<sub>Out</sub> = 5 + (5-1) x 2 = 13


-----
**Layer4: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 13

j<sub>In</sub> = 2

j<sub>Out</sub> = 2 * 1 = 2

r<sub>Out</sub> = 13 + (5-1) x 2 = 21

-----
**Layer5: MaxPool 3x3 + 2(S)**

k,p,s = 3,0,2

r<sub>In</sub> = 21

j<sub>In</sub> = 2

j<sub>Out</sub> = 2 x 2 = 4

r<sub>Out</sub> = 21 + (3-1) x 2 = 25

-----
**Layer6: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 25

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 25 + (5-1) x 4 = 41

-----
**Layer7: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 41

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 41 + (5-1) x 4 = 57

-----
**Layer8: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 57

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 57 + (5-1) x 4 = 73

-----
**Layer9: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 73

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 73 + (5-1) x 4 = 89

-----
**Layer10: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 89

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 89 + (5-1) x 4 = 105

-----
**Layer11: MaxPool 3x3 + 2(S)**

k,p,s = 3,0,2

r<sub>In</sub> = 105

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 x 2 = 8

r<sub>Out</sub> = 105 + (3-1) x 4 = 113

-----
**Layer12: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 113

j<sub>In</sub> = 8

j<sub>Out</sub> = 8 * 1 = 8

r<sub>Out</sub> = 113 + (5-1) x 8 = 145

-----
**Layer13: Inception Block: Conv 5x5 + 1(S)**

k,p,s = 5,4,1

r<sub>In</sub> = 145

j<sub>In</sub> = 8

j<sub>Out</sub> = 8 * 1 = 8

r<sub>Out</sub> = 145 + (5-1) x 8 = 177

-----
**Layer14: Conv 7x7 + 1(S)**

k,p,s = 7,0,1

r<sub>In</sub> = 177

j<sub>In</sub> = 8

j<sub>Out</sub> = 8 * 1 = 8

r<sub>Out</sub> = 177 + (7-1) x 8 = 225

----
**Final Receptive Field = 225x225**

