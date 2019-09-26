# Architectural Basics

## **1.How many layers**
           
>In a CNN, primary aim is that at the end of the network we should have seen the entire image (case where object covers entire image) or the region size we are targetting in the image. If we had infinite computing power we could have taken kernels of the size of image and be done with it in 1 layer but such s not the case. Hence we accomplish the task of covering the image in number of steps i.e. layer by layer. Hence number of layers will be mainly governed by hardware capacity we have.
           
## **3x3 Convolutions**

>3x3 is smallest odd size kernel with axis of symmetry

>3x3 can be used to achieve receptive field equivalent to higher kernel size eg. 5x5(apply 3x3 twice),7x7(apply 3x3 trice) etc with lesser number of parameters.

>GPUs from nVidia etc have optimised hardware for 3x3 kernel.

>Hence in our CNNs we generally use 3x3 kernels

##**Receptive Field**

>Receptive field is the area of the input that the a feature is seeing. We have two types of RF.

##**Kernels and how do we decide the number of kernels?**


## **MaxPooling**

>MaxPooling reduces the size of the channel by carrying forward only max values in given filter region eg. 2x2 MaxPooling will carry 25% values forward

>We prefer 2x2 maxPooling because if we go for larger size the loss will be too high and we might lose important information

##**Position of MaxPooling**

## **1x1 Convolutions**

>1x1 convolutions helps us to bring down the number of channels. It does this by combining large number channels into important fewer chnnels by weighted average of sort. What it effectively does is combine contextually related channels to form common channel. Eg. suppose inputs are human hand, human leg, human eyes, dog leg, dog eyes, cat leg, cat eye etc. all jumbeled up, 1x1 will try to combine human features together, similarly for dog and cat.

>Although its called so its not a convolution because it is just multiplications with entire images followed by sum of it. Hence it is much faster compared to convolution operation. 

##**SoftMax**
##**Learning Rate**

##**Batch Normalization**
##**Image Normalization**

##**Concept of Transition Layers**
##**Position of Transition Layer**
##**Number of Epochs and when to increase them**
##**DropOut**
##**When do we introduce DropOut, or when do we know we have some overfitting**
##**The distance of MaxPooling from Prediction**
##**The distance of Batch Normalization from Prediction**
##**When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)**
##**How do we know our network is not going well, comparatively, very early**
##**Batch Size, and effects of batch size**
##**When to add validation checks**
##**LR schedule and concept behind it**
##**Adam vs SGD**
