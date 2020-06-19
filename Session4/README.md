# Session4 : DNN Architecture Search

## Assignment

 1. Explain following concepts

    How many layers,
    MaxPooling,
    1x1 Convolutions,
    3x3 Convolutions,
    Receptive Field,
    SoftMax,
    Learning Rate,
    Kernels and how do we decide the number of kernels?
    Batch Normalization,
    Image Normalization,
    Position of MaxPooling,
    Concept of Transition Layers,
    Position of Transition Layer,
    Number of Epochs and when to increase them,
    DropOut
    When do we introduce DropOut, or when do we know we have some overfitting
    The distance of MaxPooling from Prediction,
    The distance of Batch Normalization from Prediction,
    When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)
    How do we know our network is not going well, comparatively, very early
    Batch Size, and effects of batch size,
    When to add validation checks,
    LR schedule and concept behind it,
    Adam vs SGD
    etc 

2. Write your Assignment 3 again such that after 4 code iterations achieve:  
   - 99.4% accuracy
   - Less than 15k Parameters
   - Have started from a Vanilla network (no BN, DropOut, LR, larger batch size, change in Optimizer, etc)
   - Make sure you are tracking your code's performance, and writing down your observations are you achieve better or worse results
   - Your second code can only have max 2 improvements over first one, third can have only max 2 over second and so on. 
   - All of your iterations are in different CODE BLOCKS and named properly like First CODE, Second CODE, etc
   - All of your iterations have a Header note, describing what all you are planning to do in this code
   - All of your code is very well documented
   - There is a readme file (or embedded documentation) describing your codes and steps you've taken



## Files

**ArchitecturalBasics.md**
> This file describes important terms and points to consider in design of CNN Architecture and training.

**Session4_ArchitecturalBasics.ipynb**
> This is a google colab file containing codes for MNIST training. Final result achieved is validation accuracy of 99.45% with 9310 parameters. The google colab file can be found [here](https://colab.research.google.com/drive/1qtshfvPd-lcKDYKszfCSBZgBymJLgEpP)

>  Code follows incremental approach to improve the model performance. This helps in understanding how different techniques and algorithms affect the performance of the CNN model.
 
