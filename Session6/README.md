# Session6 : Advanced Convolutions


# Assignment

1. Run [this](https://colab.research.google.com/drive/1STOg33u7haqSptyjUL40FZIxNW4XdBQK) network. After training the network, whatever accuracy you get is your base accuracy. Epochs = 100  

    Fix the network above:
    - remove dense
    - add layers required to reach RF
    - fix kernel scaleup and down (1x1)
    - see if all dropouts are properly placed
    - Get accuracy more than the base accuracy in less number 100 epochs. 


2. Rewrite it again using these convolutions in the order given below:

    - Normal Convolution
    - Spatially Separable Convolution  (Conv2d(x, (3,1)) followed by Conv2D(x,(3,1))
    - Depthwise Separable Convolution
    - Grouped Convolution (use 3x3, 5x5 only)
    - Grouped Convolution (use 3x3 only, one with dilation = 1, and another with dilation = 2)  
    
    You must use all of the 5 above at least once. Train this new model for 50 epochs. 


## Files

**Assignment6A.ipynb**

-- This file shows the training of base model shared in session 6 for 100 epochs. Best test accuracy observed is 83.43 with 1,172,410 model parameters. After this file shows the modified model following concepts discussed in classes. It achieves best testing accuracy of 85.90% with 370,570 model parameters. The google colab file can be accessed [here](https://colab.research.google.com/drive/1VsokpQARK-YvGfyU52eJhTSfv8S8C9VS)

 
**Assignment6B.ipynb**

-- This file contains a cnn model defined using keras functional api. The model incorporates variety of convolutions like normal convolution, Spatially separable convolution, depthwise separable convolution, group convolution etc. It achieves test accuracy of 39.74% with 648,490 model parameters. The Google colab file can be accessed [here](https://colab.research.google.com/drive/1IQXNI75uGkqvqKTz7ljcyVeIkZlmHlad)


