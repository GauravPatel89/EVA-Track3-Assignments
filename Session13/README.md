# Session13 : Understanding ResNet

## Assignment

Refer to your Assignment 12. Replace whatever model you have there with the ResNet18 model.
Your model must look like Conv->B1->B2->B3->B4 and not individually called Convs. 
Ensure  

- Use Batch Size 128
- Use Normalization values of: (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
- Random Crop of 32 with padding of 4px
- Horizontal Flip (0.5)
- Optimizer: SGD, Weight-Decay: 5e-4
- NOT-OneCycleLR
- Save model (to drive) after every 50 epochs or best model till now

Describe your blocks, and the stride strategy you have picked. Train for 300 Epochs
Assignment Target Accuracy is 90%, so exit gracefully if you reach 90%.

## Files

**Session13.ipynb**

In this assignment a ResNet18 model has been trained on CIFAR10 datase with target accuracy of 92%.
Google Colab file can be found [here](https://colab.research.google.com/drive/11Mwd_NeGiZYcFZ23L9X4T2HdROhBKSp3)
