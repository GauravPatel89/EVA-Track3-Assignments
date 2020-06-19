# Session3 : First Neural Networks

## Assignment

Open this [LINK](http://bit.ly/2IBqQJD) in your browser. Copy the file to your collaboratory
You need to:
1. write comments for all the cells
2. Define a new network such that:  
    - It has less than 20000 parameters
    - It achieves validation accuracy of more than 99.4% (basically print(score) should be more than 0.994




## Files
**Session3.ipynb**
-- this is a google colab file containing commented file of session3.
  The file can also be found at https://colab.research.google.com/drive/15qnS1aSNrt5po4BNr73aixSP06rKqxIn
  
  
I experimented with various model layers, batch size and number of epochs. 

For this particular model
Once I finalised the layers, while training I observed that with smaller batch sizes, although training time was large, model was reaching validation accuracy of as high as 99.2 % (epoch 21) in lesser epochs. Once that kind of accuracy is reached when I trained the network further with larger batch size and very large number of epochs, I could consistently achieve validation accuracy of 99.3% and more. Here in this file I could reach highest accuracy of 99.34% (Epoch 60). I could not improve the validation accuracy any further although the training accuracy reached 100 %.

Following such methodology I could achieve accuracy of 99.36%.


  
 
