# Session9 : GradCam


## Assignment
1. Take your 6A Code (your optimized version, not the base version with Dense layer), and learn how to integrate gradCAM with your code. Learn more about gradCAM [here](http://www.hackevolve.com/where-cnn-is-looking-grad-cam/).  
Test 4 images (remember the image ids or names) from your network and show the visualization like this: enter image description here


2. Train your 6A model again, but this time add CutOut. Use this [link](https://github.com/yu4u/cutout-random-erasing) for reference.  
Show the same 4 images again with gradCAM's result. 
    
## Files

**Assignment9.ipynb**
-- In this assignment's 1st part, model from assigment 6A is trained. Then in 2nd part same model with is trained with cutout data augmnetation. Finally outputs of both these models with gradCam heatmap superimposed, are compared side-by-side. The original google colab can be found at https://colab.research.google.com/drive/1wsu5BY9fGNYUfWpypH48k8zDdnRf6wmw

 
