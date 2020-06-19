# Session10 : GradCam with VGG16 on ImageNet Dataset


## Assignment

1. Refer to the GRADCAM code from previous session. Build GradCAM images for the one layer before the one we used, and one layer before this one. Show the results.  
2. Load this [image](https://media.karousell.com/media/photos/products/2018/08/20/16_scale_tony_stark_avenger3_headscrupt_with_glasses_1534759826_e79b0cf4.jpg). "Find"  "sunglasses" in the image using GradCAM


## Files

**Assignment10.ipynb**

-- Part 1:

In this assignment's 1st part, a pretrained VGG 16 model is loaded and for a given dog image (https://www.rspcapetinsurance.org.au/rspca/media/images/hero/dog-insurance-hero.jpg) GradCam is performed for last layer, one layer before last and two layer before last layer.

-- Part 2: 

In the second part, image of a person with sunglasses (https://media.karousell.com/media/photos/products/2018/08/20/16_scale_tony_stark_avenger3_headscrupt_with_glasses_1534759826_e79b0cf4.jpg) is loaded and using GradCam, sunglasses are detected in the image. The original google colab can be found at https://colab.research.google.com/drive/1I_XwDsQVO0KUoSCnTP_JK0QSKoBgh5et
