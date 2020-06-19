# Session16 : YOLO V2


## Assignment

Refer to this TSAI COLAB [FILE](https://colab.research.google.com/drive/1iZdzI0VEG8ieRgHXKT7tNEE7iN3gw4tN). We have stitched multiple projects into 1, so you can train tiny-YOLO on COLAB!  

Refer to this blog: [LINK](https://medium.com/@today.rafi/train-your-own-tiny-yolo-v3-on-google-colaboratory-with-the-custom-dataset-2e35db02bf8f) and [LINK](https://github.com/rafiuddinkhan/Yolo-Training-GoogleColab/blob/master/helmet.ipynb). This is the main source for our project. 

Refer to the "main.py" in Step 1. 
Here is what you need to do:
- create your own dataset using main.py.
- collect 200 images (more the better) for any 1 class of your choice. e.g. this project is for the helmet. You cannot use the same dataset as in this project. 
- annotate 200 images as explained in the links above
- replace the data_for_colab folder with your custom folder
- train YOLO for 1000 epochs (more the better)
- download 1 youtube video  which has your class, 
- run this command

            !./darknet detector demo data_for_colab/obj.data data_for_colab/yolov3-tiny-obj.cfg backup/yolov3-tiny-obj_1000.weights  -dont_show youtube_video.mp4 -i 0 -out_filename veout.avi

 - upload your video on YouTube.
    

## Group Members

Atul Gupta (samatul@gmail.com)

Gaurav Patel (gaurav4664@gmail.com)

Ashutosh Panda (ashusai.panda@gmail.com)


## Files

**Assignment16YOLO.ipynb**

In this assignment code,We used yolo trained weights to detect Cat Images from Cat video.

**We Annotated only Cat Faces and used that for the training**

In the below Youtube link we have uploaded the video in which cat images are being detected from video.

https://youtu.be/THiM7_LKFR0

The Data folder Contains 200 original and annotated cat images.
