## Project EndGame: Car Navigation using Deep Reinforcement Learning

### Problem Statement:  
Simulate Car navigation on city map. Objective is to teach car to reach different goals on the city map while traveling on road using **Twin Delayed DDPG (TD3)** algorithm. 

### Submission:

Click following image to play video.
[![EndGame Final Submission](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/endGameVideoSnapShot.png)](https://youtu.be/77mfBYC8f0g)


#### Files:
1. 'endGameEnv.py' :  
This file contains Environment Class definition for simulation of the Car Navigation Game. For ease of use the environment class has been defined on the lines of standard Gym environments i.e. class has methods like reset(),step(),render(),sampleActionSpace(). Environment has been explained in detail later.

2. 'endGameModels.py' :  
This file contains definition of Actor and Critic DNNs, ReplayBuffer used for storing step transitions and TD3 class which implements TD3 algorithm.

3. 'endGameInference.py' :  
This file contains code for evaluating the trained models. This is done by instantiating TD3 class object, loading saved Actor and Critic models and repeatedly generating action and taking action on defined environment to generate visualization.

4. 'endGameUtilities.py' :  
This file contains some of the utilities used by other files.

5. 'endGameEnvAllGoals.py' :  
This file has Environment definition similar to 'endGameEnv.py'. Difference is 'endGameEnv.py' environment's episode runs for only 3 random goal values while environment defined in this file runs the episode untill all the goals in the goalList have been achieved or car gets stuck to boundaries. This file useful for evaluating how model is working for all the goal values. It was used for generating the submission video.

6. 'endgameTD3.py' :  
This file contains code for TD3 training. It combines components in other files to create the main training loop.

7. 'endGameTD3.ipynb'
This is the Google Colab file for TD3 training. 'endgameTD3.py' is simple .py version of this file. This file can be accessed on Google Colab [here.](https://colab.research.google.com/drive/1S3kT0hJlK4Uzh10DrAFE55l9OtZAbLyc?usp=sharing)

#### Directories
1. 'pytorch_models':  
This directory contains best saved models for Actor and Critic. Example code for evaluating and generating a video for this models can be found in ['endGameInference.py'](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/endGameInference.py). 

2. 'images':  
This directory contains image files used by the carEndgameEnv environment. It has following files
    
    a. 'car.png': Used for visualization of Car on city map. 
    b. 'citymap.png' : Used as city map
    c. 'MASK1.png': Used for representing road and non-road pixels in city map

3. 'Figures':  
This directory contains support files for README.md file
    
### Environment

To simulate the car navigation task an environment has been defined. Environment definition is similar to standard Gym environments. Important functions like step(),reset(),render() have been provided for ease of use.
Important components and utilities of the environment are explained below.

#### State space:
State for the environment consists of list of 4 elements.  
1. Current state image:  
Current state image for the environment is a cropped view of road network in car's front view i.e. how car is viewing the area around it. In this view car is always facing front but area around it changes as car navigates as shown below.  
![stateImg](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/stateImage.gif)  
Currently it is selected to be 40x40 size image.This image is estimated by cropping an area twice the required crop size, rotating it by (90-car.angle) then again cropping it to required crop size. The state image is normalized by max pixel value i.e. 255 to normalize it to range [0.0,1.0]

2. Normalized Distance to Goal:  
This value corresponds to euclidean distance of Goal from car normalized by max possible goal distance (diagonal distance of citymap image).

3. Goal Orientation:  
This value corresponds orientation of Goal wrt to Car's current inclination angle. It is calculated as angular difference between vector1, joining car location and goal location, and vector2, representing Car's pointing direction.

4. -ve Goal Orientation: 
This value is same as previous value but with -ve sign.

#### Action space:  
Action space for this environment is 1 dimensional i.e. just one value, 'angle of rotation'. For each of the env.step(action) execution, car is first rotated by given 'action' and then displaced as per its velocity along 'car.angle'.





