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
This is the Google Colab file for TD3 training. 'endgameTD3.py' is simple .py version of this file. This file can be accessed on Google Colab [here.](https://colab.research.google.com/drive/1ofPhEBYm1s99HXcnZffW4uyVQcD0rTcA?usp=sharing)

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
State for the environment consists of list of 4 elements  
1. Current state image:  
Current state image for the environment is a cropped view of road network in car's front view i.e. how car is viewing the area around it. In this view car is always facing front but area around it changes as car navigates as shown below.

![stateImg](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/EndGame/Figures/stateImage.gif)

