### Initializations:

![Initializations](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Initialization.png)

Import all the necessary packages and libraries.

### Step 1:
![Step 1](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step1.png)

Define ReplayBuffer class. Object of this class will be used to store the experiences of the agent while acting in the environment. It stores each experience as a tuple **<state,next_state,action,reward,done>**. Its size will be specified as **max_size** at the time of its instance creation.

To utilize this class following methods are defined.

1. add(transition)  
Adds a transition (an experience tuple) into ReplayBuffer. If the memory is full it will add the transition to the beginning. Effectively it implements a circular memory buffer.

2. sample(batch_size)  
It creates a batches of **state,next_state,action,reward and done** of size **batch_size** by uniformly sampling the ReplayBuffer.
These batches are returned as numpy arrays as seen in the last code statement.


### Step 2:
![Step 2](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step2.png)
![](file:///D:\gaurav\eva\session9\Assignment\images\Step2.png)

Define Actor class. This class defines the model or network used by Actors. In TD3 algorithm we need two actors, actor_model and actor_target. Since both the actors need to be identical we define a common Actor class. We will use two instances of this class as actor model and actor target.  
We have two methods in this class.

1. \_\_init__(self, state_dims, action_dim, max_action)  
In this method we define the layers used by Actor class. These layers will be connected together later in the **forward()** method to create a network.  
We define 3 layers. Each of the layers are linear layers meaning they are fully connected layers.

2.  forward(self,x)  
This method links the layers defined in the above method to create a complete network. The network takes *state_dim* number of inputs and gives *action_dim* number of outputs. The first two layers have *relu* as its activation functions. The last layer has *tanh* as activation function. It must be noted the output of final layer is multiplied by *max_action*, since output of *tanh* varies from -1 to 1, output of Actor network will vary from *-max_action* to *max_action*.
    

### Step 3:
![Step 3](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step3.png)
![](file:///D:\gaurav\eva\session9\Assignment\images\Step3.png)
Define Critic class. This class defines the model or network used by Critics. In TD3 algorithm we need two pairs of competing critics each pair consisting of a critic model  and a critic target. All of these 4 critics need to be identical. Here in this class we jointly define two competing critics. Although they are defined together they both act independently. We will use two instances of this class as critic model and critic target
We have 3 methods in this class.
1. \_\_init__(self, state_dims, action_dim)  
In this method we define the layers used by Critic class. These layers will be connected together later in the **forward()** method to create a network.  
We define 3 layers. Each of the layers are linear layers meaning they are fully connected layers.

2.  forward(self,x)  
This method links the layers defined in the above method to create a complete network. The network concatenates the state and action to generate an input of size *(state_dim + action_dim)*. It generates a single state value as its output. First two layers have *relu* as its activation functions while the last layer has no activation function.  
3. Q1(self, x, u)
This method is similar to *forward()* method except that here forward pass of only the first critic has been defined. This is in order to save the computation in cases where output of only first critic is required hence we need not compute forward pass of both the critics. This method will be useful in **Step 13** while updating the actor model. 



### Step 4-15:
![Step 4-15](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step4-15.png)
![](file:///D:\gaurav\eva\session9\Assignment\images\Step4-15.png)

Define TD3 class. This class will define our entire TD3 algorithm. It will define all the Actor, Critic objects and its optimizers. It will also define the training algorithm. Two methods of the class are shown in above image.

1. \_\_init__(self, state_dim, action_dim, max_action)  
In this method we define all the objects of TD3 algorithm  
* actor (Acts as Actor Model)
* actor_target (Acts as Actor Target)
* actor_optimizer (Adam optimizer for training Actor model)
* critic (Acts as Critic Model. It has 2 competing Critics inbuilt)
* critic_target (Acts as Critic Target. It has 2 competing Critics inbuilt)
* critic_optimizer (Adam optimizer for training Critic model)

  We also copy Actor Model into Actor Target (This is done to have identical model and targets at the beginning)  

        self.actor_target.load_state_dict(self.actor.state_dict())
  Similarly we copy Critic Model into Critic Target  

        self.critic_target.load_state_dict(self.critic.state_dict())
 2. select_action(self, state)  
 In this method we find the best action for given state.  
 For this  we first reshape input *state* into a row vector then convert it to a pytorch tensor because the torch model expects a tensor. As shown below.  
 
        state = torch.Tensor(state.reshape(1, -1)).to(device)  
        
    Here *to(device)* transfers state to a cpu or gpu as per the availability.  
    Next, the *state* tensor is forward passed through the Actor Model network to get the computed *action*. The action is converted to numpy and flattened (make 1D) before returning.
### Step 4:
![Step 4](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step4.png)
![](file:///D:\gaurav\eva\session9\Assignment\images\Step4.png)
Define Train Method of TD3 class.  
For each of the iterations, sample a batch of size *batch_size* from ReplayBuffer.  

    batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
  If we refer to *sample()* method of ReplayBuffer class, we were returning batch components as numpy_arrays while the pytorch networks expect inputs as tensors hence we convert all of the sampled components into pytorch tensors.  
  
    state = torch.Tensor(batch_states).to(device)
    next_state = torch.Tensor(batch_next_states).to(device)
    action = torch.Tensor(batch_actions).to(device)
    reward = torch.Tensor(batch_rewards).to(device)
    done = torch.Tensor(batch_dones).to(device)

### Step 5:
![Step 5](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step5.png)
![](file:///D:\gaurav\eva\session9\Assignment\images\Step5.png)
Generate actions for next_states through Actor Target. It must be noted here that *next_state* here corresponds to entire batch of next states obtained by sampling the ReplayBuffer in the previous step. Also we are using the Actor Target and not Actor Model. In TD3 algorithm for acting purpose Actor Model is used while for equation calculations Actor Model is used. *next_action* obtained here will be used in calculation of Q-Values.

### Step 6:
![Step 6](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step6.png)
![](file:///D:\gaurav\eva\session9\Assignment\images\Step6.png)
Add noise to predicted actions. 
Noise values are sampled from a normal distribution with 0 mean and *policy_noise* as standard deviation. *noise* here is a tensor of size same as *batch_actions* i.e. batch_size .

    noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)
We don't want noise values to be too high hence we clip the noise value.  

    noise = noise.clamp(-noise_clip, noise_clip)
Finally we add the noise to *next_action*. We again clip the values to range of allowed action values.

    next_action = (next_action + noise).clamp(-self.max_action, self.max_action)



### Step 7:
![Step 7](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step7.png)
Obtain Q-Values from Critic Targets for sampled batch of *next_state* and *next_actions* generated by Actor Target.  

    target_Q1, target_Q2 = self.critic_target(next_state, next_action)
It must again be noted that we are using Critic Target here since these Q-values will be used in equation computations. When we are acting in the environment we will used Critic Model. 

### Step 8:
![Step 8](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step8.png)

Select the minimum of two Critic Q-values.

    target_Q = torch.min(target_Q1, target_Q2)

### Step 9:
![Step 9](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step9.png)
Compute the expected Q value for current state using the Bellman Equation. This value will be compared with predicted Q-Value later to get the temporal difference. 

    target_Q = reward + ((1 - done) * discount * target_Q).detach()
Some important points.  

* Multiplication by *(1 - done)* :  
This is to take care of the terminal states (episode ends).  
So whenever the transition is a terminal state, we will have *done=1* hence *(1-done)=0* so discount part will be made 0. This is because in this case *next_state*,*next_action* and *target_Q* will be meaningless.  
Whenever transition is non-terminal state, *target_Q* will be normally evaluated because *done=0*

* *detach()* 
This is to detach *target_Q* computation from the computation graph. Doing this, pytorch will not track the operations on this subgraph.

### Step 10:
![Step 10](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step10.png)
Obtain predicted Q-Values from Model Critics for current state and action.  

    current_Q1, current_Q2 = self.critic(state, action)
    
Here we obtain Q-Values from two critics for a batch *current_state* and *current_action* obtained through ReplayBuffer sampling. It must be noted here that we are using Critic Model and not Critic Target. 


### Step 11:
![Step 11](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step11.png)
Calculate Critic loss. We have already calculated expected Q-Values (*target_Q*) and predicted Q-Values (*current_Q1, current_Q2*). We find the temporal difference between them to get Critic loss. 

    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

It must be noted that we have calculated combined loss of Critic 1 and Critic 2, this is because both the critics are defined in Critic class together. So *critic_loss* will be used for combined training of the critics.

### Step 12:
![Step 12](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step12.png)  
Update Critic Model through back propagation.  

    self.critic_optimizer.zero_grad()
    critic_loss.backward()
    self.critic_optimizer.step()
    
We first make gradients of *critic_optimizer* zero. Then we propagate *critic_loss* backwards so that gradients are recorded. Finally we ask *critic_optimizer to update Critic Model. 

It must be noted here that we are updating Critic Model. Critic Targets are never updated through backpropagation. They are updated throgh Prolyak Averaging.



### Step 13:
![Step 13](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step13.png)

Update Actor Model through backpropagation every *policy_freq*'th iteration.

    if it % policy_freq == 0:
        actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

We first find whether this iteration is divisible by *policy_freq*
    
    if it % policy_freq == 0:
    
We calculate *actor_loss*. We need to update Actor Model parameters such that action taken by it maximizes Q-values output of Critic Model. So we select *actor_loss* as follows. 

    actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
Since we have -ve sign, optimizer will perform gradient ascent on Actor Model parameter so as to maximize Critic Q-Value.  
*mean()* here corresponds to average of *actor_loss* over a batch.

Next we initialize actor_optimizer gradients to zero, perform backward pass to record gradients and finally step through to update Actor Model parameters.
    
### Step 14:
![Step 14](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step14.png)
Update Actor Target through Polyak Averaging.

    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
          
We generate an iterator of matched pair of Actor Model and Actor Target parameters by zipping them as

    zip(self.actor.parameters(), self.actor_target.parameters())
    
We now loop through this iterator and copy Actor Model parameters into Actor Target with Polyak averaging.

### Step 15:
![Step 15](https://github.com/GauravPatel89/EVA-Track3-Assignments/blob/master/P2S9/Images/Step15.png)
Update Critic Target through Polyak Averaging.

    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
          target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
          
We generate an iterator of matched pair of Critic Model and Critic Target parameters by zipping them as

    zip(self.critic.parameters(), self.critic_target.parameters())
    
We now loop through this iterator and copy Critic Model parameters into Critic Target with Polyak averaging.


    
