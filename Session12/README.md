# 94% Accuracy on CIFAR10 using One Cycle training policy

Cifar10 is a classic dataset in the field of deep learning. It has 60000 colour images of size 32×32 images belonging to 10 different classes (6000 images/class). Training Cifar10 to accuracy of 94% is quite challenging and doing it in few 100 seconds and in cost effective way is even more challenging. DAWNBench competition (https://dawn.cs.stanford.edu/benchmark/index.html#cifar10) targets this particular aspect. One of the winner model(Nov 2018 - Apr 2019) on DAWNBench, is by David C. Page. It is a custom built 9-Layer ResNet model. It takes just 1 min 15 sec to achieve 94.08% accuracy. Following is an implementation of model used by David C. Page and trained using One cycle training policy.

Complete code can be found at https://colab.research.google.com/drive/1FL6G-b9PsD5wzITORZnSyjbwLdEG6dK0#scrollTo=172sWTxXxgJ1
Code segments are described below.
 
1.Import necessary modules


     import numpy as np
     import time, math
     from tqdm import tqdm_notebook as tqdm
     import tensorflow as tf
     import tensorflow.contrib.eager as tfe
    
    
2.Enable tensorflow eagermode. In tensorflow everything is computational graph so if we wanted to debug or check
  small part of our code, it will not be possible unless entire graph has been defined and we run entire graph. This can be cumbersome process. Enabling tensorflow eagermode allows us to evaluate code segments immediately without building graphs. For more information refer https://www.tensorflow.org/guide/eager

    tf.enable_eager_execution()
    
3.Use Forms to parameterize the code. Using this we can allow user input for adjusting parameters in our code. We parameterize Batch size, momentum, learning rate, weight decay and number of epochs.

    BATCH_SIZE = 512 #@param {type:"integer"}
    MOMENTUM = 0.9 #@param {type:"number"}
    LEARNING_RATE = 0.4 #@param {type:"number"}
    WEIGHT_DECAY = 5e-4 #@param {type:"number"}
    EPOCHS = 24 #@param {type:"integer"}
    
4.Define a function to initialize layer weights. This function has been defined here because DavidNet model uses PyTorch but the way PyTorch intializes model layer weights has no equivalent in tensorflow.

    def init_pytorch(shape, dtype=tf.float32, partition_info=None):
         fan = np.prod(shape[:-1])
         bound = 1 / math.sqrt(fan)
         return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)
         
5.Define a class for basic Conv2D layer followed BatchNormalization,DropOut and ReLU activation.  This class has been inherited from "tf.keras.Model" class thus it inherits training and inference functionalities. Instances of this class can be used as parts or building blocks of our DNN model. 

The class groups number of layers into a single class object. These layers are defined in __\_\_init\_\_()__  and how these layers are connected i.e. model forward pass is defined in  __call()__. Output of basic Conv2D layer is passed to Dropout which is in turn passed to Batchnormalization followed by ReLU activation.

Number of kernels are passed as parameter _c_out_. Padding is set as _SAME_ so input and output channel size remains same. 
kernels are initialized using previously defined _init_pytorch()_ function. 

    class ConvBN(tf.keras.Model):
      def __init__(self, c_out):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.drop = tf.keras.layers.Dropout(0.05)

      def call(self, inputs):
        return tf.nn.relu(self.bn(self.drop(self.conv(inputs))))
        
6.Define a class for custom ResNET block. Just like __ConvBN__  various components of ResNet, like basic Conv block, pooling layer and residual connection are defined in __\_\_init\_\_()__ and how these layers are connected together is defined in __call()__. Output of Basic ConvBN block is passed to pooling layer and if residual connection is required connection consisting of two ConvBN blocks is added. 

Number of kernels are passed as parameter _c_out_. Pooling layer to be used is passed as _pool_ parameter. _res_ parameter defines wheather residual connection is required.

    class ResBlk(tf.keras.Model):
      def __init__(self, c_out, pool, res = False):
        super().__init__()
        self.conv_bn = ConvBN(c_out)
        self.pool = pool
        self.res = res
        if self.res:
          self.res1 = ConvBN(c_out)
          self.res2 = ConvBN(c_out)

      def call(self, inputs):
        h = self.pool(self.conv_bn(inputs))
        if self.res:
          h = h + self.res2(self.res1(h))
        return h

7.Define the full model. Again the same procedure is followed. We define a DavidNet class. Various layers are defined in __\_\_init\_\_()__ and their connection chain is defined in __call()__ . Complete model is shown in the image below.

<img src="https://github.com/davidcpage/cifar10-fast/blob/d31ad8d393dd75147b65f261dbf78670a97e48a8/net.svg">

It has initial ConvBN block with 64 kernels. Followed by a ResNet block with residual connection with 128 kernels, then ResNet block with 256 kernels and no residual connection. This is followed by ResNet block with 512 kernels and residual connection. After this we have globalmMaxPooling layer. Output of GlobalMaxPooling is then passed to a Dense layer to get 10x1x1 output. These outputs are scaled by 0.125 and passed to a softmax layer to get the final output. 

We must pay attention to some of the steps.

* _h = self.linear(h) * self.weight_

This step applies scaling to the output of dense layer. self.weight is a hyperparameter which is manually tuned to 0.125 by DavidNet. Explanation given by David is 

"_By default in PyTorch (0.4), initial batch norm scales are chosen uniformly at random from the interval [0,1]. Channels which are initialised near zero could be wasted so we replace this with a constant initialisation at 1. This leads to a larger signal through the network and to compensate we introduce an overall constant multiplicative rescaling of the final classifier. A rough manual optimisation of this extra hyperparameter suggest that 0.125 is a reasonable value. (The low value makes predictions less certain and appears to ease optimisation.)_" 

So basically batch normalization leads to slightly larger signal at the output. It is scaled down using hyperparameter weight factor. This parameter is manually tuned to 0.125 in this case.

* _ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)_

This step calculates softmax cross entropy i.e. softmax classification error between h(model prediction) and y (true labels).

* _loss = tf.reduce_sum(ce)_ 

This step sums the error cross entropy computed in previous step.

* _correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))_ 

In this step _tf.argmax()_ gets actual class id from one-hot encoded _h_ vector. then _tf.math.equal()_ compares the predicted and true labels elementwise output of this is in the form of array of True and False values. _tf.cast(*,tf.float32)_ converts the truth values to float32 numbers. Finally _tf.reduce_sum()_ is used to sum the numbers to in effect find total number of true predictions. Function _call()_ returns loss value and total correct predictions.


     class DavidNet(tf.keras.Model):
       def __init__(self, c=64, weight=0.125):
         super().__init__()
         pool = tf.keras.layers.MaxPooling2D()
         self.init_conv_bn = ConvBN(c)
         self.blk1 = ResBlk(c*2, pool, res = True)
         self.blk2 = ResBlk(c*4, pool)
         self.blk3 = ResBlk(c*8, pool, res = True)
         self.pool = tf.keras.layers.GlobalMaxPool2D()
         self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
         self.weight = weight

      def call(self, x, y):
        h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
        h = self.linear(h) * self.weight
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
        loss = tf.reduce_sum(ce)
        correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
        return loss, correct
    
8.Now with our model ready we can proceed to prepare our data. First we load the standard cifar10 dataset and reshape it. Next we have to do padding by 4. This is done as follow

 _np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')_

It tells the padding function that dont pad first (image id) and last dimension(channel number). Pad dimension 2 (rows) and 3 (columns) with 4 elements before and after the array. The padded elements are reflections of the elements anchored on first and last elements. i.e. [1,2,3,4,5,6] will be padded (4,4) as [5,4,3,2,**1,2,3,4,5,6**,5,4,3,2].

After padding normalize the data by subtracting mean and dividing by standard deviation of the data.

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    len_train, len_test = len(x_train), len(x_test)
    y_train = y_train.astype('int64').reshape(len_train)
    y_test = y_test.astype('int64').reshape(len_test)

    train_mean = np.mean(x_train, axis=(0,1,2))
    train_std = np.std(x_train, axis=(0,1,2))

    normalize = lambda x: ((x - train_mean) / train_std).astype('float32') # todo: check here
    pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')

    x_train = normalize(pad4(x_train))
    x_test = normalize(x_test)
    
9.Next steps to be done are setting up Learning rate scheduler,Momentum optimizer and data augmentation. 

* LR Scheduler 

_lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]_ 

Here _[t]_ represents x coordinates at which we need interpolated values. 

[0, (EPOCHS+1)//5, EPOCHS] represents known x coordinates

[0, LEARNING_RATE, 0] represents known y coordinates. 

So basically we are providing 3 keypoints (0,0), ((EPOCHS+1)//5,LEARNING_RATE) and (EPOCHS,0) to the _np.interp()_ and expecting interpolated values at x coordinates provided in _[t]_. 

* Momentum optimizer 

_opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)_

It uses inbuilt tensorflow momentum optimizer. It takes learning rates as input and computes momentum at each step.

* Data augmentation 

_data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)_ 

We have already padded (with 4 elements) and normalized our data. We now randomly crop 32x32x3 image. Next we perform random horizontal flips.


    model = DavidNet()
    batches_per_epoch = len_train//BATCH_SIZE + 1

    lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
    global_step = tf.train.get_or_create_global_step()
    lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/BATCH_SIZE
    opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)
    data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)

9.Now we are all set to train our model. Some important steps are as follows.

* Generate test batches

_test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)_

Here _from_tensor_slices((x_test, y_test))_ will create a Dataset whose elements are slices of tensors _x_test_ and _y_test_. It gives us slices of input data combined with its ground truth labels. 

_batch(BATCH_SIZE)_ method will generate batches of size _BATCH_SIZE_ from the dataset.

Effectively this piece of code will generate a batchwise test data similar to imageDataGenerator in Keras.

* Generate train batches each epoch

_train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(data_aug).shuffle(len_train).batch(BATCH_SIZE).prefetch(1)_

Similar to test set, _rom_tensor_slices((x_train, y_train))_ will create a dataset from _x_train_ and _y_train_. further we apply data augmentation we prepared earlier using _map(data_aug)_. Next we shuffle the data using _shuffle(len_train)_. Finally we generate batches of size _BATCH_SIZE_ from the prepared dataset. 

To speed up the training process we make use of multithreading functionality provided by tensorflow. This is done using _prefetch()_ method. What it does is, when the model is executing training step s it prepares the data for step s+1. Here _batch(BATCH_SIZE).prefetch(1)_ indicates that 1 batch of size BATCH_SIZE will be prefetched while one batch is being consumed by training process.

* Set learning phase 

_tf.keras.backend.set_learning_phase(1)_

This sets learning phase as '1' indicating training phase. This is passed as input to keras functions that uses a different behavior at train time and test time.(0 = test, 1 = train).

* Perform training

Iterate through the training set : _for (x, y) in tqdm(train_set)_

Function tqdm displays progress bar as we proceed through the dataset. 

For each loaded data, evaluate the model to find loss and number of correct predictions : _loss, correct = model(x, y)_

To perform training we will be needing weight gradients. This is accomplished using _tf.GradientTape()_. When this is used all the operations in its context manager are recorded. All the trainable parameters are watched and we can find gradient for it using _tape.gradient(loss, var)_. 

Setup _GradientTape_ inside for loop:  _with tf.GradientTape() as tape:_

Find gradient of loss w.r.t model parameters :

    var = model.trainable_variables
    grads = tape.gradient(loss, var)
    
Apply weight decay:

    for g, v in zip(grads, var):
       g += v * WEIGHT_DECAY * BATCH_SIZE
    
we add this to 
\begin{equation}
w_i \leftarrow w_i-\eta\frac{\partial E}{\partial w_i}-\eta\lambda w_i.
\end{equation}






        why:  train_loss += loss.numpy()
        train_acc += correct.numpy()

    t = time.time()
    test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)

    for epoch in range(EPOCHS):
      train_loss = test_loss = train_acc = test_acc = 0.0
      train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(data_aug).shuffle(len_train).batch(BATCH_SIZE).prefetch(1)

      tf.keras.backend.set_learning_phase(1)
      for (x, y) in tqdm(train_set):
        with tf.GradientTape() as tape:
          loss, correct = model(x, y)

        var = model.trainable_variables
        grads = tape.gradient(loss, var)
        for g, v in zip(grads, var):
          g += v * WEIGHT_DECAY * BATCH_SIZE
        opt.apply_gradients(zip(grads, var), global_step=global_step)

        train_loss += loss.numpy()
        train_acc += correct.numpy()

      tf.keras.backend.set_learning_phase(0)
      for (x, y) in test_set:
        loss, correct = model(x, y)
        test_loss += loss.numpy()
        test_acc += correct.numpy()

      print('epoch:', epoch+1, 'lr:', lr_schedule(epoch+1),
      'train loss:', train_loss / len_train, 'train acc:', train_acc / len_train,
      'val loss:', test_loss / len_test, 'val acc:', test_acc / len_test, 'time:', time.time() - t)
