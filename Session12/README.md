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
    
3 Define a function to initialize layer weights. This function has been defined here because DavidNet model uses PyTorch but the way PyTorch intializes model layer weights has no equivalent in tensorflow.

    def init_pytorch(shape, dtype=tf.float32, partition_info=None):
         fan = np.prod(shape[:-1])
         bound = 1 / math.sqrt(fan)
         return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)
         
4 Define a class for basic Conv2D layer followed BatchNormalization and ReLU activation.  This class has been inherited from "tf.keras.Model" class thus it inherits training and inference functionalities. Instances of this class can be used as parts or building blocks of our DNN model. The class groups number of layers into a single class object. These layers are defined in __\_\_init\_\_()__  and how these layers are connected i.e. model forward pass is defined in  __call()__ 

    class ConvBN(tf.keras.Model):
      def __init__(self, c_out):
        super().__init__()
        self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
        self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
        self.drop = tf.keras.layers.Dropout(0.05)

      def call(self, inputs):
        return tf.nn.relu(self.bn(self.drop(self.conv(inputs))))
        
5

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

6

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
        ** h = self.linear(h) * self.weight
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
        loss = tf.reduce_sum(ce)
        correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
        return loss, correct
    
7

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
    
8

    model = DavidNet()
    batches_per_epoch = len_train//BATCH_SIZE + 1

    ** lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
    global_step = tf.train.get_or_create_global_step()
    lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/BATCH_SIZE
    ** opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)
    data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)

9  explain :tensor_slices, prefetch, "for g, v in zip(grads, var):
          g += v * WEIGHT_DECAY * BATCH_SIZE
        opt.apply_gradients(zip(grads, var), global_step=global_step)"
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
