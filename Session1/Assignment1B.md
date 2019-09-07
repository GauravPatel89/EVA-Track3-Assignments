# EVA-Track3-Poject1

## Assignment 1B
### Q.1 What are Channels and Kernels (according to EVA)?
-Channels are containers of similar type of features and kernels are the feature extractors which
will extract specific features from the input and store them into channels. Say in case of some song. kernel can be said to be some kind of technology by which we can take only say guitar track or drum track of the song. The entire track itself of say guitar or drum will be called channel. 

In terms of images a filter which can take out simply Red (or green or blue or any particular color) part of the colored image can be called kernel while the output image containing only the extracted component of the image can e called a channel. Also an image can be divided into any number of channels eg. if we talk about color we can have 1 channel say brightness or we can have 256k channels representing as many colors. 

Similarly we can extract various kinds of features from the input image. Collection of similar features can be called a channel e.g. image containing only vertical edges. The filter which can extract these features is known as a kernel.

### Q.2 Why should we only (well mostly) use 3x3 Kernels?
-This is because of following reasons
1. This is smallest possible kernel with odd number of rows and columns thus providing axis of symmetry. Using this kernel we can obtain similar result to that obtained by larger kernels with less number of parameters.

eg.5x5 i/p     | 5x5 kernel convolution  | 1x1 o/p     (5x5 = 25 parameters) 
  
   5x5 i/p     | 3x3 kernel convolution   | 3x3 o/p   |  3x3 kernel convolution  | 1x1 o/p    (3x3+3x3= 18 parameters)
    
Thus we are reducing the number of parameters. In this case (25-18=7) the saving is not much but with larger kernels, savings will be significant eg.in next question if had used just 1 kernel of 199x199, number of parameters would have been 39601  but we see the entire image by applying 3x3, 99 times,thus number of parameters used are 891 means saving of 38710 paras. This is for just 1 channel, we will have large number of channels hence the savings will be manyfold. 
  
2. Modern day GPUs like nVidia are providing optimization for 3x3 kernels thus by breaking down larger kernel ( >3x3) operations into combinations of 3x3 operations hence we will save significant calculation time. 

### Q.3 How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

We need 99 layers. Calculations shown below.
        
    Layer No.   |    i/p    |   Operation   |    o/p
         1      |  199x199  |   3x3 Conv    |  197x197 
         2      |  197x197  |   3x3 Conv    |  195x195     
         3      |  195x195  |   3x3 Conv    |  193x193
         4      |  193x193  |   3x3 Conv    |  191x191
         5      |  191x191  |   3x3 Conv    |  189x189
         6      |  189x189  |   3x3 Conv    |  187x187
         7      |  187x187  |   3x3 Conv    |  185x185
         8      |  185x185  |   3x3 Conv    |  183x183
         9      |  183x183  |   3x3 Conv    |  181x181
         10     |  181x181  |   3x3 Conv    |  179x179
         11     |  179x179  |   3x3 Conv    |  177x177
         12     |  177x177  |   3x3 Conv    |  175x175
         13     |  175x175  |   3x3 Conv    |  173x173
         14     |  173x173  |   3x3 Conv    |  171x171
         15     |  171x171  |   3x3 Conv    |  169x169
         16     |  169x169  |   3x3 Conv    |  167x167
         17     |  167x167  |   3x3 Conv    |  165x165
         18     |  165x165  |   3x3 Conv    |  163x163
         19     |  163x163  |   3x3 Conv    |  161x161
         20     |  161x161  |   3x3 Conv    |  159x159
         21     |  159x159  |   3x3 Conv    |  157x157
         22     |  157x157  |   3x3 Conv    |  155x155
         23     |  155x155  |   3x3 Conv    |  153x153
         24     |  153x153  |   3x3 Conv    |  151x151
         25     |  151x151  |   3x3 Conv    |  149x149
         26     |  149x149  |   3x3 Conv    |  147x147
         27     |  147x147  |   3x3 Conv    |  145x145
         28     |  145x145  |   3x3 Conv    |  143x143
         29     |  143x143  |   3x3 Conv    |  141x141
         30     |  141x141  |   3x3 Conv    |  139x139
         31     |  139x139  |   3x3 Conv    |  137x137
         32     |  137x137  |   3x3 Conv    |  135x135
         33     |  135x135  |   3x3 Conv    |  133x133
         34     |  133x133  |   3x3 Conv    |  131x131
         35     |  131x131  |   3x3 Conv    |  129x129
         36     |  129x129  |   3x3 Conv    |  127x127
         37     |  127x127  |   3x3 Conv    |  125x125
         38     |  125x125  |   3x3 Conv    |  123x123
         39     |  123x123  |   3x3 Conv    |  121x121
         40     |  121x121  |   3x3 Conv    |  119x119
         41     |  119x119  |   3x3 Conv    |  117x117
         42     |  117x117  |   3x3 Conv    |  115x115
         43     |  115x115  |   3x3 Conv    |  113x113
         44     |  113x113  |   3x3 Conv    |  111x111
         45     |  111x111  |   3x3 Conv    |  109x109
         46     |  109x109  |   3x3 Conv    |  107x107
         47     |  107x107  |   3x3 Conv    |  105x105
         48     |  105x105  |   3x3 Conv    |  103x103
         49     |  103x103  |   3x3 Conv    |  101x101
         50     |  101x101  |   3x3 Conv    |  99 x99 
         51     |  99 x99   |   3x3 Conv    |  97 x97 
         52     |  97 x97   |   3x3 Conv    |  95 x95 
         53     |  95 x95   |   3x3 Conv    |  93 x93 
         54     |  93 x93   |   3x3 Conv    |  91 x91 
         55     |  91 x91   |   3x3 Conv    |  89 x89 
         56     |  89 x89   |   3x3 Conv    |  87 x87 
         57     |  87 x87   |   3x3 Conv    |  85 x85 
         58     |  85 x85   |   3x3 Conv    |  83 x83 
         59     |  83 x83   |   3x3 Conv    |  81 x81 
         60     |  81 x81   |   3x3 Conv    |  79 x79 
         61     |  79 x79   |   3x3 Conv    |  77 x77 
         62     |  77 x77   |   3x3 Conv    |  75 x75 
         63     |  75 x75   |   3x3 Conv    |  73 x73 
         64     |  73 x73   |   3x3 Conv    |  71 x71 
         65     |  71 x71   |   3x3 Conv    |  69 x69 
         66     |  69 x69   |   3x3 Conv    |  67 x67 
         67     |  67 x67   |   3x3 Conv    |  65 x65 
         68     |  65 x65   |   3x3 Conv    |  63 x63 
         69     |  63 x63   |   3x3 Conv    |  61 x61 
         70     |  61 x61   |   3x3 Conv    |  59 x59 
         71     |  59 x59   |   3x3 Conv    |  57 x57 
         72     |  57 x57   |   3x3 Conv    |  55 x55 
         73     |  55 x55   |   3x3 Conv    |  53 x53 
         74     |  53 x53   |   3x3 Conv    |  51 x51 
         75     |  51 x51   |   3x3 Conv    |  49 x49 
         76     |  49 x49   |   3x3 Conv    |  47 x47 
         77     |  47 x47   |   3x3 Conv    |  45 x45 
         78     |  45 x45   |   3x3 Conv    |  43 x43 
         79     |  43 x43   |   3x3 Conv    |  41 x41 
         80     |  41 x41   |   3x3 Conv    |  39 x39 
         81     |  39 x39   |   3x3 Conv    |  37 x37 
         82     |  37 x37   |   3x3 Conv    |  35 x35 
         83     |  35 x35   |   3x3 Conv    |  33 x33 
         84     |  33 x33   |   3x3 Conv    |  31 x31 
         85     |  31 x31   |   3x3 Conv    |  29 x29 
         86     |  29 x29   |   3x3 Conv    |  27 x27 
         87     |  27 x27   |   3x3 Conv    |  25 x25 
         88     |  25 x25   |   3x3 Conv    |  23 x23 
         89     |  23 x23   |   3x3 Conv    |  21 x21 
         90     |  21 x21   |   3x3 Conv    |  19 x19 
         91     |  19 x19   |   3x3 Conv    |  17 x17 
         92     |  17 x17   |   3x3 Conv    |  15 x15 
         93     |  15 x15   |   3x3 Conv    |  13 x13 
         94     |  13 x13   |   3x3 Conv    |  11 x11 
         95     |  11 x11   |   3x3 Conv    |   9 x 9 
         96     |   9 x 9   |   3x3 Conv    |   7 x 7 
         97     |   7 x 7   |   3x3 Conv    |   5 x 5 
         98     |   5 x 5   |   3x3 Conv    |   3 x 3 
         99     |   3 x 3   |   3x3 Conv    |   1 x 1
(*generated using my code (https://github.com/GauravPatel89/EVA-Track3-Poject1/blob/master/LayerCalculationsPrinting.ipynb) :D I totally understand the need of maxPooling layer.)
