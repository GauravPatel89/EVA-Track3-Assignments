## Receptive Field calcuations for GoogLeNet network

j = jump

s = stride

k = kernel size

n<sub>Out</sub> = output size

n<sub>In</sub> = Input size

r<sub>Out</sub> = Receptive field Out

r<sub>In</sub> = Receptive field In

**n<sub>Out</sub> = Floor((n<sub>In</sub> + 2p - k)/s) + 1**

**j<sub>Out</sub> = j<sub>In</sub> x s**

**r<sub>Out</sub> = r<sub>In</sub> + (k-1) x j<sub>In</sub>**

-----
**Layer1**

k,p,s = 3,2,1

r<sub>In</sub> = 1

j<sub>In</sub> = 1

j<sub>Out</sub> = 1 x 1 = 1

r<sub>Out</sub> = 1 + (3-1) x 1 = 3


-----
**Layer2**

k,p,s = 3,1,2

r<sub>In</sub> = 3

j<sub>In</sub> = 1

j<sub>Out</sub> = 1 x 2 = 2

r<sub>Out</sub> = 3 + (3-1) x 1 = 5


-----
**Layer3**

k,p,s = 5,4,1

r<sub>In</sub> = 5

j<sub>In</sub> = 2

j<sub>Out</sub> = 2 * 1 = 2

r<sub>Out</sub> = 5 + (5-1) x 2 = 13


-----
**Layer4**

k,p,s = 5,4,1

r<sub>In</sub> = 13

j<sub>In</sub> = 2

j<sub>Out</sub> = 2 * 1 = 2

r<sub>Out</sub> = 13 + (5-1) x 2 = 21

-----
**Layer5**

k,p,s = 3,1,2

r<sub>In</sub> = 21

j<sub>In</sub> = 2

j<sub>Out</sub> = 2 x 2 = 4

r<sub>Out</sub> = 21 + (3-1) x 2 = 25

-----
**Layer6**

k,p,s = 5,4,1

r<sub>In</sub> = 25

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 25 + (5-1) x 4 = 41

-----
**Layer7**

k,p,s = 5,4,1

r<sub>In</sub> = 41

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 41 + (5-1) x 4 = 57

-----
**Layer8**

k,p,s = 5,4,1

r<sub>In</sub> = 57

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 57 + (5-1) x 4 = 73

-----
**Layer9**

k,p,s = 5,4,1

r<sub>In</sub> = 73

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 73 + (5-1) x 4 = 89

-----
**Layer10**

k,p,s = 5,4,1

r<sub>In</sub> = 89

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 * 1 = 4

r<sub>Out</sub> = 89 + (5-1) x 4 = 105

-----
**Layer11**

k,p,s = 3,1,2

r<sub>In</sub> = 105

j<sub>In</sub> = 4

j<sub>Out</sub> = 4 x 2 = 8

r<sub>Out</sub> = 105 + (3-1) x 4 = 113

-----
**Layer12**

k,p,s = 5,4,1

r<sub>In</sub> = 113

j<sub>In</sub> = 8

j<sub>Out</sub> = 8 * 1 = 8

r<sub>Out</sub> = 113 + (5-1) x 8 = 145

-----
**Layer13**

k,p,s = 5,4,1

r<sub>In</sub> = 145

j<sub>In</sub> = 8

j<sub>Out</sub> = 8 * 1 = 8

r<sub>Out</sub> = 145 + (5-1) x 8 = 177

-----
**Layer14**

k,p,s = 7,0,1

r<sub>In</sub> = 177

j<sub>In</sub> = 8

j<sub>Out</sub> = 8 * 1 = 8

r<sub>Out</sub> = 177 + (7-1) x 8 = 225
