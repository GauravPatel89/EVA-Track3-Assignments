## Input I1

## L1

Sep 5x5  

I1

## L2 

Conv 5x5 

L1

## L3

Conv 5x5   

L2

## L4  

Sep 5x5

L1 + L3

## L5

MaxPool

L1 + L4

## L6      

Sep 3x3

L5

## L7

Conv 5x5

L1 + L4 + L6

## L8 

Sep 3x3     

L3 + L4 + L6 + L7

## L9

Sep 5x5     

L1 + L3 + L4 + L6 + L7 + L8

## L10

MaxPool     

L1 + L4 + L6 + L8 + L9

## L11

Conv 5x5  

L7 + L10

## L12

Sep 5x5 

L2 + L4 + L8 + L11

## L13

Conv 3x3   

L2 + L3 + L6 + L11 + L12

## L14

Sep 5x5     

L1 + L3 + L4 + L6 + L8 + L12 + L13

## 0utPut 

O1 = L4 + L8 + L12 + L14





