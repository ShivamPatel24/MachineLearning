#Dataset Import
import csv
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import seaborn as sns

data = pd.read_csv("heart.csv")
print(data)

print("--------------------------------")

#Check for missing values
print(data.columns[data.isnull().any()])


print("--------------------------------")

#Checking for number of class 0s
totalZeros = data['chd'].value_counts()
print(totalZeros)


print("--------------------------------")

#One Hot Encoding
#One Hot Encoding
#One column is removed because only a single column is needed to encode the family history column where it is either present or absent
encodedData = pd.get_dummies(data['famhist'], prefix = 'famhist')
data = data.drop('famhist', axis=1)
DataFrameEncoded = pd.concat([data[['row.names','sbp','tobacco','ldl','adiposity']], encodedData['famhist_Present'], data[['typea','obesity','alcohol','age','chd']]], axis=1)
data = DataFrameEncoded
print(data)

print("--------------------------------")


#Standardization
#Standardization
dataToStandardize = data[['sbp','tobacco','ldl','adiposity','typea','obesity','alcohol','age']]
standard = (dataToStandardize-dataToStandardize.mean())/dataToStandardize.std()
dataStandard = pd.concat([data['row.names'],standard[['sbp','tobacco','ldl','adiposity']], encodedData['famhist_Present'], standard[['typea','obesity','alcohol','age']],data['chd']], axis=1)
data = dataStandard
print(data)

print("--------------------------------")

#Finding maximum correlation
fig = plt.figure(figsize=(12,10))
corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True, cmap=plt.cm.Blues)

#Implementation of Machine Learning Algorithm
#Implented in function allowing us to choose batch size (batch or mini-batch)

def cost(data,theta):
    m = data.shape[0]
    dataCopy = data #Ensure we are not mutating original data frame
    dataCopy = dataCopy.drop('row.names', axis=1) #We do not need row numbers
    dataMatrix = dataCopy.to_numpy()
    jFunc = 0
    
    for sample in range(462):
        #Find z for hypothesis 
        zFunc = (theta[0]*dataMatrix[sample,0] + theta[1]*dataMatrix[sample,1] + theta[2]*dataMatrix[sample,2] +
                theta[3]*dataMatrix[sample,3] + theta[4]*dataMatrix[sample,4] + theta[5]*dataMatrix[sample,5] +
                theta[6]*dataMatrix[sample,6] + theta[7]*dataMatrix[sample,7] + theta[8]*dataMatrix[sample,8] + theta[9])
        #Calculate hypothesis for this sample
        hFunc = 1/(1+np.exp(-(zFunc)))
        #Calculate cost function through summing error for this sample + error from all previous samples
        jFunc = jFunc + (dataMatrix[sample,9]*np.log(hFunc) + (1-dataMatrix[sample,9])*np.log(1-hFunc))
     
    jFunc = jFunc/(-462) #Cost function is calcualted over the full epoch thus divided by 462
    return jFunc


def logReg(data, alpha, epochNum, miniBatch):
    #Declare Variables
    sampleNum = data.shape[0]
    dataCopy = data #Ensure we are not mutating original data frame
    dataCopy = dataCopy.drop('row.names', axis=1) #We do not need row numbers
    dataMatrix = dataCopy.to_numpy()
    theta = [0,0,0,0,0,0,0,0,0,0]
    thetaChange = [0,0,0,0,0,0,0,0,0,0] #used to calculate the summation part of gradient descent
    jFunc = [0]*epochNum #Initlize an array of size N for cost function (to be graphed over epochs)

    #Initlize thetas to random number between 0-1
    for iterations in range(10):
        theta[iterations] = np.random.random_sample()
    
    #Initlize mini-batch or batch depending on batch size
    if(miniBatch == 0):
        allSamples = [0,sampleNum]
    else:
        allSamples = [0,50,100,150,200,250,300,350,400,450,462] #Hard coded because mini-batch will always be a size of 50
        
    #Running Iterations 
    for epoch in range(epochNum):
        print("Epoch Number:" + str(epoch))
        jFunc[epoch] = cost(data,theta)
        for arrNum in range(len(allSamples)-1): #This for loop does nothing unless function is set for mini-batch
            thetaChange = [0,0,0,0,0,0,0,0,0,0] #Initlize thetaChange back to 0 for new iteration
            for sample in range(allSamples[arrNum],allSamples[arrNum+1]):
                sampleNum = allSamples[arrNum+1]-allSamples[arrNum]
                #Find z for hypothesis 
                zFunc = (theta[0]*dataMatrix[sample,0] + theta[1]*dataMatrix[sample,1] + theta[2]*dataMatrix[sample,2] +
                    theta[3]*dataMatrix[sample,3] + theta[4]*dataMatrix[sample,4] + theta[5]*dataMatrix[sample,5] +
                    theta[6]*dataMatrix[sample,6] + theta[7]*dataMatrix[sample,7] + theta[8]*dataMatrix[sample,8] + theta[9])
                #Calculate hypothesis for this sample
                hFunc = 1/(1+np.exp(-(zFunc)))
              
                #Calculate theta 0-9 which has changing Xjs'
                for thetaNum in range(9):
                    thetaChange[thetaNum] = thetaChange[thetaNum] + (dataMatrix[sample,9] - hFunc)*dataMatrix[sample,thetaNum]
                #Calculate theta 1 which has an X0 of 1
                thetaChange[9] = (thetaChange[9] + (dataMatrix[sample,9] - hFunc)*1)

            #Batch Gradietn Descent Iteration Over, Update Thetas 
            for thetaNum in range(10):
                theta[thetaNum] = theta[thetaNum] + thetaChange[thetaNum]*alpha*(1/sampleNum)
                
    #Plot graph of cost function vs epochs
    plt.figure()
    plt.title("Cost Function vs. Epochs -- Alpha = " + str(alpha) + " -- Mini-batch = " + str(miniBatch))
    plt.xlabel("Epoch")
    plt.ylabel("Cost Function")
    plt.plot(jFunc)
    print(theta)
    
    return


logReg(data, 0.001, 10000, 0) #alpha = 0.001 -- epochs = 10000 -- Batch

logReg(data, 0.001, 10000, 1) #alpha = 0.001 -- epochs = 10000 -- Mini-batch


#--------------------------------------------TENSORFLOW IMPLEMENTATION-------------------------------------------------------
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def logRegTensor(data, alpha, epochNum, miniBatch):
  dataCopy = data #Ensure we are not mutating original data frame
  dataCopy = dataCopy.drop('row.names', axis=1) #We do not need row numbers
  dataMatrix = dataCopy.to_numpy()

  xFeatures = dataMatrix[:,(0,1,2,3,4,5,6,7,8)]
  yTarget = dataMatrix[:,9]
  yTarget = yTarget.reshape(462,1)

  X = tf.placeholder(tf.float32, [None, 9])
  Y = tf.placeholder(tf.float32, [None, 1])

  W = tf.Variable(np.random.randn(9,1).astype(np.float32))
  B = tf.Variable(np.random.randn(1).astype(np.float32))

  hypTensor = tf.math.sigmoid(tf.add(tf.matmul(X, W),B))
  costTensor = tf.reduce_mean(-tf.reduce_sum(Y*tf.log(hypTensor) + (1-Y)*tf.log(1-hypTensor),axis=1))
  optimizerTensor = tf.train.GradientDescentOptimizer(alpha).minimize(costTensor)

  jFuncTensor = [0]*epochNum

  #Initlize mini-batch or batch depending on batch size
  if(miniBatch == 0):
      allSamples = [0,462]
  else:
      allSamples = [0,50,100,150,200,250,300,350,400,450,462] #Hard coded because mini-batch will always be a size of 50

  with tf.Session() as sesh:
    sesh.run(tf.global_variables_initializer())
    for epoch in range(epochNum):
      for arrNum in range(len(allSamples)-1): #This for loop does nothing unless function is set for mini-batch
        xSub = xFeatures[allSamples[arrNum]:allSamples[arrNum+1]]
        ySub = yTarget[allSamples[arrNum]:allSamples[arrNum+1]]
        sesh.run(optimizerTensor, feed_dict={X: xSub, Y: ySub})
      jFuncTensor[epoch] = sesh.run(costTensor, feed_dict={X: xFeatures, Y: yTarget})
    #Copy final theta values for output
    thetasTensor = sesh.run(W)
    thetaZeroTensor = sesh.run(B)

  plt.figure()
  plt.title("Cost Function vs. Epochs -- Alpha = " + str(alpha) +  " -- Mini-batch = " + str(miniBatch))
  plt.xlabel("Epoch")
  plt.ylabel("Cost Function")
  plt.plot(jFuncTensor)
  print(thetasTensor.reshape(1,9))
  print(thetaZeroTensor)

  return


logRegTensor(data, 0.001, 10000, 0) #alpha = 0.001 -- epochs = 10000 -- Batch

logRegTensor(data, 0.001, 10000, 1) #alpha = 0.001 -- epochs = 10000 -- Mini-batch

plt.show()
   
print("Finished Execution")