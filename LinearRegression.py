#Dataset Import
import csv
import pandas as pd
dataframe = pd.read_csv("student_marks.csv")

#Data prepration and EDA
mean = dataframe.mean()

stdev = dataframe.std()

#Standardization 
dataStand = (dataframe-mean)/stdev

#Initial Linear Regression
import matplotlib.pyplot as plt
import numpy as np

dataX = dataStand['Midterm mark']
dataY = dataStand['Final mark']

#Initial regression line of m=-0.5, b=0
lineX = np.linspace(min(dataX),max(dataX),100)
lineY = (-0.5)*lineX+0
plt.plot(lineX,lineY)

#scatter plot of standardized data
plt.title("Standardized Data Scatter Plot")
plt.xlabel("Midterm Mark")
plt.ylabel("Final Mark")
plt.scatter(dataX,dataY)


#Implementation

def linReg(slope, intercept, alpha, errorFunc, dataX, dataY, iterationNum):
    N = dataX.shape[0]
    slopeOld = 0 
    interceptOld = 0 
    changeSlope = 0
    changeIntercept = 0

    index = 0
    iterations = 0

    #Running Iterations 
    for iterations in range(iterationNum):
        for index in range(N):
            errorFunc[iterations] = errorFunc[iterations]  + np.square((dataY[index] - (slope*dataX[index] + intercept)))    
            changeSlope = changeSlope - dataX[index]*(dataY[index] - (slope*dataX[index] + intercept))
            changeIntercept = changeIntercept - 1*(dataY[index] - (slope*dataX[index] + intercept))

        errorFunc[iterations] = errorFunc[iterations]/N
        changeSlope = changeSlope*(2/N)
        changeIntercept = changeIntercept*(2/N)

        slopeOld = slope
        interceptOld = intercept

        slope = slopeOld - alpha*changeSlope
        intercept = interceptOld - alpha*changeIntercept


    #Plotting regression line
    lineX100 = np.linspace(min(dataX),max(dataX),100)
    lineY100 = slope*lineX100+intercept
    plt.plot(lineX100,lineY100)

    #scatter plot of standardized data
    plt.title("Regression Line at " + str(iterationNum) + " Iterations")
    plt.xlabel("Midterm Mark")
    plt.ylabel("Final Mark")
    plt.scatter(dataX,dataY)    
    plt.show()
    return;


#2000 Iterations (Standardized)
slope = -0.5
intercept = 0
alpha = 0.0001
errorFunc = [0]*2000
iterationNum = 2000
linReg(slope, intercept, alpha, errorFunc, dataX, dataY, iterationNum)

#Error plot for 2000 iterations (Standardized)
plt.plot(errorFunc)
plt.title("Error Over 2000 Iterations")
plt.xlabel("Iterations")
plt.ylabel("Error")
