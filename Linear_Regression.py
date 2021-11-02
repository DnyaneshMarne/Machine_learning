# -*- coding: utf-8 -*-
"""
Created on Fri Sep  3 18:04:47 2021

@author: Dnyanesh
"""


import numpy as np
from tabulate import tabulate

table = [['Ground Truth', 'Predicted Value', 'Difference(Ground Truth - Predicted value)']]
filename = input("Enter name of training file: ")
inputFile = open(filename, "r")
aString = inputFile.readline()
t = aString.strip()
lst = t.split("\t")

X = np.ones((int(lst[0]),int(lst[1])+1),dtype = np.float)
Y = np.ones((int(lst[0]),1),dtype = np.float)
for i in range(0,int(lst[0]),1):
    aString = inputFile.readline()
    t = aString.split("\t")
    for j in range(int(lst[1])+1):
        if j !=int(lst[1]):
            #print(j)
            #print(t[j])
            X[i,j+1] = float(t[j])
        else:
            Y[i,0] = float(t[j])
        
WforTrainingData = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),Y)
JforTrainingData=np.dot(np.dot((1/int(lst[0])),np.dot(X,WforTrainingData)-Y).T,np.dot(X,WforTrainingData)-Y)

testingFile = input("Enter name of test file: ")
inputFile2 = open(testingFile, "r")
aString2 = inputFile2.readline()
t2 = aString2.strip()
lst2 = t2.split("\t")
trainingDataX = np.ones((int(lst2[0]),int(lst2[1])+1),dtype = np.float)
predictionY = np.ones((int(lst2[0]),1),dtype = np.float)
groundTruthY = np.ones((int(lst2[0]),1),dtype = np.float)
for k in range(0,int(lst2[0]),1):
    aString2 = "1\t"+inputFile2.readline()
    t2 = aString2.split("\t")
    prediction = 0
    for l in range(0,int(lst2[1])+1,1):
        trainingDataX[k,l] = float(t2[l])
        prediction += WforTrainingData[l,0]*float(t2[l])
        if l == int(lst2[1]):
            groundTruthY[k,0]=float(t2[l+1])        
    predictionY[k,0]=prediction
            

for z in range(0,int(lst2[0]),1):
    table.append([groundTruthY[z,0],predictionY[z,0],groundTruthY[z,0]-predictionY[z,0]])

JforPredictedValues =np.dot(np.dot((1/int(lst2[0])),np.dot(trainingDataX,WforTrainingData)-predictionY).T,np.dot(trainingDataX,WforTrainingData)-predictionY)    
JforGroundTruthValues =np.dot(np.dot((1/int(lst2[0])),np.dot(trainingDataX,WforTrainingData)-groundTruthY).T,np.dot(trainingDataX,WforTrainingData)-groundTruthY)    

print("W from training data :"+ str(WforTrainingData))
print("J for training data : "+ str(JforTrainingData[0,0]))  
print("J for ground truth data : "+ str(JforGroundTruthValues[0,0]))   
print(tabulate(table)) 
inputFile.close()
inputFile2.close()
