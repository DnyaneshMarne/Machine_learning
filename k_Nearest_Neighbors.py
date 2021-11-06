# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 16:05:23 2021

@author: Dnyanesh
"""


import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import math as mt


recordsAndAttr=list()
finalErrorMatrix = list()
errMat = {}
totalErrMat = {}
showErrMatrix=[["Fold index","k=1","k=3","k=5","k=7","k=9","k=11","k=13","k=15","k=17","k=19","k=21","k=23"]]


    
def getData():
    filename = input("Enter name of input file for training: ")
    inputFile = open(filename, "r")
    filenameTest = input("Enter name of input file for training: ")
    testingFile = open(filenameTest, "r")
    firstLine = inputFile.readline()
    lineText = firstLine.strip()
    recordsAndAttr = lineText.split("\t")
    dataSet = np.ones((int(recordsAndAttr[0]), int(recordsAndAttr[1])+1), dtype=np.float)
    firstLineTest = testingFile.readline()
    lineTextTest = firstLineTest.strip()
    recordsAndAttrTest = lineTextTest.split("\t")
    dataSetTest = np.ones((int(recordsAndAttrTest[0]), int(recordsAndAttrTest[1])+1), dtype=np.float)
    
    xplot1=[]
    yplot1=[]
    xplot2=[]
    yplot2=[]
    xplot1T=[]
    yplot1T=[]
    xplot2T=[]
    yplot2T=[]
    for i in range(0,int(recordsAndAttr[0]),1):
        actualDataText = inputFile.readline()
        data = actualDataText.split("\t")
        for j in range(int(recordsAndAttr[1])+1):
            dataSet[i,j]=float(data[j])
            if(j==0):
                if(float(data[j+2]) == 0.0):
                    xplot1.append(float(data[j]))
                else:
                    xplot2.append(float(data[j]))
            elif(j ==1):
                if(float(data[j+1]) == 0.0):
                    yplot1.append(float(data[j]))
                else:
                    yplot2.append(float(data[j]))
            
    for i in range(0,int(recordsAndAttrTest[0]),1):
        actualDataTest = testingFile.readline()
        dataTest = actualDataTest.split("\t")
        for j in range(int(recordsAndAttrTest[1])+1):
            dataSetTest[i,j]=float(dataTest[j])
            if(j==0):
                if(float(dataTest[j+2]) == 0.0):
                    xplot1T.append(float(dataTest[j]))
                else:
                    xplot2T.append(float(dataTest[j]))
            elif(j==1):
                if(float(dataTest[j+1]) == 0.0):
                    yplot1T.append(float(dataTest[j]))
                else:
                    yplot2T.append(float(dataTest[j]))
            
    fig, ax = plt.subplots()
    ax.scatter(xplot1, yplot1, color = "red", marker = "o", label = "class 0")
    ax.scatter(xplot2, yplot2, color = "green", marker = "o", label = "class 1")
    plt.title("Training QC capacitor data")
    plt.xlabel("x feature")
    plt.ylabel("y feature")
    plt.legend()
    plt.show()
    fig, ax1 = plt.subplots()
    ax1.scatter(xplot1T, yplot1T, color = "red", marker = "o", label = "class 0")
    ax1.scatter(xplot2T, yplot2T, color = "green", marker = "o", label = "class 1")
    plt.title("Testing QC capacitor data")
    plt.xlabel("x feature")
    plt.ylabel("y feature")
    plt.legend()
    plt.show()
    best_k = splitData(dataSet,recordsAndAttr,5,dataSetTest)
    
    print()
    print(tabulate(showErrMatrix))    
    confusionMatrix(dataSet,dataSetTest,best_k)
    
    
def splitData(data,nOfRecords,nOfFolds,testDataFinal):    
    xCord=[]
    yCord=[]
    ind = int(0)
    for i in range(0,5,1):
        
        trainingData = np.delete(data, slice(ind+0, ind+int(int(nOfRecords[0])/nOfFolds)), axis=0)
        testingData = data[ind+0:ind+int(int(nOfRecords[0])/nOfFolds), :]
        
        ind += int(int(nOfRecords[0])/int(nOfFolds))
        
        processFolds(trainingData,testingData,nOfRecords,i)
    for i in range(0,5,1):
       showErrMatrix.append(["Fold "+str(i+1),errMat[i][1],errMat[i][3],errMat[i][5],errMat[i][7],errMat[i][9],errMat[i][11],errMat[i][13],errMat[i][15],errMat[i][17],errMat[i][19],errMat[i][21],errMat[i][23]])       
       if(i == 4):
           showErrMatrix.append(["Total ",totalErrMat[1],totalErrMat[3],totalErrMat[5],totalErrMat[7],totalErrMat[9],totalErrMat[11],totalErrMat[13],totalErrMat[15],totalErrMat[17],totalErrMat[19],totalErrMat[21],totalErrMat[23]])
    min_key = min(totalErrMat, key=totalErrMat.get)
    
    for k in range(1,24,2):
        xCord.append(k)
        yCord.append((1-(totalErrMat[k]/int(nOfRecords[0])))*100)
            
    fig, ax = plt.subplots() 
    ax.scatter(xCord, yCord, color = "blue", marker = "o", label = "blue") 
    plt.plot(xCord, yCord)
    plt.xlabel("k values")
    plt.ylabel("accuracy")
    plt.show() 
    return min_key
         
    
def processFolds(trainData,testData,records,fold):
    for i in range(0,int(int(records[0])/5),1):        
        
        for k in range(1,24,2):
            if(errMat.get(fold) ==None ):
                errMat[fold] ={}
            prediction = classify(trainData, testData[i], k)
            if(testData[i][-1] != prediction):
                if(totalErrMat.get(k)== None):
                    totalErrMat[k]=1
                else:
                    totalErrMat[k]=totalErrMat[k]+1
                if(errMat[fold].get(k) == None):
                    errMat[fold][k]= 1
                else:
                    errMat[fold][k]= errMat[fold][k]+1
                    
            

def findNeighbours(train, test, k):
	dist = list()
	for trainRow in train:
		eucli = calcDistance(test, trainRow)
		dist.append((trainRow, eucli))
	dist.sort(key=lambda tup: tup[1])
	neighbors = list()
	for i in range(k):
		neighbors.append(dist[i][0])
	return neighbors 
            
def classify(train, test_row, num_neighbors):
    neighbors = findNeighbours(train, test_row, num_neighbors)
    output_values = [row[-1] for row in neighbors]
    prediction = max(set(output_values), key=output_values.count)
    return prediction
     

def calcDistance(firstCord,secondCord):
    distance = 0.0
    for i in range(len(secondCord)-1):
        distance += (firstCord[i] - secondCord[i])**2
    return mt.sqrt(distance)      

def confusionMatrix(trainData,testData,k):
    truePositive = 0
    trueNegaitive = 0
    falsePositive = 0
    falseNegaitive = 0
    for i in range(0,len(testData),1):
        prediction = classify(trainData, testData[i], k)
        if(testData[i][-1] != prediction):
            if(prediction == 0): 
                falseNegaitive=falseNegaitive+1
            else:
                falsePositive=falsePositive+1
        elif(testData[i][-1] == prediction):
            if(prediction == 1):
                truePositive=truePositive+1
            else:
                trueNegaitive=trueNegaitive+1 
        
                
    
    accuracy = (truePositive+trueNegaitive)/(truePositive+trueNegaitive+falseNegaitive+falsePositive)
   
    precision = truePositive/(truePositive+falsePositive)
    recall = truePositive/(truePositive+falseNegaitive)
    f1Value = 2*(1/((1/precision)+(1/recall)))
    print()
    print("The confusion matrix is as follows")
    print("FP is : "+str(falsePositive))
    print("FN is : "+str(falseNegaitive))
    print("TP is : "+str(truePositive))
    print("TN is : "+str(trueNegaitive))
    print(tabulate([[trueNegaitive,falsePositive],[falseNegaitive,truePositive]]))
    print("Accuracy is : "+str(round(accuracy*100,2)))    
    print("Precision is : "+str(round(precision*100,2)))
    print("Recall is : "+str(round(recall*100,2)))
    print("F1 is : "+str(round(f1Value*100,2)))
    
getData()
