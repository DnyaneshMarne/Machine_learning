# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 22:30:55 2021

@author: Dnyanesh
"""


import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt


def getData():
    filename = input("Enter name of input file for training: ")
    inputFile = open(filename, "r")
    firstLine = inputFile.readline()
    lineText = firstLine.strip()
    recordsAndAttr = lineText.split("\t")
    dataSet = np.ones((int(recordsAndAttr[0]), int(recordsAndAttr[1])+1), dtype=np.float)
    dataSetY = np.ones((int(recordsAndAttr[0]), 1), dtype=np.float)
    
    
    
    for i in range(0,int(recordsAndAttr[0]),1):
        actualDataText = inputFile.readline()
        data = actualDataText.split("\t")
        for j in range(int(recordsAndAttr[1])):
            dataSet[i,j+1]=float(data[j])
            if(j==int(recordsAndAttr[1])-1):
                dataSetY[i,0] = float(data[j+1])  
            
    
    
    W=calcHypthesis (dataSet,dataSetY,recordsAndAttr)
    filenameTest = input("Enter name of input file for training: ")
    testingFile = open(filenameTest, "r")
    firstLineTest = testingFile.readline()
    lineTextTest = firstLineTest.strip()
    recordsAndAttrTest = lineTextTest.split("\t")
    dataSetTest = np.ones((int(recordsAndAttrTest[0]), int(recordsAndAttrTest[1])+1), dtype=np.float)
    dataSetTestY = np.ones((int(recordsAndAttrTest[0]), 1), dtype=np.float)
    for i in range(0,int(recordsAndAttrTest[0]),1):
        actualDataTest = testingFile.readline()
        dataTest = actualDataTest.split("\t")
        for j in range(int(recordsAndAttrTest[1])):
            dataSetTest[i,j+1]=float(dataTest[j])
            if(j==int(recordsAndAttrTest[1])-1):
                dataSetTestY[i,0] = float(dataTest[j+1])
    classifyTestData(W,dataSetTest,dataSetTestY,recordsAndAttrTest)
    inputFile.close()
    testingFile.close()
            
def calcHypthesis(X,Y,sizeOfArr):
    print()
    iterations = 200000
    learningRate = 0.08
    print('Iterations : '+str(iterations))
    print('learning Rate : '+str(learningRate))
    plotX=[]
    plotY=[]
    W=np.zeros((1,int(sizeOfArr[1])+1),dtype=np.int)
    print('Initial weights : '+str(W))
    hW = 1 / ( 1 + np.exp( - ( X.dot( W.T ) ) ) )        
        #j=0.0
    j=(-1/int(sizeOfArr[0]))*(Y.T.dot(np.log(hW))+((1-Y.T).dot(np.log(1-hW))))
    print('Initial J value : '+str(j))
    for i in range(iterations):
        hW = 1 / ( 1 + np.exp( - ( X.dot( W.T ) ) ) )        
        #j=0.0
        j=(-1/int(sizeOfArr[0]))*(Y.T.dot(np.log(hW))+((1-Y.T).dot(np.log(1-hW))))
        plotX.append(i)
        plotY.append(j)
        tmp = ( hW - Y )
        dW = np.dot( X.T, tmp ) / int(sizeOfArr[0])
        W = W - (learningRate * dW.T)
        
    print('J value on training data after iterations : '+str(j[0][0]))
    print('W value after iterations : '+str(W))
    fig, ax = plt.subplots() 
    ax.scatter(plotX, plotY, color = "blue", marker = "o", label = "blue") 
    plt.xlabel("iterations")
    plt.ylabel("j")
    plt.show()
    return W

def classifyTestData(W,testData,dataSetTestY,recN):
    truePositive = 0
    trueNegaitive = 0
    falsePositive = 0
    falseNegaitive = 0
    classf=0.0
    hW = 1 / ( 1 + np.exp( - ( testData.dot( W.T ) ) ) )        
    j=0.0
    j=(-1/int(recN[0]))*(dataSetTestY.T.dot(np.log(hW))+((1-dataSetTestY.T).dot(np.log(1-hW))))
    print('J for test data :'+str(j[0][0]))
    for i in range(0,int(recN[0]),1):
        #print(dataSetTestY)
        hW = 1 / ( 1 + np.exp( - ( testData[i].dot( W.T ) ) ) )
        if (hW >= 0.5):
            classf = 1.0
        else:
            classf = 0.0
        if(classf == 2.0):
            print(hW)   
            print(classf) 
            print(dataSetTestY[i][0])
        if(dataSetTestY[i][0] != classf):
            if(classf == 0.0): 
                falseNegaitive=falseNegaitive+1
            else:
                falsePositive=falsePositive+1
        elif(dataSetTestY[i][0] == classf):
            if(classf == 1.0):
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