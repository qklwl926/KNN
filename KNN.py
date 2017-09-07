# -*- coding: utf-8 -*-

"""
Created on Mon Sep 04 18:48:06 2017

@author: WL
"""
'''ukNN,
(1)计算已知类别数据集中的点与当前点的距离；
(2)按照距离递增排序
(3)选取与当前点距离最小的K个点
(4)确定前K个点所在类别的出现频率
(5)返回前K个点出现频率最高的类别作为当前点的预测分类 
'''
from numpy import *
import operator
from os import listdir
import matplotlib
import matplotlib.pyplot as plt





def creatDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['a','a','b','b']
    return group,labels
    

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]    #.shape[0]行数看第一个维度
    diffMat = tile(inX, (dataSetSize,1)) - dataSet#tile（A,reps） reps重复的次数，列方向dataSetSize次，行方向1次
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)#对于二维数组axis=1表示按行相加 , axis=0表示按列相加
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()#argsort(a, axis=-1（行列改变）, kind='quicksort', order=None)
                                            #Returns the indices that would sort an array.排序返回序列号
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #dict.get(key, default=None)
                                                                #查找字典如果不存在返回默认
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
         #sorted(itrearble, cmp=None, key=None, reverse=False)
        #=号后面是默认值 默认是升序排序的， 如果想让结果降序排列，用reverse=True
        #interitems 迭代器函数，sort()与sorted()的不同在于，sort是在原位重新排列列表，而sorted()是产生一个新的列表。
        #返回值：是一个经过排序的可迭代类型，与iterable一样。
        #itemgetter(1)按照第二个排序
    return sortedClassCount[0][0]

def file2matrix(filename):
    #对文件操作
    fr=open(filename)
    arrayOLines=fr.readlines()
    munberOfLines=len(arrayOLines)
    #创建返回的矩阵
    returnMat=zeros((munberOfLines,3))
    classLabelVector=[]
  
    index=0
    for line in arrayOLines:
        line=line.strip()#str.strip([chars]);Python strip() 方法用于移除字符串头尾指定的字符（默认为空格）。当rm空时,默认删除空白符(包括'\n','\r','\t',' ')
        #首先是使用函数strip()截取所有回车字符，然后使用tab字符\t将上一步得到的整行数据分割成一个元素列表。
        listFromLine=line.split("\t")#将数据前三列提取出来,存放到returnMat的NumPy矩阵中,也就是特征矩阵
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector.append(int(listFromLine[-1]))#这里是把最后一列加到classlabelvector
        #int() python 内部函数 将一个数字或base类型的字符串转换成整数。
        index+=1
    return returnMat,classLabelVector


group,labels=creatDataSet()

print(classify0([0,0],group,labels,3))
Mat,ClassLabel=file2matrix("datingTestSet2.txt")#注意加""

def autoNorm(dataSet):
    minVals=dataSet.min(0)    #归一化首先找到每一列也就是每种信息的最小值
    maxVals=dataSet.max(0)    #找到最大值
    ranges=maxVals-minVals    #两者差
    normDataSet=zeros(shape(dataSet)) #生成返回的矩阵
    m=dataSet.shape[0]               #找到原先数组的行数
    normDataSet=dataSet-tile(minVals,(m,1))    #采用tile对最小值进行复制
    normDataSet=normDataSet/tile(ranges,(m,1))   
    return normDataSet,ranges,minVals

fig=plt.figure()

ax=fig.add_subplot(111)
ax.scatter(Mat[:,0],Mat[:,1],15.0*array(ClassLabel),15.0*array(ClassLabel))
#前两个参数为x，y坐标s表示点点的大小，c就是color嘛，marker就是点的形状
normMat,ranges,minVals=autoNorm(Mat)
def datingClassTest():
    hoRatio=0.10               #训练集的大小
    datingDataMat,datingLabels=file2matrix("datingTestSet.txt")
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    numTestVecs=int(m*hoRatio)#确定训练集大小
    errorCount=0.0
    for i in range(numTestVecs):
        classifierResult=classify0(normMat[i,:],normMat[numTestVecs:m,:],\
                                   datingLabels[numTestVecs:m],3)
        print("the calssifier came back with:%d, the real answer is:%d"\
              %(classifierResult,datingLabels[i]))
        if(classifierResult!=datingLabels[i]):
            errorCount+=1
    print("the total error rate is:%f"%(errorCount/float(numTestVecs)))
datingClassTest()
