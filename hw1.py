# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 22:37:15 2021

@author: Li Fei
"""

import pandas as pd
import numpy as np
import math
import csv

#导入数据
data=pd.read_csv('train.csv',encoding='ANSI')#有表头时默认第一行为表头，此时不能设置header=None

#分割出前三列，从第四列开始将数据存储到data
data=data.iloc[:,3:]
data[data=='NR']=0
#print(data)
raw_data=data.to_numpy()
# print(raw_data)

#对data进行调整，将4320*24重组为12*18*480
month_data={}
for month in range(12):
    sample=np.empty([18,480])
    for day in range(20):
        sample[:,day*24:(day+1)*24]=raw_data[18*(20*month+day):18*(20*month+day+1),:]
    month_data[month]=sample
    
x=np.empty([12*471,18*9],dtype=float)
y=np.empty([12*471,1],dtype=float)
for month in range(12):
    for day in range(20):
        for hour in range(24):
            if day==19 and hour>14:
                continue
            x[471*month+24*day+hour,:]=month_data[month][:,24*day+hour:24*day+hour+9].reshape(1,-1)
            y[471*month+24*day+hour,0]=month_data[month][9,24*day+hour+9]
print(x)
print(y)
            
#按列进行归一化
mean_x=np.mean(x,axis=0)
std_x=np.std(x,axis=0)
for i in range(len(x)):
    for j in range(len(x[0])):
        if std_x[j]!=0:
            x[i][j]=(x[i][j]-mean_x[j])/std_x[j]
#将训练集分成训练-验证集，用来最后检验我们的模型
x_train_set = x[: math.floor(len(x) * 0.8), :]
y_train_set = y[: math.floor(len(y) * 0.8), :]
x_validation = x[math.floor(len(x) * 0.8): , :]
y_validation = y[math.floor(len(y) * 0.8): , :]
# print(x_train_set)
# print(y_train_set)
# print(x_validation)
# print(y_validation)
# print(len(x_train_set))
# print(len(y_train_set))
# print(len(x_validation))
# print(len(y_validation))

dim=18*9+1
w=np.zeros([dim,1])
x2=np.concatenate((np.ones([12*471,1]), x),axis=1).astype(float)
learning_rate=2
iter_time=1000
adagrad=np.zeros([dim,1])
eps=0.00000000001
for t in range(iter_time):
    loss=np.sqrt(np.sum(np.power(np.dot(x2,w)-y,2))/471/12)
    if(t%100==0):
        print("迭代的次数：%i ， 损失值：%f"%(t,loss))
    gradient=2*np.dot(x2.transpose(),np.dot(x2,w)-y)
    adagrad+=gradient**2
    w = w - learning_rate * gradient / np.sqrt(adagrad + eps)
# if(t%100==0):
#         print(str(t)+":"+str(loss))
#     gradient=2*np.dot(x2.transpose(),np.dot(x2,w)-y)
    # adagrad+=gradient**2
#     w = w - learning_rate * gradient / (np.sqrt(adagrad + eps))

np.save('weight_copy.npy', w)

testdata=pd.read_csv("test.csv",header=None,encoding='ANSI')#没有表头若不设置header会默认第一行为表头，
testdata=testdata.iloc[:,2:]
testdata[testdata=='NR']=0
test_data=testdata.to_numpy()

test_x=np.empty([240,18*9])
for i in range(240):
    test_x[i,:]=test_data[18*i:18*(i+1),:].reshape(1,-1)
mean_test_x=np.mean(test_x,axis=0)
std_test_x=np.std(test_x,axis=0)
for i in range(len(test_x)):
    for j in range(len(test_x[0])):
        if std_test_x[j]!=0:
            test_x[i][j]=(x[i][j]-mean_test_x[j])/std_test_x[j]
test_x=np.concatenate((np.ones([240,1]),test_x),axis=1).astype(float)
test_x

w=np.load('weight.npy')
ans_y=np.dot(test_x,w)
# print(ans_y)

with open('submit.csv', mode='w', newline='') as submit_file:
    csv_writer = csv.writer(submit_file)
    header = ['id', 'value']
    print(header)
    csv_writer.writerow(header)
    for i in range(240):
        row = ['id_' + str(i), ans_y[i][0]]
        csv_writer.writerow(row)
        print(row)

