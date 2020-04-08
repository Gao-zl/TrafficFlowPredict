from array import *
import numpy as np
from numpy import array
import pandas as pd
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, ConvLSTM2D, Conv2D,Dropout,Conv3D,UpSampling3D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import datetime
import matplotlib.pyplot as plt
import keras.backend as K
import random


DATA_file='BaseStation/'
def get_df(i):                  #读取csv中的数据
  BaseStation = DATA_file+'BaseStation' + str(i) + '.csv'
  dataframe = pd.read_csv(BaseStation, usecols= [1])
  datavalue = dataframe.values
  df = datavalue.tolist()
  df = eval("[%s]"%repr(df).replace('[','').replace(']',''))
  return df

dataset=np.zeros((7200,1))     #初始化一个dataset
for i in range(9):
  df = get_df(i+1)           #循环读取
  df=np.array(df)            #list转numpy
  df=df.reshape((7200,1))    #numpy尺寸重构
  dataset=np.concatenate((dataset,df),axis=1)     #从第二个维度把数据整合

dataset=dataset[:,1:]      #从第二个维度剔除掉第一位 dataset=np.zeros((7200,1))
print(dataset.shape)
dataset2D=dataset.reshape((7200,3,3))      #将数据集重构为7200*3*3的数组 即 7200个3*3矩阵

# 分割数据集
def split_sequence(sequence, n_steps):     #输入序列
  X, y = np.zeros((1,n_steps,3,3)), np.zeros((1,3,3))   #初始化
  for i in range(len(sequence)-n_steps-1):
      seq_x, seq_y = sequence[i:i+n_steps], sequence[i+n_steps]              #保存序列分段
      seq_x, seq_y=seq_x.reshape((1,n_steps,3,3)),seq_y.reshape((1,3,3))    #将序列分段增加一个维度
      X=np.concatenate((X,seq_x),axis=0)                #序列分段按第一个维度合并
      y=np.concatenate((y,seq_y),axis=0)
  X=X[1:]                  #剔除序列的第一位
  y=y[1:]
  return X, y

step=10    #输入序列长度
X,y=split_sequence(dataset2D,step)     #生成数据集
print(X.shape)
print(y.shape)
X=X.reshape((len(X),step,3,3,1))
y=y.reshape((len(y),9))
MIN=np.min(dataset)
MAX=np.max(dataset)
X=(X-MIN)/(MAX-MIN)    #数据归一化
y=(y-MIN)/(MAX-MIN)    #数据归一化

def recall(y_true, y_pred):
    # Calculates the recall
    return K.mean(K.square(K.square(y_pred - y_true)), axis=-1)  # 计算mse误差

#构建模型输入为（NONE,3,3,3,1)
model = Sequential()
model.add(ConvLSTM2D(filters = 64, kernel_size = (1, 1),padding='same',input_shape = (step,3,3,1),return_sequences=True))  # 2D 卷积LSTM层
model.add(Dropout(0.2))  # 让神经元有0.3的概率不激活 防止过拟合
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=32, kernel_size=(1, 1), padding='same',return_sequences=True))  # 2D 卷积LSTM层
model.add(BatchNormalization())
model.add(ConvLSTM2D(filters=1, kernel_size=(1, 1), padding='same',return_sequences=True))  # 2D 卷积LSTM层
# model.add(BatchNormalization())
# model.add(Conv2D(filters=8, kernel_size=(1, 1),activation = "sigmoid", padding='same'))  # 2D 卷积LSTM
# model.add(BatchNormalization())
# model.add(Conv2D(filters=1, kernel_size=(1, 1),activation = "sigmoid", padding='same'))  # 2D 卷积LSTM
model.add(BatchNormalization())
model.add(Flatten())
# model.add(Dense(63,activation = "sigmoid", use_bias= True))
# model.add(BatchNormalization())
# model.add(Dense(31,activation = "sigmoid", use_bias= True))
# model.add(BatchNormalization())
model.add(Dense(9,activation = "sigmoid", use_bias= True))
model.compile(optimizer = "adam", loss = "mse", metrics= [ recall])
print(model.summary())

#设置训练集为前67%
trainingfraction = 0.67                            #设置训练集样本比例
train_size = round(len(X) * trainingfraction)      #取整
#训练网络
filepath = "trainedmodel/weights.best32.hdf5"       #训练的模型保存路径
model.load_weights(filepath)                       #模型权值加载   （如果重新训练可以注释掉这个代码）
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, mode='max')  # 构建检查点
# callbacks_list = [checkpoint]
# model.fit(X[:train_size], y[:train_size], batch_size=80, epochs=5000, validation_split=0.05,
#           callbacks=callbacks_list)  # 每迭代一次保存一次
# model.save('trainedmodel/nice_model.h5')  # 最终训练完毕保存的模型

#网络测试取后面33%任意三个序列的数 来预测未来200个时刻
num_test_time = 50  #预测时刻
index=800   #预测开始点
result = np.zeros((1, 9))  # 初始化一个（1，9）的数组

for j in range(num_test_time):
    train_pred = X[index+j][:, :, :, :]  # 初始原序列
    new_pos = model.predict(train_pred[np.newaxis, :, :, :, :])  # 网络预测
    new=new_pos.reshape(1,3,3,1)
    result=np.concatenate((result,new_pos),axis=0)
    # train_pred = np.concatenate((train_pred[1:, :, :, :], new), axis=0)  # 原来的数据的后2位和新预测数据组合，组成新的三天序列
result = result[1:, :]  # 将预测结果去除掉第一位原序列
# result=result*(MAX-MIN)+MIN
# result=np.round(result)
truth = np.zeros((1, 9))  # 初始化一个（1，9）的数组
for j in range(num_test_time):
    a=y[index+j]
    a=a.reshape((1,9))
    truth = np.concatenate((truth, a), axis=0)
truth=truth[1:, :]
# truth=truth*(MAX-MIN)+MIN
print(result.shape)
print(truth.shape)

k=0

fig, ax = plt.subplots(3,3,figsize=(15, 15))  # 绘图
for i in range(3):
    for j in range(3):
      k=k+1
      ax[i,j].set_title("basestation {:d}".format(k))
      data = result[:, k-1].reshape((len(result)))
      x=np.arange(len(result))
      ax[i,j].plot(x,data, linewidth=1.0)  # 显示预测的
      data1 = truth[:, k-1].reshape((len(truth)))
      x1 = np.arange(len(truth))
      ax[i,j].plot(x1,data1, linewidth=1.0)  # 显示实际的

plt.savefig("result/单步预测result.png")

result = np.zeros((1, 9))  # 初始化一个（1，9）的数组
train_pred = X[index][:, :, :, :]  # 初始原序列
for j in range(num_test_time):
    new_pos = model.predict(train_pred[np.newaxis, :, :, :, :])  # 网络预测
    new=new_pos.reshape(1,3,3,1)
    result=np.concatenate((result,new_pos),axis=0)
    train_pred = np.concatenate((train_pred[1:, :, :, :], new), axis=0)  # 原来的数据的后2位和新预测数据组合，组成新的三天序列
result = result[1:, :]  # 将预测结果去除掉第一位原序列

# result=result*(MAX-MIN)+MIN
# result=np.round(result)
truth = np.zeros((1, 9))  # 初始化一个（1，9）的数组
for j in range(num_test_time):
    a=y[index+j]
    a=a.reshape((1,9))
    truth = np.concatenate((truth, a), axis=0)
truth=truth[1:, :]
# truth=truth*(MAX-MIN)+MIN
print(result.shape)
print(truth.shape)

k=0

fig, ax = plt.subplots(3,3,figsize=(15, 15))  # 绘图
for i in range(3):
    for j in range(3):
      k=k+1
      ax[i,j].set_title("basestation {:d}".format(k))
      data = result[:, k-1].reshape((len(result)))
      x=np.arange(len(result))
      ax[i,j].plot(x,data, linewidth=1.0)  # 显示预测的
      data1 = truth[:, k-1].reshape((len(truth)))
      x1 = np.arange(len(truth))
      ax[i,j].plot(x1,data1, linewidth=1.0)  # 显示实际的

plt.savefig("result/多步预测result.png")