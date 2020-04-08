#这么模型的输入为前20的数据 输出为向后顺延一位的20个数据
from array import *
import numpy as np
from numpy import array
import pandas as pd
from pandas import read_csv
import math
import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Flatten, ConvLSTM2D, Conv2D,Dropout,Conv3D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import ModelCheckpoint
from keras.utils import plot_model
import datetime
import matplotlib.pyplot as plt
from keras import backend as K


DATA_file='D:\github\ConvLSTM\预测车流量项目\预测车流量项目\BaseStation\\'
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

dataset=dataset[:,1:]      #从第二个维度剔除掉第一位 dataset=np.zeros((7200,9))
trainingfraction = 0.67                            #设置训练集样本比例
train_size = round(len(dataset) * trainingfraction)      #取整
traindata=dataset[:train_size,:]
testdata=dataset[train_size:,:]
a=np.zeros((len(traindata),1))
for i in range(9):
    b=traindata[:,i]
    b=b.reshape((len(traindata),1))
    b=(b-np.min(b))/(np.max(b)-np.min(b))
    a=np.concatenate((a,b),axis=1)
datatrain=a[:,1:]
datatrain2D=datatrain.reshape((len(datatrain),3,3))

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
def split_sequence1(sequence, n_steps):     #输入序列
  X, y = np.zeros((1,n_steps,3,3)), np.zeros((1,n_steps,3,3))   #初始化
  for i in range(len(sequence)-n_steps-1):
      seq_x, seq_y = sequence[i:i+n_steps], sequence[i+1:i+n_steps+1]              #保存序列分段
      seq_x, seq_y=seq_x.reshape((1,n_steps,3,3)),seq_y.reshape((1,n_steps,3,3))    #将序列分段增加一个维度
      X=np.concatenate((X,seq_x),axis=0)                #序列分段按第一个维度合并
      y=np.concatenate((y,seq_y),axis=0)
  X=X[1:]                  #剔除序列的第一位
  y=y[1:]
  return X, y

step=20    #输入序列长度
trainX,trainy=split_sequence1(datatrain2D,step)     #生成数据集
print(trainX.shape)
print(trainy.shape)

c=np.zeros((len(testdata),1))
MAX=np.zeros((9))
MIN=np.zeros((9))
for i in range(9):
    b=testdata[:,i]
    MIN[i]=np.min(b)
    MAX[i]=np.max(b)
    b=b.reshape((len(testdata),1))
    b=(b-MIN[i])/(MAX[i]-MIN[i])
    c=np.concatenate((c,b),axis=1)
datatest=c[:,1:]
datatest2D=datatest.reshape((len(datatest),3,3))

testX,testy=split_sequence1(datatest2D,step)     #生成数据集
print(testX.shape)
print(testy.shape)
print(MAX)
print(MIN)
trainX=trainX.reshape((len(trainX),step,3,3,1))
trainy=trainy.reshape((len(trainy),step,3,3,1))


#损失函数自定义
def loss1(y_true, y_pred):                 #定义一个loss函数
    # y_true=y_true[:, -1, :, :, :]          #按照第二维度 取序列的最后一位
    # y_pred=y_pred[:, -1, :, :, :]
    return K.mean(K.square(K.square(y_pred-y_true)),axis=-1)      #计算mse误差

#构建模型输入为（NONE,step,3,3,1)
model = Sequential()  # 模型维序列训练型
model.add(Conv3D(filters=35, kernel_size=(2, 2, 2), padding='same', data_format='channels_last'))  # 3D卷积层
model.add(Dropout(0.2))  # 让神经元有0.2的概率不激活 防止过拟合
model.add(Conv3D(filters=15, kernel_size=(2, 2, 2), padding='same', data_format='channels_last'))  # 3D卷积层
model.add(Dropout(0.2))  # 让神经元有0.2的概率不激活 防止过拟合
model.add(Conv3D(filters=1, kernel_size=(2, 2, 2), padding='same', data_format='channels_last'))  # 3D卷积层
model.add(BatchNormalization())  # 使样本正则化符合正态分布，防止过拟合
model.add(ConvLSTM2D(filters=65, kernel_size=(2, 2), input_shape=(None, 3, 3, 1), padding='same',
                     return_sequences=True))  # 2D 卷积LSTM层
model.add(BatchNormalization())
model.add(Dropout(0.3))  # 让神经元有0.3的概率不激活 防止过拟合
model.add(ConvLSTM2D(filters=65, kernel_size=(2, 2), padding='same', return_sequences=True))  # 2D 卷积LSTM层
model.add(BatchNormalization())
model.add(Dropout(0.3))  # 让神经元有0.3的概率不激活 防止过拟合
model.add(ConvLSTM2D(filters=65, kernel_size=(2, 2), padding='same', return_sequences=True))  # 2D 卷积LSTM层
model.add(BatchNormalization())
model.add(Dropout(0.3))  # 让神经元有0.3的概率不激活 防止过拟合
model.add(ConvLSTM2D(filters=35,kernel_size=(2, 2), padding='same', return_sequences=True))  # 2D 卷积LSTM层

model.add(Dropout(0.3))  # 让神经元有0.3的概率不激活 防止过拟合
model.add(Conv3D(filters=35,kernel_size=(2, 2, 2), padding='same', data_format='channels_last'))  # 3D卷积层

model.add(Conv3D(filters=15, kernel_size=(2, 2, 2), padding='same', data_format='channels_last'))  # 3D卷积层

model.add(Conv3D(filters=3,kernel_size=(2, 2, 2), padding='same', data_format='channels_last'))  # 3D卷积层
model.add(BatchNormalization())
model.add(Conv3D(filters=1,kernel_size=(1, 1, 1), padding='same', data_format='channels_last'))  # 3D卷积层
model.compile(loss=loss1, optimizer='adam',metrics= [loss1])  # 模型编译
model.build((None, step, 3, 3, 1))  # 模型构建  输入为(None, None, 50, 100, 1)
print(model.summary())  # 显示模型信息

#训练网络
filepath = "D:\github\ConvLSTM\预测车流量项目\预测车流量项目\\trainedmodel\weights.best71.hdf5"  # 训练的模型保存路径
model.load_weights(filepath)                       #模型权值加载   （如果重新训练可以注释掉这个代码）
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only=False, mode='max')  # 构建检查点
# callbacks_list = [checkpoint]
# model.fit(trainX, trainy,batch_size=200, epochs=500, validation_split=0.5, callbacks=callbacks_list)  # 每迭代一次保存一次
# model.save('trainedmodel/nice_model.h5')  # 最终训练完毕保存的模型

#验证网络
testX=testX.reshape((len(testX),step,3,3,1))
testy=testy.reshape((len(testy),step,3,3,1))

#网络测试取后面33%任意三个序列的数 来预测未来200个时刻

num_test_time = 20  #预测时刻
index=1000   #预测开始点
#多步预测
result = np.zeros(((1, 3, 3, 1)))  # 初始化一个（1，9）的数组
train_pred = testX[index][:, :, :, :]    # 初始原序列
for j in range(num_test_time):
    new_pos = model.predict(train_pred[np.newaxis, :, :, :, :])  # 网络预测
    new=new_pos[:, -1, :, :, :]
    result=np.concatenate((result,new),axis=0)
    train_pred = np.concatenate((train_pred[1:, :, :, :], new), axis=0)  # 原来的数据的后2位和新预测数据组合，组成新的三天序列
    # train_pred=new_pos[0]
result=result[1:, :, :, :]   # 将预测结果去除掉第一位原序列
result=result.reshape((num_test_time,9))
a1=np.zeros((num_test_time,1))
for i in range(9):
    b=result[:,i]
    b=b.reshape((num_test_time,1))
    b=b*(MAX[i]-MIN[i])+MIN[i]
    a1=np.concatenate((a1,b),axis=1)
result=a1[:, 1:]  # 将预测结果去除掉第一位原序列
result=np.round(result)
truth = np.zeros((1, 9))  # 初始化一个（1，9）的数组
for j in range(num_test_time):
    a=testy[index+j-1][-1:,:,:,:]
    a=a.reshape((1,9))
    truth = np.concatenate((truth, a), axis=0)
truth=truth[1:, :]
a2=np.zeros((num_test_time,1))
for i in range(9):
    b=truth[:,i]
    b=b.reshape((num_test_time,1))
    b=b*(MAX[i]-MIN[i])+MIN[i]
    a2=np.concatenate((a2,b),axis=1)
truth=a2[:, 1:]  # 将预测结果去除掉第一位原序列
truth=np.round(truth)
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

plt.savefig("D:\github\ConvLSTM\预测车流量项目\预测车流量项目\\result_new\多步预测result.png")

#单步预测
result = np.zeros(((1, 3, 3, 1)))  # 初始化一个（1，9）的数组
for j in range(num_test_time):
    train_pred = testX[index+j][:, :, :, :]  # 初始原序列
    new_pos = model.predict(train_pred[np.newaxis, :, :, :, :])  # 网络预测
    new=new_pos[:, -1, :, :, :]
    result=np.concatenate((result,new),axis=0)
    # train_pred = np.concatenate((train_pred[1:, :, :, :], new), axis=0)  # 原来的数据的后2位和新预测数据组合，组成新的三天序列
result=result[1:, :, :, :]   # 将预测结果去除掉第一位原序列
result=result.reshape((num_test_time,9))
a1=np.zeros((num_test_time,1))
for i in range(9):
    b=result[:,i]
    b=b.reshape((num_test_time,1))
    b=b*(MAX[i]-MIN[i])+MIN[i]
    a1=np.concatenate((a1,b),axis=1)
result=a1[:, 1:]  # 将预测结果去除掉第一位原序列
result=np.round(result)
truth = np.zeros((1, 9))  # 初始化一个（1，9）的数组
for j in range(num_test_time):
    a=testy[index+j-1][-1:,:,:,:]
    a=a.reshape((1,9))
    truth = np.concatenate((truth, a), axis=0)
truth=truth[1:, :]
a2=np.zeros((num_test_time,1))
for i in range(9):
    b=truth[:,i]
    b=b.reshape((num_test_time,1))
    b=b*(MAX[i]-MIN[i])+MIN[i]
    a2=np.concatenate((a2,b),axis=1)
truth=a2[:, 1:]  # 将预测结果去除掉第一位原序列
truth=np.round(truth)
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

plt.savefig("D:\github\ConvLSTM\预测车流量项目\预测车流量项目\\result_new\单步预测result.png")