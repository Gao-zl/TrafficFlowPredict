  # 添加rmse结果
# 修改后续的look_back = 4,增加预测长度
# 修改为读取所有基站的一号文件来分析v2.4
# 增加箱形图展示v2.5
# 提取参数方便修改v2.6
# 整合代码v2.7
# ------------------运行前的注意内容：------------------
# 注意事项：1、运行前修改参数，代码位置：最下方的main中设定
#          2、修改箱线图保存位置和文件名，代码位置：查找plt.savefig所在行
#          3、修改预测图保存位置和文件名，代码位置：查找plt.savefig所在行

import os
import numpy
import math
import pandas as pd
import matplotlib.pyplot as plt
from pandas import read_csv
from keras.models import Sequential
from keras.models import Sequential, load_model
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

def run(V_epoches,V_BaseStation,V_forward,V_look_back):
    rmse_list = [0] * V_BaseStation
    for i in range(1,V_BaseStation + 1):
        current = i
        print(current)
        file_name = "BaseStation" + str(i) + ".csv"
        # test_file_name = "train2.csv"
        dataframe = read_csv(file_name,usecols = [1], engine = "python", skipfooter = 3)
        dataset  = dataframe.values
        dataset = dataset.astype("float32")

        def creat_dataset(dataset, look_back = 4):
            dataX, dataY = [], []
            for i in range(len(dataset) - look_back - 1):
                a = dataset[i: (i + look_back), 0]
                dataX.append(a)
                dataY.append(dataset[i + look_back, 0])
            return numpy.array(dataX), numpy.array(dataY)

        numpy.random.seed(7)

        scaler = MinMaxScaler(feature_range = (0, 1))
        dataset = scaler.fit_transform(dataset)

        train_size  = int(len(dataset) * 0.67)
        test_size = len(dataset) - train_size
        train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

        look_back = V_look_back
        trainX, trainY = creat_dataset(train, look_back)
        testX, testY = creat_dataset(test, look_back)

        # 修改模型构建
        trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
        testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

        # 模型构建
        model = Sequential()
        model.add(LSTM(4, input_shape = (1, look_back)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(trainX, trainY, epochs=V_epoches, batch_size=1, verbose=2)

        trainPredict = model.predict(trainX)
        testPredict = model.predict(testX)

        # 查看测试集和训练集的形状，后续可删
        # print(trainPredict.shape)
        # print(testPredict.shape)

        #多步预测代码
        num = V_forward   #想要预测多少步
        index = 15   #从测试集的那个时刻开始预测
        result= numpy.zeros((1))     #构造一个空的预测值的初始向量
        train_pred=testX[index][:,:]     #选取测试集第index的输入序列
        for i in range(num):
            a=model.predict(train_pred[numpy.newaxis, :, :])   #预测
            print(a.shape)
            a=a.reshape((1))
            b=a.reshape((1,1))       #预测值reshape
            result=numpy.concatenate((result,a),axis=0)     #将预测的值都扩充到result里
            train_pred=numpy.concatenate((train_pred[:,1:],b),axis=1)    #将旧的输入序列的第一位剔除 再最后一位插入预测的值

        result = result[1:]  #多步预测的值剔除第一位空向量
        print(result.shape)
        result = scaler.inverse_transform(result.reshape(-1,1))
        truth = numpy.zeros((1))     #构造一个空的真值
        for i in range(num):
            a=testY[index+i]     #循环读取
            a=a.reshape((1))
            truth=numpy.concatenate((truth,a),axis=0)

        truth=truth[1:]   #真值
        print(truth.shape)

        # 将归一化的值恢复原始值来计算rmse
        truth = scaler.inverse_transform(truth.reshape(-1, 1))

        # 修正rmse的结果测试集合
        # print("result")
        # print(result)
        # print("truth")
        print(truth)

        rmse = math.sqrt(mean_squared_error(truth, result))
        print("test RMSE: %.3f" % rmse)
        rmse_list[current - 1] = rmse
        # print(rmse_list)

        # 绘制箱线图,最后一步的时候展示
        # 运行前修改文件存放位置和文件名
        if (current == V_BaseStation):
            box_df = pd.DataFrame(rmse_list)
            box_df.plot.box(title="box of rmse")
            plt.grid(linestyle="--", alpha=0.3)
            # plt.show()
            plt.savefig("D:\college后续更新文件\毕设\VanillaLSTM\VanillaLSTM\img_new\epochs30_look_back10")
            plt.close()

        #多步预测图像展示
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))  # 绘图
        ax.set_title("basestation {:d}".format(current) + "| test RMSE %.3f" % rmse)
        data = result
        x = numpy.arange(len(result))
        ax.plot(x, data, linewidth=1.0)  # 显示预测的
        data1 = truth
        x1 = numpy.arange(len(truth))
        ax.plot(x1, data1, linewidth=1.0)  # 显示实际的
        # plt.show()    # 是否要展示图形，方便后续统一查看，此步删
        # 保存截图，决定放入的位置以及命名的名称的设定，注：已改，下次需重新设定
        plt.savefig('D:\college后续更新文件\毕设\VanillaLSTM\VanillaLSTM\img\Each_BaseStation_look_back10_epochs30\epochs30_forward6_look_back10_test' + str(current) + '.png')
        # 必须先关闭再重新绘图，否则后续只会产生一张图
        plt.close()

        # 旧的单步预测结果，后续删
        # trainPredict = scaler.inverse_transform(trainPredict)
        # trainY = scaler.inverse_transform([trainY])
        #
        # testPredict = scaler.inverse_transform(testPredict)
        # testY = scaler.inverse_transform([testY])
        # print("testY")
        # print(testY)
        #
        # trainPredictPlot = numpy.empty_like(dataset)
        # trainPredictPlot[:, :] = numpy.nan
        # trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict
        #
        # testPredictPlot = numpy.empty_like(dataset)
        # testPredictPlot[:, :] = numpy.nan
        # testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict
        #
        # # 均方根误差RMSE计算损失
        # # invert scaling for forcast
        # forcast = testPredict[:, 0]
        # # invert scaling for actual
        # actual = scaler.inverse_transform(dataset)
        # actual = actual[-len(testPredict):, 0]
        # rmse = math.sqrt(mean_squared_error(actual, forcast))
        # print("test RMSE: %.3f" % rmse)
        #
        # # print("=================================================================")
        # # print('第' + str(i) + "幅图像")
        # # plt.plot(scaler.inverse_transform(dataset))
        # # # plt.plot(trainPredictPlot)
        # # plt.plot(testPredictPlot)
        # # plt.show()

if __name__ == '__main__':
    # 定义参数后续运算
    V_epoches = 30   # 运行次数
    V_BaseStation = 9  # 基站数量，目前为9
    V_forward = 6   # 向前预测多少步
    V_look_back = 10 # 使用前几个值预测下一个
    run(V_epoches,V_BaseStation,V_forward,V_look_back)