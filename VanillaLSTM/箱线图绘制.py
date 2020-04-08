import pandas as pd
import numpy
import matplotlib.pyplot as plt

# 从各张图像上去获取值
look_back4_forward3 = [1.411,0.745,2.670,3.032,2.806,1.069,0.718,0.592,0.170]
look_back4_forward6 = [1.806,1.395,3.360,2.339,3.344,1.811,0.590,1.972,0.264]
look_back4_forward20 = [2.409,1.931,3.782,2.623,8.996,3.118,1.262,4.443,0.843]
look_back10_forward3 = [0.694,2.413,0.455,0.834,2.863,0.414,0.345,1.759,1.741]
look_back10_forward6 = [2.510,1.875,0.798,0.946,2.756,0.453,1.064,1.422,2.279]
look_back10_forward20 = [1.776,4.995,2.301,3.332,4.468,3.018,5.986,1.142,3.166]
look_back20_forward3 = [0.631,0.458,1.090,4.871,3.368,1.098,1.661,1.403,1.062]
look_back20_forward6 = [0.583,2.365,1.563,6.208,4.380,1.425,1.473,2.159,1.052]
look_back20_forward20 = [3.338,4.209,5.319,8.930,11.170,1.502,4.212,2.083,1.246]
look_back50_forward3 = [1.316,1.495,2.390,0.839,0.767,1.071,0.546,1.282,0.582]
look_back50_forward6 = [1.705,1.603,3.610,2.035,0.918,1.379,0.422,2.702,1.365]
look_back50_forward20 = [3.613,2.571,5.153,6.190,9.284,2.716,1.361,8.140,3.981]
look_back100_forward3 = [1.523,2.144,1.522,1.815,1.450,0.331,0.960,0.454,0.637]
look_back100_forward6 = [3.723,2.633,3.858,2.206,3.371,1.308,1.266,0.411,0.906]
look_back100_forward20 = [4.492,1.835,8.555,1.853,16.104,4.949,5.122,0.922,2.115]

# 格式为-->底下的x坐标值：数据
box_df = pd.DataFrame({
    "4-3":look_back4_forward3,
    "4-6":look_back4_forward6,
    "4-20":look_back4_forward20,
    "10-3":look_back10_forward3,
    "10-6":look_back10_forward6,
    "10-20":look_back10_forward20,
    "20-3":look_back20_forward3,
    "20-6":look_back20_forward6,
    "20-20":look_back20_forward20,
    "50-3":look_back50_forward3,
    "50-6":look_back50_forward6,
    "50-20":look_back50_forward20,
    "100-3":look_back100_forward3,
    "100-6":look_back100_forward6,
    "100-20":look_back100_forward20
})
# 设定颜色
color = ['g','r','b','g','r','b','g','r','b','g','r','b','g','r','b']

# 绘图
p = box_df.plot.box(title="box of rmse",
                    patch_artist=True,
                    return_type = 'dict',   # 指出输出类型为字典，否则程序无法获得下标上不了色
                    positions=[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19]
                    )
plt.grid(linestyle="--", alpha=0.3)
# 坐标轴说明
plt.xlabel("look_back-forward")
# 上色
for box,c in zip(p['boxes'], color):
    box.set(color=c)
# 展示，点图像上的保存按钮选保存位置
plt.show()
plt.close()