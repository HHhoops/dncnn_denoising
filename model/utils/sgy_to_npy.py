# -*-coding:utf-8-*-
"""
Created on 2022.4.19
programing language:python
@author:夜剑听雨
"""
import numpy as np
import matplotlib.pyplot as plt
import random
from GetPatches import read_segy_data
import os

path = os.path.join('../data/sgy_data', 'SYNTHETIC.segy')  # 获取sgy数据路径
# path = 'E:\pythonproject\dncnn_pytorchdz\dncnn_pytorch\data\sgy_data\SYNTHETIC.segy'
sgy_data = read_segy_data(path)  # 读取sgy地震数据
print(sgy_data.shape)   # 查看数据尺寸
  
for i in range(1):
    # 读取干净的炮集
    clean_shot = sgy_data
    # 保存干净的炮集记录
    clean_name = f'clean{i + 1}'  # 给每一个炮集命名，采用format方法
    np.save('..\\data\\sgy_data\\clean\\' + clean_name, clean_shot)  # 设置保存路径

    # 对数据加噪并且保存
    clean_shot_max = abs(clean_shot).max()    # 获取数据最大幅值
    clean_shot = clean_shot / clean_shot_max  # 将数据归一化到(-1,1)
    noise = np.random.random([clean_shot.shape[0], clean_shot.shape[1]])  # 生成幅值为0~1的随机噪声
    rates = [0.05, 0.1, 0.15]  # 设置随机噪声的幅值
    rate = random.sample(rates, 1)  # 产生一个随机数mode: a <= mode <= b
    noise_shot = clean_shot + rate[0] * noise  # 加入数据幅值rate[0]随机噪声
    noise_shot = noise_shot * clean_shot_max  # 逆归一化
    # 保存含噪声的炮集记录
    noise_name = f'noise{i + 1}'
    np.save('..\\data\\sgy_data\\noise\\' + noise_name, noise_shot)

print(f'第{i + 1}个地震数据已经抽稀保存完毕')

# 查看数据
x1 = np.load('../data/sgy_data/clean/clean1.npy')
# x2 = np.load('../data/sgy_data/clean/clean11.npy')
# x3 = np.load('../data/sgy_data/clean/clean23.npy')

y1 = np.load('../data/sgy_data/noise/noise1.npy')
# y2 = np.load('../data/sgy_data/noise/noise11.npy')
# y3 = np.load('../data/sgy_data/noise/noise23.npy')

fig1 = plt.figure()
# 三个参数分别为：行数，列数，
ax1 = fig1.add_subplot(1, 2, 1)
ax2 = fig1.add_subplot(1, 2, 2)
# ax3 = fig1.add_subplot(2, 3, 3)
# ax4 = fig1.add_subplot(2, 3, 4)
# ax5 = fig1.add_subplot(2, 3, 5)
# ax6 = fig1.add_subplot(2, 3, 6)
# 绘制曲线gray
ax1.imshow(x1, cmap=plt.cm.seismic, interpolation='nearest', aspect=3.0, vmin=-0.5, vmax=0.5)
ax2.imshow(y1, cmap=plt.cm.seismic, interpolation='nearest', aspect=3.0, vmin=-0.5, vmax=0.5)
# ax3.imshow(x3, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
# ax4.imshow(y1, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
# ax5.imshow(y2, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
# ax6.imshow(y3, cmap=plt.cm.seismic, interpolation='nearest', aspect=0.25, vmin=-0.5, vmax=0.5)
plt.tight_layout()  # 自动调整子图位置
plt.show()


