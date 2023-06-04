# -*-coding:utf-8-*-
"""
Created on 2022.5.1
programing language:python
@author:夜剑听雨
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.GetPatches import read_segy_data
from utils.Cut_combine import cut, combine
import torch

# 加载数据
seismic_noise = read_segy_data('../data/sgy_data/test.sgy')  # 野外地震数据
seismic_block_h, seismic_block_w = seismic_noise.shape
# 数据归一化处理
seismic_noise_max = abs(seismic_noise).max()  # 获取数据最大幅值
seismic_noise = seismic_noise / seismic_noise_max  # 将数据归一化到(-1,1)
# 对缺失的炮集数据进行膨胀填充，并且切分
patch_size = 64
patches, strides_x, strides_y, fill_arr_h, fill_arr_w = cut(seismic_noise, patch_size, patch_size, patch_size)

# 检测是否有GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# 加载模型
model = torch.load('./save_dir/model_epoch20.pth')
model.to(device=device)  # 模型拷贝至GPU
model.eval()  # 开启评估模式
predict_datas = []  # 空列表，用于存放网络预测的切片数据
# 对切片数据进行网络预测
for patch in patches:
    patch = np.array(patch)  # 转换为numpy数据
    patch = patch.reshape(1, 1, patch.shape[0], patch.shape[1])  # 对数据维度进行扩充(批量，通道，高，宽)
    patch = torch.from_numpy(patch)  # python转换为tensor
    patch = patch.to(device=device, dtype=torch.float32)  # 数据拷贝至GPU
    predict_data = model(patch)  # 预测结果
    predict_data = predict_data.data.cpu().numpy()  # 将数据从GPU中拷贝出来，放入CPU中，并转换为numpy数组
    print(predict_data.shape)
    predict_data = predict_data.squeeze()  # 默认压缩所有为1的维度
    print(predict_data.shape)
    predict_datas.append(predict_data)  # 添加至列表中

# 对预测后的数据进行还原，裁剪
seismic_predict = combine(predict_datas, patch_size, strides_x, strides_y, seismic_block_h, seismic_block_w)
# 数据逆归一化处理
seismic_predict = seismic_predict*seismic_noise_max  # 将数据归一化到(-1,1)
#  显示处理效果
fig1 = plt.figure()
# 三个参数分别为：行数，列数，
ax1 = fig1.add_subplot(1, 3, 1)
ax2 = fig1.add_subplot(1, 3, 2)
ax3 = fig1.add_subplot(1, 3, 3)
# 绘制曲线
ax1.imshow(seismic_noise, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
ax2.imshow(seismic_predict, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
ax3.imshow(seismic_noise-seismic_predict, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
plt.tight_layout()  # 自动调整子图位置
plt.show()
