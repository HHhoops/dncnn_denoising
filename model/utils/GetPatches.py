# -*-coding:utf-8-*-
"""
Created on 2022.3.1
programing language:python
@author:夜剑听雨
"""
import glob
import cv2
import numpy as np
import segyio
import matplotlib.pyplot as plt
import random
import os

def read_segy_data(filename):
    """
    读取segy或者sgy数据，剥离道头信息
    :param filename: segy或者sgy文件的路径
    :return: 不含道头信息的地震道数据
    """
    print("### Reading SEGY-formatted Seismic Data:")
    print("Data file-->[%s]" %(filename))
    with segyio.open(filename, "r", ignore_geometry=True)as f:
        f.mmap()
        data = np.asarray([np.copy(x) for x in f.trace[:]]).T
    f.close()
    return data

def data_augmentation(img, mode=None):
    """
    data augmentation 数据扩充
    :param img: 二维矩阵
    :param mode: 对矩阵的翻转方式
    :return: 翻转后的矩阵
    """
    if mode == 0:
        # original 原始的
        return img
    elif mode == 1:
        # flip up and down 上下翻动
        return np.flipud(img)
    elif mode == 2:
        # 逆时针旋转90度
        return np.rot90(img)
    elif mode == 3:
        #  先旋转90度，在上下翻转
        return np.flipud(np.rot90(img))
    elif mode == 4:
        #  旋转180度
        return np.rot90(img, k=2)
    elif mode == 5:
        # 旋转180度并翻转
        return np.flipud(np.rot90(img, k=2))
    elif mode == 6:
        # 旋转270度
        return np.rot90(img, k=3)
    elif mode == 7:
        # 旋转270度并翻转
        return np.flipud(np.rot90(img, k=3))

def gen_patches(file_path, patch_size, stride_x, stride_y, scales):
    """
    对单炮数据进行数据切片，需要先将单炮数据的数据和道头剥离。
    Args:
        file_path:地震道数据的文件路径。
        patch_size:切片数据的大小，都是方形所以高宽一致。
        stride_x:在地震道数据x方向的滑动步长。
        stride_y:在地震道数据y方向的滑动步长。
        scales:输入为列表，对数据进行放缩。
    Returns:返回一系列的小数据块
    """
    shot_data = np.load(file_path)  # 加载npy数据
    time_sample, trace_number = shot_data.shape  # 获取数据大小
    patches = []   # 生成空列表用于添加小数据块
    for scale in scales:  # 遍历数据的缩放方式
        time_scaled, trace_scaled = int(time_sample * scale), int(trace_number * scale)  # 缩放后取整
        shot_scaled = cv2.resize(shot_data, (trace_scaled, time_scaled), interpolation=cv2.INTER_LINEAR) # 获得缩放后的数据，采用双线性插值
        # 数据归一化处理
        shot_scaled = shot_scaled / abs(shot_scaled).max()  # 将数据归一化到(-1,1)
        # 从放缩之后的shot_scaled中提取多个patch
        # 计算x方向滑动步长位置
        s1 = 1
        while (patch_size + (s1-1)*stride_x) <= trace_scaled:
            s1 = s1 + 1
        # python中索引默认0开始，而且左闭右开。patch_size + (n-1)*stride_x就是切片滑动时候的实际位置加1
        # 这里的n算出来大了1
        strides_x = []  # 用于存储x方向滑动步长位置
        x = np.arange(s1-1)  # 生成0~s1-2的序列数字
        x = x + 1  # 将序列变成1~s1-1
        for i in x:
            s_x = patch_size + (i-1)*stride_x  # 计算每一次的步长位置(实际位置加1)
            strides_x.append(s_x)  # 添加到列表
        # 计算y方向滑动步长位置
        s2 = 1
        while (patch_size + (s2-1)*stride_y) <= time_scaled:
            s2 = s2 + 1
        strides_y = []
        y = np.arange(s2-1)
        y = y + 1
        for i1 in y:
            s_y = patch_size + (i1-1)*stride_y
            strides_y.append(s_y)
        #  通过切片的索引位置在数据中提取小patch
        for index_x in strides_y:  # x方向索引是patch的列
            for index_y in strides_x:  # y方向索引是patch的行
                patch = shot_scaled[index_x-patch_size: index_x, index_y-patch_size: index_y]
                patches.append(patch)
    return patches

def data_generator(data_dir, patch_size, stride_x, stride_y, scales):
    """
    对整个目录下的npy文件进行，数据的切片。
    Args:
        data_dir: 文件夹路径
        patch_size:切片数据的大小，都是方形所以高宽一致。
        stride_x:在地震道数据x方向的滑动步长。
        stride_y:在地震道数据y方向的滑动步长。
        scales:输入为列表，对数据进行放缩。
    Returns:总的切片数据
    """
    file_list = glob.glob(os.path.join(data_dir, '*npy'))
    data = []
    for i in range(len(file_list)):
        patches = gen_patches(file_list[i], patch_size, stride_x, stride_y, scales)
        for patch in patches:
            data.append(patch)
    print("获得切片数量：{}".format(len(data)))
    return data
def calculate_patches(time_number, trace_number, blocks, patch_size, stride_x, stride_y, scales):
    """
    计算数据的切片数量
    Args:
        time_number: 单个数据块的时间采样点数
        trace_number: 单个数据块的地震道数
        blocks: 需要切片的数据块的数量
        patch_size: 切片patch的大小
        stride_x:在地震道数据x方向的滑动步长
        stride_y:在地震道数据y方向的滑动步长
        scales:输入为列表，对数据进行放缩
    Returns: 总的切片数据的数量
    """
    sum_patches = 0
    for scale in scales:  # 便利数据的缩放方式
        time_scaled, trace_scaled = int(time_number * scale), int(trace_number * scale)  # 缩放后取整
        # 从放缩之后的shot_scaled中提取多个patch
        # 数据切分收到滑动步长和数据块尺寸的共同影响，先确定数据块滑动步长位置
        n = 1
        while (patch_size + (n - 1) * stride_x) <= trace_scaled:
            n = n + 1
        # python中索引默认0开始，而且左闭右开。patch_size + (n-1)*stride_x就是切片滑动时候的实际位置加1
        # 这里的n算出来大了1
        strides_x = []  # 用于存储x方向滑动步长位置
        x = np.arange(n - 1)  # 生成0~n-2的序列数字
        x = x + 1  # 将序列变成1~n-1
        for i in x:
            s_x = patch_size + (i - 1) * stride_x  # 计算每一次的步长位置(实际位置加1)
            strides_x.append(s_x)  # 添加到列表
        # 计算y方向滑动步长位置
        n = 1
        while (patch_size + (n - 1) * stride_y) <= time_scaled:
            n = n + 1
        strides_y = []
        y = np.arange(n - 1)
        y = y + 1
        for i in y:
            s_y = patch_size + (i - 1) * stride_y
            strides_y.append(s_y)
        numbers = len(strides_y) * len(strides_x) * blocks
        sum_patches += numbers
    return sum_patches
if __name__ == "__main__":

    # 可用calculate_patches函数提前估算切片数量
    # patch_num = calculate_patches(1501, 301, 30, 64, 32, 64, [1])
    # print(patch_num)

    # 对剥离后的数据进行切分
    data_dir1 = "..\\data\\sgy_data\\noise\\"  # 含噪数据文件路径
    data_dir2 = "..\\data\\sgy_data\\clean\\"  # 干净数据文件路径
    patch_size = 64  # 数据块patch的大小
    scales = [1]     # 数据块拉伸的方式
    xs = data_generator(data_dir1, patch_size, 32, 64, scales)  # 含噪和抽稀数据patches，即特征。
    ys = data_generator(data_dir2, patch_size, 32, 64, scales)  # 干净数据patches，即标签。

    # 对标签和样本数据进行随机翻转, len(xs)=len(ys)
    patches_index = range(len(xs))  # 获取标签数据或者样本数据的长度，变成索引
    enhance_number = int(len(xs) * 0.2)    # 数据增强的数量,20%的比例
    enhance_numbers = random.sample(patches_index, enhance_number)  # 从patches_index随机抽取20%个元素
    for k in enhance_numbers:  # 遍历随机抽取出来的索引
        random_number = random.randint(0, 7)  # 产生一个随机数mode: a <= mode <= b
        # 对标签和样本同步随机翻转
        data_augmentation(xs[k], mode=random_number)
        data_augmentation(ys[k], mode=random_number)

    # 保存数据集
    for j in range(len(xs)):
        feature = xs[j]
        label = ys[j]
        noise_name = f'feature{j+1}'
        label_name = f'label{j+1}'
        np.save("..\\data\\feature\\" + noise_name, feature)
        np.save("..\\data\\label\\" + label_name, label)
    print(f'一共保存{len(xs)}个训练集地震数据切片！')

    # 查看若干个切片
    c1 = np.load('../data/feature/feature12.npy')
    n1 = np.load('../data/label/label12.npy')
    c2 = np.load('../data/feature/feature60.npy')
    n2 = np.load('../data/label/label60.npy')
    fig1 = plt.figure()
    # 三个参数分别为：行数，列数，
    ax1 = fig1.add_subplot(2, 2, 1)
    ax2 = fig1.add_subplot(2, 2, 2)
    ax3 = fig1.add_subplot(2, 2, 3)
    ax4 = fig1.add_subplot(2, 2, 4)
    # 绘制曲线
    ax1.imshow(c1, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax2.imshow(n1, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax3.imshow(c2, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    ax4.imshow(n2, cmap=plt.cm.seismic, interpolation='nearest', aspect=1, vmin=-0.5, vmax=0.5)
    plt.tight_layout()  # 自动调整子图位置
    plt.show()
