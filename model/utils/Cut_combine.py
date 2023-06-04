# -*-coding:utf-8-*-
"""
Created on 2022.5.1
programing language:python
@author:夜剑听雨
"""
# '''
#     1.对原始数据块(arr1)的右方和下方进行填充，使其横向和竖向都可以整除patch(L*L)。
#     2.将切好的patch喂入网络训练后，只取数据的中心部分(L * L),按照顺序拼起来既可以和arr1一样大的数据。
# '''
import numpy as np

def cut(seismic_block, patch_size, stride_x, stride_y):
    """
    :param seismic_block: 地震数据
    :param patch_size: 切片大小
    :param stride_x: 横向切片步长，大小等于patch_size
    :param stride_y: 竖向切片步长，大小等于patch_size
    :return: 按照规则填充后，获得的切片数据(以列表形式存储)，高方向切片数量，宽方向切片数量
    """
    [seismic_h, seismic_w] = seismic_block.shape  # 获取地震数据块的高(seismic_block_h)和宽(seismic_block_w)
    # 对数据进行填充，确保可以完整切片
    # 确定宽方向填充后大小
    n1 = 1
    while (patch_size + (n1 - 1) * stride_x) <= seismic_w:
        # 判断长为patch_size,步长为stride_x在长为seismic_w的时候能滑动多少步
        n1 = n1 + 1
    # 循环结束后计算的patch_size + (n1-1)*stride_x) > seismic_w，在滑动整数步长的时候可以完全覆盖数据
    arr_w = patch_size + (n1 - 1) * stride_x
    # 确定高方向填充后大小
    n2 = 1
    while (patch_size + (n2 - 1) * stride_y) <= seismic_h:
        n2 = n2 + 1
    arr_h = patch_size + (n2 - 1) * stride_y
    # # 对seismic_block数据块的右方和下方进行填充，填充内容为0
    fill_arr = np.zeros((arr_h, arr_w), dtype=np.float32)
    fill_arr[0:seismic_h, 0:seismic_w] = seismic_block
    # 对数据填充后，我们切分的数据是填充后的数据
    # 计算arr_w方向滑动步长位置
    # python中索引默认0开始，而且左闭右开。patch_size + (n-1)*stride_x就是切片滑动时候的实际位置加1
    # 这里的n算出来大了1
    path_w = []  # 用于存储x方向滑动步长位置
    x = np.arange(n1)  # 生成[0~n1-1]的序列数字
    x = x + 1  # 将序列变成[1~n1]
    for i in x:
        s_x = patch_size + (i - 1) * stride_x  # 计算每一次的步长位置(实际位置加1)
        path_w.append(s_x)  # 添加到列表
    number_w = len(path_w)
    path_h = []
    y = np.arange(n2)
    y = y + 1
    for k in y:
        s_y = patch_size + (k - 1) * stride_y
        path_h.append(s_y)
    number_h = len(path_h)
    #  通过切片的索引位置在数据中提取小patch
    cut_patches = []
    for index_x in path_h:  # path_h索引是patch的行
        for index_y in path_w:  # path_w索引是patch的列
            patch = fill_arr[index_x - patch_size: index_x, index_y - patch_size: index_y]
            cut_patches.append(patch)
    return cut_patches, number_h, number_w, arr_h, arr_w

def combine(patches, patch_size, number_h, number_w, block_h, block_w):
    """
    完整数据用get_patches切分后，将数据进行还原会原始数据块大小
    :param patches: get_patches切分后的结果，以列表形式传入
    :param patch_size: 数据切片patch的大小
    :param number_h: 高方向切出的patch数量
    :param number_w: 宽方向切出的patch数量
    :param block_h: 地震数据块的高
    :param block_w: 地震数据块的宽
    :return: 还原后的地震数据块
    """
    # 将列表patch1中的数据取出，转换成二维矩阵。按照列表元素顺序拼接。
    # patch_size = int(patch_size)
    temp = np.zeros((int(patch_size), 1), dtype=np.float32)  # 临时拼接矩阵，后面要删除
    # 取出patch1中的每一个元素，在列方向(axis=1)拼接
    for i in range(len(patches)):
        temp = np.append(temp, patches[i], axis=1)
    # 删除temp后，此时temp1的维度是 patch_size * patch_size*number_h*number_w
    temp1 = np.delete(temp, 0, axis=1)  # 将temp删除

    # 将数据变成 (patch_size*number_h) * (patch_size*number_w)
    test = np.zeros((1, int(patch_size*number_w)), dtype=np.float32)  # 临时拼接矩阵，后面要删除
    # 让temp1每隔patch_size/2*number_w列就进行一个换行操作
    for j in range(0, int(patch_size*number_h*number_w), int(patch_size*number_w)):
        test = np.append(test, temp1[:, j:j + int(patch_size*number_w)], axis=0)
    test1 = np.delete(test, 0, axis=0)  # 将test删除
    block_data = test1[0:block_h, 0:block_w]
    return block_data