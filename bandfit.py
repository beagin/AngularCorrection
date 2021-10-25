"""
高分二期验收 - 发射率验证工作
用MODIS的发射率产品对GF5反演得到的发射率进行验证
主要包括
"""
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy import optimize
import sys
import os
import shutil
import requests
from dateutil.parser import parse
import datetime
import re


class SimuBandEmiss(object):

    def __init__(self, emissivityfilefolderPath, SRFlist):
        self.emissFolderPath = emissivityfilefolderPath
        self.emissfileName = os.listdir(emissivityfilefolderPath)
        self.SRFlist = SRFlist

    def _readEmissivityfileTXT(self,filename):  # 读取发射率文件TXT格式
        with open(filename, "r") as f:
            data = f.readlines()
            usefullenth = int(len(data)) - 26
            WL_EMISS = np.empty((2, usefullenth), dtype=float,
                                order='C')  # 建立一个二维数组，分别代表波长和发射率
            for nx in range(usefullenth):
                WL_EMISS[0][nx] = data[nx+26].split()[0]  # 空格分隔后的第一位为波长数据
                WL_EMISS[1][nx] = 1 - eval(data[nx+26].split()[1])/100.0  # 第二位为反射率数据，需要用1-
            if WL_EMISS[0][0] > WL_EMISS[0][1]:
                WL_EMISS = WL_EMISS[:, ::-1]  # 前后翻转
            return WL_EMISS

    def _readEmissivityfileMHT(self,filename):  # 读取发射率文件mht格式
        with open(filename, "r") as f:
            data = f.readlines()
            usefullenth = int(len(data)) - 35 - 1  # 尾部有一行
            WL_EMISS = np.empty((2, usefullenth), dtype=float, order='C')  # 建立一个二维数组，分别代表波长和发射率
            for nx in range(usefullenth):
                # 空格分隔后的第一位为波长数据,微米
                WL_EMISS[0][nx] = data[nx+35].split()[0]
                # 第二位为波数数据，第三位为发射率数据，txt和mht的区别
                WL_EMISS[1][nx] = data[nx+35].split()[2]
            if WL_EMISS[0][0] > WL_EMISS[0][1]:
                WL_EMISS = WL_EMISS[:, ::-1]  # 前后翻转
            return WL_EMISS

    def readEmissivityfile(self):
        WL_EMISSlist = []
        for file in self.emissfileName:
            if file[-3:] != 'txt':
                WL_EMISSlist.append(self._readEmissivityfileMHT(self.emissFolderPath + '/'+file))
            else:
                WL_EMISSlist.append(self._readEmissivityfileTXT(self.emissFolderPath + '/'+file))

        # for i in range(len(WL_EMISSlist)):  # 地物光谱画图
        #     plt.plot(WL_EMISSlist[i][0], WL_EMISSlist[i][1])
        # plt.show()

        return WL_EMISSlist

    def _inter2ChannelEmiss(self,SRF, WL_EMISS):  # 对光谱曲线进行插值，并求积分,得到通道发射率
        # return  np.sum(interp1d(WL_EMISS[0], WL_EMISS[1],bounds_error=False, fill_value=0)(SRF[0]) * SRF[1]) / np.sum(SRF[1])
        return np.sum(np.interp(SRF[0], WL_EMISS[0], WL_EMISS[1]) * SRF[1]) / np.sum(SRF[1])

    def stdEmiss(self):
        WL_EMISSlist = self.readEmissivityfile()
        channelEmiss = []  # 存储各个地物不同通道的发射率
        channelEmiss_std = []  # 存储各个地物通道发射率的标准差
        channelEmiss_max = []  # 存储各个通道的发射率最大值

        for wl_emiss in WL_EMISSlist:  # 每个地物
            try:
                temp = []  # 存储通道发射率
                for SRF in self.SRFlist:  # 每个波段
                    temp.append(self._inter2ChannelEmiss(SRF, wl_emiss))
                channelEmiss_std.append(np.std(temp, ddof=1))
                channelEmiss.append(temp)
                channelEmiss_max.append(np.max(temp))
            except Exception as e:
                print(e)

        print('The number of sample:', len(channelEmiss))
    # 得到所有的标准差和最大发射率
        return channelEmiss_std, channelEmiss_max, np.array(channelEmiss)


def read_srf_MODIS(filePath):
    """
    读取MODIS的波段响应函数文件
    :param filePath: 文件路径
    :return: 响应函数数组
    """
    with open(filePath, 'r') as file:
        # 读取文件中的所有内容
        lines = file.readlines()
        # 记录光谱响应函数的二维数组：（波长，响应率）
        SRF = np.empty((2, len(lines)), dtype=float, order='C')
        for i in range(len(lines)):
            curItems = lines[i].split()
            SRF[0][i] = float(curItems[0])
            SRF[1][i] = float(curItems[1])
    print(SRF)
    return SRF


def read_srf_GF5(filePath):
    """
    读取GF5的波段响应函数文件
    :param filePath:
    :return:
    """
    with open(filePath, 'r') as file:
        # 读取文件中的所有内容
        lines = file.readlines()
        # GF5文件中多了一行
        lines = lines[1:]
        # 记录光谱响应函数的二维数组：（波长，响应率）
        SRF = np.empty((2, len(lines)), dtype=float, order='C')
        for i in range(len(lines)):
            curItems = lines[i].split()
            SRF[0][i] = float(curItems[0])
            SRF[1][i] = float(curItems[1])
    print(SRF)
    return SRF


def fit_bands(x, y):
    """
    对两个波段的发射率进行拟合，默认线性
    :param x: 第一个波段的发射率
    :param y: 第二个波段的发射率
    :return: 拟合的方程
    """
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    return p


def main():
    # 读取光谱响应函数文档，获取两个传感器的响应函数
    GF5_SRF_path = 'SRF/GF5_band4.txt'
    MODIS_SRF_path = 'SRF/MODIS/MODIS29.txt'
    GF5_SRF = read_srf_GF5(GF5_SRF_path)
    MODIS_SRF = read_srf_MODIS(MODIS_SRF_path)
    # 获取发射率库中的样本发射率，并根据响应函数计算为两个传感器的通道发射率
    emiss_lab_path = 'Emissivity'
    simu_GF5 = SimuBandEmiss(emiss_lab_path, [GF5_SRF])
    std, max, emiss_lab_GF5 = simu_GF5.stdEmiss()
    simu_MODIS = SimuBandEmiss(emiss_lab_path, [MODIS_SRF])
    std_, max_, emiss_lab_MODIS = simu_MODIS.stdEmiss()
    print(emiss_lab_GF5)
    print(emiss_lab_MODIS)
    # 对两个通道发射率进行拟合，输出拟合的线性关系
    print(fit_bands(emiss_lab_MODIS[:, 0], emiss_lab_GF5[:, 0]))
    print(fit_bands(emiss_lab_GF5[:, 0], emiss_lab_MODIS[:, 0]))


if __name__ == '__main__':
    main()