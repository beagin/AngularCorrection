# code for simulation experiment

import pandas as pd
import numpy as np
from enum import Enum
from scipy import stats, misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import metrics
from PIL import Image
from osgeo import gdal, gdalconst
import math
from code_2020.kernals import Ross_thick, LI_SparseR
import random
import warnings
warnings.filterwarnings("ignore")   # 忽略warning


# ****************************************** 一些声明 **************************************
# ASTER
# 2327
# file_LST_ASTER_hdf = "data/ASTER/AST_08_00307062019032327_20211109202424_8804.hdf"
# file_refl_ASTER = "data/ASTER/AST_07XT_00307062019032327_20211109044429_8561.hdf"
# file_SE_ASTER = "data/ASTER/AST_05_00307062019032327_20211109202540_19729.hdf"
# 2318
file_LST_ASTER_hdf = "data/ASTER/AST_08_00307062019032318_20211109202424_8809.hdf"
file_refl_ASTER = "data/ASTER/AST_07XT_00307062019032318_20211109044429_8560.hdf"
file_SE_ASTER = "data/ASTER/AST_05_00307062019032318_20211109202540_19735.hdf"
# 2309
# file_LST_ASTER_hdf = "data/ASTER/AST_08_00307062019032309_20211109202424_8812.hdf"
# file_refl_ASTER = "data/ASTER/AST_07XT_00307062019032309_20211109044449_8628.hdf"
# file_SE_ASTER = "data/ASTER/AST_05_00307062019032309_20211109202540_19741.hdf"

# MOD09
file_MOD09_1 = "data/MODIS/MOD09GA.sur_refl_b01_1.tif"
file_MOD09_2 = "data/MODIS/MOD09GA.sur_refl_b02_1.tif"
file_MOD09_SZA = "data/MODIS/MOD09GA.SolarZenith_1.tif"
file_MOD09_SAA = "data/MODIS/MOD09GA.SolarAzimuth_1.tif"
file_MOD09_VZA = "data/MODIS/MOD09GA.SensorZenith_1.tif"
# MOD15
file_MOD15 = "data/MODIS/LAI_2019185.Lai_500m.tif"

# 聚集指数
file_CI = "data/CI/CI_2019185.tif"
# 查找表
file_LUT = "SRF/LUT12.txt"

# 判断植被/土壤的NDVI阈值
threshold_NDVI = 0.45
threshold_NDVI_min = 0.3
# 各波段平均组分发射率
# 2318
SEs_aver = [0.9556206,0.95971876,0.9583313,0.96373886,0.9579294,0.96535045,0.9737943,0.97538644,0.96828395,0.9703193]
# 2309
# SEs_aver = [0.95921916,0.96219736,0.96382433,0.96685,0.9655643,0.9693147,0.9739296,0.9761412,0.9667597,0.9716622]
# 2327
# SEs_aver = [0.95891243, 0.96218306, 0.9610082, 0.9646352, 0.961338, 0.96575534, 0.9749102, 0.975972, 0.9701367, 0.9715742]


# ****************************************** 文件操作 **************************************
# <editor-fold> 文件操作

def open_tiff(filePath:str):
    """
    To read a .tif or .tiff file and get the data
    :param filePath:
    :return:
    """
    # try:
    dataset = gdal.Open(filePath)
    if dataset == None:
        print(filePath + "文件无法打开")
        return
    im_width = dataset.RasterXSize      # 栅格矩阵的列数
    im_height = dataset.RasterYSize     # 栅格矩阵的行数
    im_bands = dataset.RasterCount      # 波段数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    im_geotrans = dataset.GetGeoTransform() # 获取仿射矩阵信息
    im_proj = dataset.GetProjection()       # 获取投影信息:6326
    # print(im_data.dtype)
    # print(im_width)
    # print(im_height)
    # print(im_bands)
    # print(im_data.shape)
    # print(len(im_data[0]))
    # print(im_geotrans)
    # print(im_proj)
    # print(dataset.__dir__())
    # print(dataset.GetMetadata())
    # except Exception as e:
    #     print(e)

    return dataset, im_data


def open_gdal(fileName):
    """
    Func: To open a gdal file
    :param fileName:
    :return: SDS
    """
    hdf = gdal.Open(fileName)
    # subDataset
    subdatasets = hdf.GetSubDatasets()
    # for subDataset in subdatasets:
    #     print(subDataset)
    # 查看元数据
    metadata = hdf.GetMetadata()
    # print(metadata)
    # print(metadata.keys())
    # SDSs
    sdsdict = hdf.GetMetadata('SUBDATASETS')
    sdslist = [sdsdict[k] for k in sdsdict.keys() if '_NAME' in k]
    sds = []
    for n in sdslist:
        sds.append(gdal.Open(n))
    # print(len(sds))

    return sds, metadata


def get_LUT(fileName):
    """
    Func: to get the lookup table and save in a list
    :param fileName: file name of the LUT text file
    :return: the list of LUT
    """
    with open(fileName, "r") as file:
        lines = file.readlines()
        LUTlist = [float(lines[i].split()[1]) for i in range(1, len(lines))]
        # print(LUTlist)
    return LUTlist


def write_tiff(data: np.ndarray, filename: str):
    """

    :param data:
    :param filename:
    :return:
    """
    Image.fromarray(data).save("pics/" + filename + '.tif')
    # misc.imsave(filename + ".tif", data)

# </editor-fold>

# ****************************************** 计算函数 **************************************
# <editor-fold> 计算函数

def cal_NDVI(data_red, data_nir):
    """
    To calculate the NDVI of each pixel and return the result.
    :param data_red:
    :param data_nir:
    :return:
    """
    result = ((data_nir - data_red) / (data_nir + data_red))
    # 进行异常值处理
    if type(data_red) == np.ndarray:
        result[np.isnan(result)] = 0
        result[result > 1] = 1
        result[result < 0] = 0
    # check the histogram
    #     hist, edges = np.histogram(result, 100)
    #     print(hist)
    #     print(edges)
    #
    #     display(result, "NDVI")
    return result


def cal_fvc(ndvi, NDVIv=0.86, NDVIs=0.156):
    """
    Func: to calculate fvc from NDVI
    :param ndvi:
    :return:
    """
    # print(ndvi)
    fvc = ((ndvi - NDVIs) / (NDVIv - NDVIs)) ** 2
    # 异常值
    fvc[np.isnan(fvc)] = 0
    fvc[np.isinf(fvc)] = 0
    fvc[fvc > 1] = 1
    fvc[fvc < 1e-7] = 0
    # check the histogram
    # hist, edges = np.histogram(fvc, 100)
    # print(hist)
    # print(edges)

    # print(np.min(fvc))

    return fvc


def cal_fvc_gap(LAI, omega, theta, G=0.5):
    """
    根据间隙率模型计算FVC，默认G为0.5，其他各变量都为相同大小的数组，返回结果也为一个数组
    :param LAI:
    :param omega:
    :param theta: 度数，需转换为角度
    :param G:
    :return:
    """
    return 1-np.exp(-LAI * omega * G / np.cos(theta*math.pi/180))


def cal_ref_BRDF(SZA, VZA, iso, vol, geo):
    """
    根据BRDF模型计算一个波段的反射率
    :param SZA:
    :param SAA:
    :param VZA:
    :param VAA:
    :param brdf_1:
    :param brdf_2:
    :param brdf_3:
    :return:
    """
    shape = SZA.shape
    result = [[] for i in range(shape[0])]
    result_vol = [[] for i in range(shape[0])]
    result_geo = [[] for i in range(shape[0])]
    for i in range(shape[0]):
        for j in range(shape[1]):
            # rAzimuth = (SAA[i, j] - VAA[i, j]) % 180
            rAzimuth = 0
            sza = 60 / 180 * math.pi
            # Kvol = Ross_thick(sza, VZA[i, j], rAzimuth)
            Kvol = Ross_thick(SZA[i, j] * math.pi / 180, VZA[i, j], rAzimuth)
            Kgeo = LI_SparseR(SZA[i, j] * math.pi / 180, VZA[i, j], rAzimuth)
            # Kgeo = LI_SparseR(sza, VZA[i, j], rAzimuth)
            result_vol[i].append(Kvol)
            result_geo[i].append(Kgeo)
            ref = iso[i, j] + vol[i, j] * Kvol + geo[i, j] * Kgeo
            result[i].append(ref)

    result_vol = np.array(result_vol)
    result_geo = np.array(result_geo)
    result = np.array(result)
    # 特殊值处理
    result_vol[np.abs(result_vol) > 1000] = 0
    result_geo[np.abs(result_geo) > 1000] = 0
    iso[np.abs(iso) > 1000] = 0
    result[np.abs(result) > 1000] = 0

    # print(np.min(result_vol))
    # print(np.min(result_geo))
    # print(np.min(iso))
    # print(np.max(result_vol))
    # print(np.max(result_geo))
    # print(np.max(iso))
    # display(result_vol, "VOL_4")
    # display(result_geo, "GEO_4")
    # display(np.array(iso), "ISO_2")

    # display(result, "BRDF_band1_2")
    return np.array(result)


def cal_index(base, interval, target):
    """
    给定真实（ASTER）坐标值，找到其在MODIS图像中的索引
    :param base:
    :param interval:
    :param target:
    :return:
    """
    return int((target - base) / interval)


def cal_mean_LSTvs():
    """
    计算ASTER图像中所有植被/土壤像元的LST均值
    :return:
    """
    sds_aster, _ = open_gdal(file_LST_ASTER_hdf)
    lst_aster = sds_aster[3].ReadAsArray() * 0.1
    lst_aster[lst_aster < 250] = 250
    sds_ref_aster, _ = open_gdal(file_refl_ASTER)
    ref_red_aster = sds_ref_aster[1].ReadAsArray() * 0.001
    ref_nir_aster = sds_ref_aster[2].ReadAsArray() * 0.001
    ndvi = cal_NDVI(ref_red_aster, ref_nir_aster)
    # 遍历所有像元计算
    LSTs = []
    LSTv = []
    num_veg = 0
    num_soil = 0
    for y in range(lst_aster.shape[0]):
        for x in range(lst_aster.shape[1]):
            # 去除边缘无数据点与云
            if lst_aster[y, x] > 285:
                # 获取对应的6*6个NDVI值，用平均值代替
                cur_ndvi = np.mean(ndvi[y*6:y*6+5, x*6:x*6+5])
                # if cur_ndvi > threshold_NDVI:
                #     num_veg += 1
                #     LSTv.append(lst_aster[y, x])
                # else:
                #     num_soil += 1
                #     LSTs.append(lst_aster[y, x])

                # 计算更极端的端元温度值
                if cur_ndvi > 0.66:
                    num_veg += 1
                    LSTv.append(lst_aster[y, x])
                elif cur_ndvi < 0.08:
                    num_soil += 1
                    LSTs.append(lst_aster[y, x])

    print(np.mean(LSTs))
    print(np.mean(LSTv))
    print(num_soil)
    print(num_veg)


def cal_mean_SEvs():
    sds_se_aster, _ = open_gdal(file_SE_ASTER)
    SE_10 = sds_se_aster[0].ReadAsArray() * 0.001
    SE_11 = sds_se_aster[1].ReadAsArray() * 0.001
    SE_12 = sds_se_aster[2].ReadAsArray() * 0.001
    SE_13 = sds_se_aster[3].ReadAsArray() * 0.001
    SE_14 = sds_se_aster[4].ReadAsArray() * 0.001
    sds_ref_aster, _ = open_gdal(file_refl_ASTER)
    ref_red_aster = sds_ref_aster[1].ReadAsArray() * 0.001
    ref_nir_aster = sds_ref_aster[2].ReadAsArray() * 0.001
    ndvi = cal_NDVI(ref_red_aster, ref_nir_aster)
    # 遍历所有像元计算
    LSTs = []
    LSTv = []
    SEv_10 = []
    SEs_10 = []
    SEv_11 = []
    SEs_11 = []
    SEv_12 = []
    SEs_12 = []
    SEv_13 = []
    SEs_13 = []
    SEv_14 = []
    SEs_14 = []
    num_veg = 0
    num_soil = 0
    for y in range(SE_10.shape[0]):
        for x in range(SE_10.shape[1]):
            # 去除边缘无数据点与云
            if SE_10[y, x] > 0.5:
                # 获取对应的6*6个NDVI值，用平均值代替
                cur_ndvi = np.mean(ndvi[y * 6:y * 6 + 5, x * 6:x * 6 + 5])
                # if cur_ndvi > threshold_NDVI:
                #     num_veg += 1
                #     LSTv.append(lst_aster[y, x])
                # else:
                #     num_soil += 1
                #     LSTs.append(lst_aster[y, x])

                # 计算更极端的端元温度值
                if cur_ndvi > 0.6:
                    num_veg += 1
                    SEv_10.append(SE_10[y, x])
                    SEv_11.append(SE_11[y, x])
                    SEv_12.append(SE_12[y, x])
                    SEv_13.append(SE_13[y, x])
                    SEv_14.append(SE_14[y, x])
                elif cur_ndvi < 0.08:
                    num_soil += 1
                    SEs_10.append(SE_10[y, x])
                    SEs_11.append(SE_11[y, x])
                    SEs_12.append(SE_12[y, x])
                    SEs_13.append(SE_13[y, x])
                    SEs_14.append(SE_14[y, x])

    print(num_soil)
    print(num_veg)
    print(np.mean(SEs_10))
    print(np.mean(SEv_10))
    print(np.mean(SEs_11))
    print(np.mean(SEv_11))
    print(np.mean(SEs_12))
    print(np.mean(SEv_12))
    print(np.mean(SEs_13))
    print(np.mean(SEv_13))
    print(np.mean(SEs_14))
    print(np.mean(SEv_14))


def getEdges(lst: np.ndarray, ndvi: np.ndarray):
    """
    To get the dry and wet edge from the LST-NDVI figure
    :return:
    """
    # average and standard deviation of all lsts
    LST_aver = lst.mean()
    LST_std = np.std(lst)
    print("LST_aver: " + str(LST_aver))
    print("LST_std: " + str(LST_std))

    # divide the NDVI into intervals, 10 * 8 subintervals
    interval_num = 20
    subinterval_num = 6
    # do the statics
    Ts = [[[] for j in range(subinterval_num)] for i in range(interval_num)]
    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            if ndvi[i, j] <= 0 or ndvi[i, j] >= 1:
                continue
            if lst[i, j] >= LST_aver + 2.5 * LST_std or lst[i, j] <= LST_aver - 2.5 * LST_std:
                continue
            index = int(ndvi[i, j] * interval_num * subinterval_num)
            Ts[int(index / subinterval_num)][index % subinterval_num].append(lst[i, j])
    # delete those lst below 270
    Ts_  = [[[] for j in range(subinterval_num)] for i in range(interval_num)]
    for i in range(interval_num):
        for j in range(subinterval_num):
            mean_all = np.mean(Ts[i][j])
            dev_all = np.std(Ts[i][j], ddof=1)
            # print(mean_all)
            # print(dev_all)
            for x in Ts[i][j]:
                # LST
                if x > mean_all + 2.5 * dev_all or x < mean_all - 3 * dev_all:
                    continue
                # LST
                if x > 260:
                    Ts_[i][j].append(x)

    # for each interval, get the Tmax_aver
    Tmax_aver = []
    Tmin_aver = []
    for i in range(interval_num):
        # print("Ts[" + str(i) + "]")
        # max Ts of subintervals in this interval
        maxTs = []
        minTs =[]
        for j in range(subinterval_num):
            if len(Ts_[i][j]) > 0:
                maxTs.append(max(Ts_[i][j]))
                minTs.append(min(Ts_[i][j]))
        # print("maxTs:")
        # print(maxTs)
        # print("minTs:")
        # print(minTs)

        if len(maxTs) == 0:
            if len(Tmax_aver) > 0:
                Tmax_aver.append(Tmax_aver[-1])
        else:
            while True:
                average_max = np.mean(maxTs)
                dev_max = np.std(maxTs, ddof=1)
                discard_max = False
                # if exist one max Ts less than ..., then discard it
                for j in range(len(maxTs)):
                    if maxTs[j] < average_max - dev_max:
                        maxTs.pop(j)
                        discard_max = True
                        break
                if not discard_max:
                    break
            Tmax_aver.append(np.mean(maxTs))

        # Tmin
        # same method
        if len(minTs) == 0:
            if len(Tmin_aver) > 0:
                Tmin_aver.append(Tmin_aver[-1])
        else:
            while True:
                average_min = np.mean(minTs)
                # print("aver_min: " + str(average_min))
                dev_min = np.std(minTs, ddof=1)
                # print("dev_min: " + str(dev_min))
                discard_min = False
                for j in range(len(minTs)):
                    if minTs[j] > average_min + dev_min:
                        minTs.pop(j)
                        discard_min = True
                        break
                if not discard_min:
                    break
            Tmin_aver.append(np.mean(minTs))

    print(Tmax_aver)
    print(Tmin_aver)

    # # do linear regression
    # A = np.vstack([ndvi_list, np.ones(len(ndvi_list))]).T
    # Tmax_aver = np.array(Tmax_aver)     # LST值（y轴）
    # Tmin_aver = np.array(Tmin_aver)
    # # 方法1：np.linalg.lstsq
    # k1, c1 = np.linalg.lstsq(A, Tmax_aver)[0]
    # # 方法2：np.polyfit
    # result = np.polyfit(ndvi_list, Tmax_aver, 1)
    # 方法3：scipy.stats.linregress

    # k1, c1, r_value, p_value, std_err = stats.linregress(ndvi_list, Tmax_aver)
    # k2, c2, _, __, ___ = stats.linregress(ndvi_list, Tmin_aver)
    # print(k1, c1)
    # print(k2, c2)
    # # print(result)

    # Tmax
    ndvi_list = np.array([(0.5/interval_num + i /interval_num) for i in range(interval_num)])     # ndvi值（x轴）
    while True:
        # do linear regression
        k1, c1, r_value, p_value, std_err = stats.linregress(ndvi_list, np.array(Tmax_aver))
        y = k1 * ndvi_list + c1
        # calculate RMSE
        RMSE = np.sqrt(metrics.mean_squared_error(np.array(Tmax_aver), y))
        # do discard
        discard_max = False
        for i in range(len(ndvi_list)):
            if y[i] - 2 * RMSE > Tmax_aver[i] or Tmax_aver[i] > y[i] + 2 * RMSE:
                Tmax_aver.pop(i)
                ndvi_list = np.delete(ndvi_list, i)
                discard_max = True
                break
        if not discard_max:
            break

    # Tmin
    ndvi_list = np.array([(0.5/interval_num + i /interval_num) for i in range(interval_num)])     # ndvi值（x轴）
    while True:
        # do linear regression
        # print(ndvi_list)
        # print(Tmin_aver)
        k2, c2, r_value, p_value, std_err = stats.linregress(ndvi_list, np.array(Tmin_aver))
        y = k2 * ndvi_list + c2
        # calculate RMSE
        RMSE = np.sqrt(metrics.mean_squared_error(np.array(Tmin_aver), y))
        # do discard
        discard_min = False
        for i in range(len(ndvi_list)):
            if y[i] + 2 * RMSE < Tmin_aver[i] or Tmin_aver[i] < y[i] - 2 * RMSE:
                Tmin_aver.pop(i)
                ndvi_list = np.delete(ndvi_list, i)
                discard_min = True
                break
        if not discard_min:
            break

    return k1, c1, k2, c2


def getEdges_fvc(BT: np.ndarray, fvc: np.ndarray):
    """
    To get the dry and wet edge from the LST-NDVI figure
    :return:
    """
    print("func get edges")
    # 去掉FVC过大的值
    BT = BT[fvc < 0.995]
    fvc = fvc[fvc < 0.995]

    # average and standard deviation of all lsts
    BT_aver = BT.mean()
    BT_std = np.std(BT)
    print("BT_aver: " + str(BT_aver))
    print("BT_std: " + str(BT_std))

    # divide the FVC into intervals, 10 * 8 subintervals
    interval_num = 10
    subinterval_num = 5

    # 划分至
    BTs = [[[] for j in range(subinterval_num)] for i in range(interval_num)]
    for i in range(BT.shape[0]):
        # 去除异常值
        if fvc[i] <= 1e-2 or fvc[i] >= 1:
            continue
        if BT[i] <= 0 or BT[i] >= BT_aver + 10 * BT_std or BT[i] <= BT_aver - 4 * BT_std:
            continue
        index = int(fvc[i] * interval_num * subinterval_num)
        BTs[int(index / subinterval_num)][index % subinterval_num].append(BT[i])
    # 第一次筛选：去除异常值
    Ts_  = [[[] for j in range(subinterval_num)] for i in range(interval_num)]
    for i in range(interval_num):
        for j in range(subinterval_num):
            mean_all = np.mean(BTs[i][j])
            dev_all = np.std(BTs[i][j], ddof=1)
            # print(mean_all)
            # print(dev_all)
            for x in BTs[i][j]:
                if x > mean_all + 4 * dev_all or x < mean_all - 4 * dev_all:
                    continue
                Ts_[i][j].append(x)

    # 记录每个区间的最值
    Tmax_aver = []
    Tmin_aver = []
    # 区间对应的NDVI值
    for i in range(interval_num):
        # print("Ts[" + str(i) + "]")
        # max Ts of subintervals in this interval
        maxTs = []
        minTs =[]
        # 计算每个子区间的最大最小值
        for j in range(subinterval_num):
            if len(Ts_[i][j]) > 0:
                maxTs.append(max(Ts_[i][j]))
                minTs.append(min(Ts_[i][j]))
        # print("maxTs:")
        # print(maxTs)
        # print("minTs:")
        # print(minTs)
        # 当前区间没有值且前一区间有值，则直接用前一区间的值
        # TODO：改这一部分，目前每一个区间都需要有有效值
        if len(maxTs) == 0 and len(Tmax_aver) > 0:
            Tmax_aver.append(Tmax_aver[-1])
        # 对当前这一区间，筛选每个子区间的最值
        else:
            while True:
                average_max = np.mean(maxTs)
                dev_max = np.std(maxTs, ddof=1)
                discard_max = False
                # if exist one max Ts less than ..., then discard it
                # 每个剩余的子区间值进行判断
                for j in range(len(maxTs)):
                    if maxTs[j] < average_max - dev_max:
                        maxTs.pop(j)
                        discard_max = True
                        break
                # 遍历了一遍都没有discard
                if not discard_max:
                    break
            Tmax_aver.append(np.mean(maxTs))

        # Tmin
        # same method
        if len(minTs) == 0 and len(Tmin_aver) > 0:
            Tmin_aver.append(Tmin_aver[-1])
        else:
            while True:
                average_min = np.mean(minTs)
                # print("aver_min: " + str(average_min))
                dev_min = np.std(minTs, ddof=1)
                # print("dev_min: " + str(dev_min))
                discard_min = False
                for j in range(len(minTs)):
                    if minTs[j] > average_min + dev_min:
                        minTs.pop(j)
                        discard_min = True
                        break
                if not discard_min:
                    break
            Tmin_aver.append(np.mean(minTs))

    # Tmax
    ndvi_list = np.array([(0.5/interval_num + i /interval_num) for i in range(interval_num)])
    # # 2318 B10
    # Tmax_aver[0] = 13
    # Tmax_aver[-1] = 9.25
    # # 2309
    # Tmax_aver[0] = 9.6
    # Tmax_aver[-1] = 8.86
    # 2327
    Tmax_aver[0] = 11.6
    Tmax_aver[-1] = 10.05

    print(ndvi_list)
    # ndvi值（x轴）
    while True:
        # do linear regression
        k1, c1, r_value, p_value, std_err = stats.linregress(ndvi_list, np.array(Tmax_aver))
        y = k1 * ndvi_list + c1
        # calculate RMSE
        RMSE = np.sqrt(metrics.mean_squared_error(np.array(Tmax_aver), y))
        # do discard
        discard_max = False
        for i in range(len(ndvi_list)):
            if y[i] - 3 * RMSE > Tmax_aver[i] or Tmax_aver[i] > y[i] + 3 * RMSE:
                Tmax_aver.pop(i)
                ndvi_list = np.delete(ndvi_list, i)
                discard_max = True
                break
        if not discard_max:
            break

    # Tmin
    ndvi_list = np.array([(0.5/interval_num + i /interval_num) for i in range(interval_num)])
    # 2318 B10
    # Tmin_aver[-1] = 8.1
    # 2318 B14
    # Tmin_aver[0] = 8.7
    # # 2309
    # Tmin_aver[0] = 8.0
    # Tmin_aver[-1] = 7.36
    # 2327
    Tmin_aver[-1] = 9.8

    # ndvi值（x轴）
    while True:
        # do linear regression
        # print(ndvi_list)
        # print(Tmin_aver)
        k2, c2, r_value, p_value, std_err = stats.linregress(ndvi_list, np.array(Tmin_aver))
        y = k2 * ndvi_list + c2
        # calculate RMSE
        RMSE = np.sqrt(metrics.mean_squared_error(np.array(Tmin_aver), y))
        # do discard
        discard_min = False
        for i in range(len(ndvi_list)):
            if y[i] + 3 * RMSE < Tmin_aver[i] or Tmin_aver[i] < y[i] - 3 * RMSE:
                Tmin_aver.pop(i)
                ndvi_list = np.delete(ndvi_list, i)
                discard_min = True
                break
        if not discard_min:
            break

    print(Tmax_aver)
    print(Tmin_aver)

    return k1, c1, k2, c2


def cal_vertex(k1, c1, k2, c2):
    """
    Func: to calculate the crossover point in Ts-NDVI triangle
    :param k1:
    :param c1:
    :param k2:
    :param c2:
    :return:
    """
    NDVI = (c2 - c1) / (k1 - k2)
    LST = (k1*c2 - c1*k2) / (k1 - k2)
    return NDVI, LST


def cal_params(BT1, fvc1, BT2, fvc2):
    """
    用于计算特征空间中两点的直线方程参数
    :param BT1:
    :param fvc1:
    :param BT2:
    :param fvc2:
    :return: 斜率与截距
    """
    slope = (BT1 - BT2) / (fvc1 - fvc2)
    const = BT1 - fvc1 * slope
    return slope, const


def calRMSE(lst: np.ndarray, lst_0: np.ndarray, VZA: np.ndarray, title: str):
    """
    计算一组校正前后的lst数据在不同角度下的RMSE，并绘制图像及保存
    :param lst:
    :param lst_0:
    :return:
    """
    print("func calRMSE")
    maxVZA = int(np.max(VZA) + 0.5)
    minVZA = int(np.min(VZA) + 0.5)

    interval_num = maxVZA - minVZA + 1
    lst_intervals = [[] for i in range(interval_num)]
    lst0_intervals = [[] for i in range(interval_num)]
    diff_intervals = [[] for i in range(interval_num)]

    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            try:
                index = int(VZA[i, j] + 0.5) - minVZA
                lst_intervals[index].append(lst[i, j])
                lst0_intervals[index].append(lst_0[i, j])
                diff_intervals[index].append(np.abs(lst[i, j] - lst_0[i, j]))
            except Exception as e:
                print(e)
                print(VZA[i, j])
                print(index)

    RMSEs = []
    for i in range(len(lst_intervals)):
        # print(i)
        # print(len(lst_intervals[i]))
        if len(lst_intervals[i]) == 0:
            RMSEs.append(RMSEs[-1])
        else:
            RMSE = np.sqrt(metrics.mean_squared_error(np.array(lst0_intervals[i]), np.array(lst_intervals[i])))
            # print(RMSE)
            RMSEs.append(RMSE)

    x = range(minVZA, maxVZA + 1, 1)
    # print(x)
    plt.plot(x, RMSEs)
    plt.xlabel("VZA")
    plt.ylabel("RMSE")
    plt.savefig("pics/" + title + "_RMSE_VZA_1.png")
    plt.show()


def calRMSE_new(lst, lst_0, VZA, title: str):
    """
    计算一组校正前后的lst数据在不同角度下的RMSE，并绘制图像及保存
    :param lst:
    :param lst_0:
    :return:
    """
    print("func calRMSE")
    # 打开文件
    _, lst = open_tiff(lst)
    _, lst_0 = open_tiff(lst_0)
    _, VZA = open_tiff(VZA)
    print(np.max(VZA))

    maxVZA = int(np.max(VZA) + 0.5)
    maxVZA = 62
    minVZA = int(np.min(VZA) + 0.5)
    minVZA = 56

    interval_num = maxVZA - minVZA + 1
    lst_intervals = [[] for i in range(interval_num)]
    lst0_intervals = [[] for i in range(interval_num)]
    diff_intervals = [[] for i in range(interval_num)]

    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            try:
                index = int(VZA[i, j] + 0.5) - minVZA
                lst_intervals[index].append(lst[i, j])
                lst0_intervals[index].append(lst_0[i, j])
                diff_intervals[index].append(np.abs(lst[i, j] - lst_0[i, j]))
            except Exception as e:
                print(e)
                print(VZA[i, j])
                print(index)

    RMSEs = []
    nums = []
    for i in range(len(lst_intervals)):
        # print(i)
        # print(len(lst_intervals[i]))
        if len(lst_intervals[i]) == 0:
            RMSEs.append(RMSEs[0])
            nums.append(0)
        else:
            RMSE = np.sqrt(metrics.mean_squared_error(np.array(lst0_intervals[i]), np.array(lst_intervals[i])))
            # print(RMSE)
            RMSEs.append(RMSE)
            nums.append(len(lst_intervals[i]))

    x = range(minVZA, maxVZA + 1, 1)
    # print(x)
    # plt.plot(x, RMSEs, marker='o', markersize=2)
    plt.plot(x, RMSEs)
    plt.xlabel("VZA")
    plt.ylabel("RMSE")
    plt.savefig("pics/" + title + "_RMSE_VZA_1.png")
    plt.show()
    plt.plot(x, nums)
    plt.xlabel("VZA")
    plt.ylabel("pixel num")
    plt.savefig("pics/pixelNum.png")
    plt.show()


def cal_windowLSTsv(window=5):
    """
    计算一定窗口大小的组分温度
    :return:
    """
    _, LSTs_all = open_tiff("pics/LSTs_all.tif")
    _, LSTv_all = open_tiff("pics/LSTv_all.tif")

    # 计算5*5的窗口内的组分温度（取极值）
    LSTs_window = np.zeros(LSTs_all.shape)
    LSTv_window = np.zeros(LSTv_all.shape)
    for y in range(LSTs_window.shape[0]):
        for x in range(LSTs_window.shape[1]):
            # 当前像元对应的窗口索引
            minY = max(y - int(window/2), 0)
            maxY = min(y + int((window+1)/2), LSTs_window.shape[0])
            minX = max(x - int(window/2), 0)
            maxX = min(x + int((window+1)/2), LSTs_window.shape[1])
            cur_LSTs = LSTs_all[minY:maxY, minX:maxX]
            cur_LSTv = LSTv_all[minY:maxY, minX:maxX]
            # 计算当前窗口的组分温度极值
            LSTs_window[y, x] = np.max(cur_LSTs)
            if len(cur_LSTv[cur_LSTv > 0]) > 0:
                LSTv_window[y, x] = np.min(cur_LSTv[cur_LSTv > 0])
    write_tiff(LSTs_window, "LSTs_window")
    write_tiff(LSTv_window, "LSTv_window")

    # 组分温度差值直方图
    diff = LSTs_window-LSTv_window
    diff = diff[diff != 0]
    display_hist(diff[np.abs(diff) < 50], "diff_LSTsv_window")


def get_mean_SE():
    for band in range(10, 15):
        _, SEs = open_tiff("pics/SEs_up_" + str(band) + ".tif")
        _, SEv = open_tiff("pics/SEv_up_" + str(band) + ".tif")
        SEs = SEs[SEs > 0]
        SEv = SEv[SEv > 0]
        print(np.mean(SEs), np.mean(SEv))


def up_sample():
    """
    对各种数据进行上采样，由0.005到0.01degree分辨率
    :return:
    """
    # 打开数据
    _, LSTv = open_tiff("pics/LSTv_all.tif")
    _, LSTs = open_tiff("pics/LSTs_all.tif")
    _, FVC_60 = open_tiff("pics/FVC.tif")
    _, FVC_0 = open_tiff("pics/FVC_0.tif")
    _, is_valid = open_tiff("pics/is_valid.tif")

    # 上采样操作
    shape = LSTs.shape
    print(shape)
    new_shape = (int(shape[0] / 2), int(shape[1] / 2))
    new_LSTv = np.zeros(new_shape, dtype=np.float64)
    new_LSTs = np.zeros(new_shape, dtype=np.float64)
    new_FVC60 = np.zeros(new_shape, dtype=np.float64)
    new_FVC0 = np.zeros(new_shape, dtype=np.float64)
    new_valid = np.zeros(new_shape, dtype=np.float64)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            # 取范围内的有效值
            cur_valid = is_valid[i*2:i*2+2, j*2:j*2+2]
            # 当前范围存在有效值
            if np.mean(cur_valid) > 0:
                # 其他数据取平均值
                new_FVC60[i, j] = np.mean((FVC_60[i*2:i*2+2, j*2:j*2+2])[cur_valid > 0])
                new_FVC0[i, j] = np.mean((FVC_0[i*2:i*2+2, j*2:j*2+2])[cur_valid > 0])
                new_valid[i, j] = 1
                # 组分LST需进行特殊处理
                new_LSTs[i, j] = np.max(LSTs[i*2:i*2+2, j*2:j*2+2])
                new_LSTv[i, j] = np.min((LSTv[i*2:i*2+2, j*2:j*2+2])[cur_valid > 0])    # 尽管是取极值，也需要先进行有效筛选，否则会是0
            else:
                pass

    # 存储上采样后的数据
    write_tiff(new_LSTv, "LSTv_up")
    write_tiff(new_LSTs, "LSTs_up")
    write_tiff(new_FVC60, "FVC_60_up")
    write_tiff(new_FVC0, "FVC_0_up")

    # 对每个波段的等效发射率进行处理
    for band in range(10, 15):
        _, SEs = open_tiff("pics/SEs_aver_" + str(band) + ".tif")
        _, SEv = open_tiff("pics/SEv_aver_" + str(band) + ".tif")
        new_SEs = np.zeros(new_shape, dtype=np.float64)
        new_SEv = np.zeros(new_shape, dtype=np.float64)
        for i in range(new_shape[0]):
            for j in range(new_shape[1]):
                cur_valid = is_valid[i*2:i*2+2, j*2:j*2+2]
                if np.mean(cur_valid) > 0:
                    cur_SEs = SEs[i*2:i*2+2, j*2:j*2+2]
                    cur_SEv = SEv[i*2:i*2+2, j*2:j*2+2]
                    new_SEs[i, j] = np.mean(cur_SEs[cur_SEs > 0])
                    new_SEv[i, j] = np.mean(cur_SEv[cur_SEv > 0])
                    if new_SEs[i, j] < 0.5:
                        print(cur_valid)
                        print((SEs[i*2:i*2+2, j*2:j*2+2])[cur_valid > 0])
        write_tiff(new_SEs, "SEs_up_" + str(band))
        write_tiff(new_SEv, "SEv_up_" + str(band))
    write_tiff(new_valid, "is_valid_up")

# </editor-fold>    计算

# ****************************************** 其他函数 **************************************
# <editor-fold> 其他


def lst2BTs(lst, band=12):
    """
    transform lst data into BTs data
    :param lst:
    :return:
    """
    fileName = "SRF/LUT" + str(band) + ".txt"
    LUT = get_LUT(fileName)
    if type(lst) == np.ndarray:
        shape = lst.shape
        BTs = np.zeros(shape, dtype=np.float64)
        for i in range(shape[0]):
            for j in range(shape[1]):
                index = int((lst[i, j] - 270) * 1000)
                BTs[i, j] = LUT[index]
    else:
        index = int((lst - 270) * 1000)
        BTs = LUT[index]
    return BTs


def BTs2lst_real(BTs, band=12, angle=0):
    """
    辐亮度转换为地表温度，需考虑发射率
    :param BTs:
    :return:
    """
    fileName = "SRF/LUT" + str(band) + ".txt"
    if angle == 0:
        _, SE =open_tiff("pics/SE_0_" + str(band) + ".tif")
    else:
        _, SE =open_tiff("pics/SE_60_" + str(band) + ".tif")
    LUT = get_LUT(fileName)
    if type(BTs) == np.ndarray:
        # 辐亮度除以发射率，再进行到地表温度的转换
        BTs = BTs / SE
        # 记录初始数组大小，再转换为一维
        original_shape = BTs.shape
        BTs = BTs.reshape(-1)
        shape = BTs.shape
        lst = np.zeros(shape, dtype=np.float64)
        # 一维数组
        for i in range(shape[0]):
            if BTs[i] <= 0:
                lst[i] = 0
                continue
            # 找到最近的温度值对应的索引
            index = np.searchsorted(np.array(LUT), BTs[i])
            lst[i] = 270 + 0.001 * index
        # 转回之前的shape
        lst = lst.reshape(original_shape)
        return lst
    else:
        return None


def get_aster_lat_lon(sds):
    """
    获取一个ASTER数据集的所有像元坐标
    :param sds:
    :return: lat与lon数组（ndarray）
    """
    # 获取11*11的坐标数据
    lat_11 = sds[6].ReadAsArray()
    lon_11 = sds[7].ReadAsArray()
    print(np.min(lat_11))
    # 将其转化为700*830的坐标数据
    # （每个小像元使用其对应大像元四个顶点在两个方向分别插值得到，一个大像元包含70*83个小像元）
    # 用于存储lat值的数组，700行
    lat = [[] for i in range(700)]
    lon = [[] for i in range(700)]
    # 进行插值
    for y in range(700):
        for x in range(830):
            # 对每个小像元，进行双线性插值
            # 获取对应大像元索引（较小值）, 0-10
            index_x = int(x / 83)
            index_y = int(y / 70)
            # 获取在大像元中的位置, 0-82
            offset_x = x - 83 * index_x
            offset_y = y - 70 * index_y
            # lat
            # x方向插值
            lat_up = lat_11[index_y, index_x] + (lat_11[index_y, index_x+1] - lat_11[index_y, index_x]) * (offset_x * 2 + 1) / 166
            lat_down = lat_11[index_y+1, index_x] + (lat_11[index_y+1, index_x+1] - lat_11[index_y+1, index_x]) * (offset_x * 2 + 1) / 166
            # 在y方向插值
            lat[y].append(lat_up + (lat_down - lat_up) * (offset_y * 2 + 1) / 140)
            # lon
            # x方向插值
            lon_up = lon_11[index_y, index_x] + (lon_11[index_y, index_x + 1] - lon_11[index_y, index_x]) * (
                        offset_x * 2 + 1) / 166
            lon_down = lon_11[index_y + 1, index_x] + (lon_11[index_y + 1, index_x + 1] - lon_11[index_y + 1, index_x]) * (offset_x * 2 + 1) / 166
            # 在y方向插值
            lon[y].append(lon_up + (lon_down - lon_up) * (offset_y * 2 + 1) / 140)

    # 将数组转化为ndarray再返回
    return np.asarray(lat), np.asarray(lon)


def generate_angles(shape:tuple, minVZA=55):
    """
    给定一个数据的shape，生成同样大小的角度（VZA）数据
    :param shape:
    :param minVZA:  范围内的最小角度，作为左上角
    :return:
    """
    VZA = np.zeros(shape)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # 每个点，根据横纵坐标计算，i为纵坐标
            VZA[i, j] = minVZA + i * 0.013*2 + j * 0.033*2
    write_tiff(VZA, "VZA_up")
    return VZA


# </editor-fold>

# ****************************************** 绘制图像 **************************************
# <editor-fold>


def display(data, title, cmap=None):
    """
    To display nparray data as picture
    :param data: nparray
    :param title: name of the figure (file)
    :return:
    """
    # matplotlib: 1 or 3 bands
    if cmap is None:
        plt.imshow(data)
    else:
        plt.imshow(data, cmap=plt.get_cmap(cmap))

    plt.colorbar()

    # 去掉坐标轴刻度
    plt.xticks([])
    plt.yticks([])
    plt.savefig("pics/" + title + ".png", dpi=300)
    plt.show()


def display_hist(data, title):
    data = data.reshape(-1)
    # print(data.shape)
    plt.hist(data, bins=30, density=True)
    plt.xlabel("difference")
    plt.ylabel("frequency")
    plt.savefig("pics/" + title + "_hist.png", dpi=400)
    # plt.show()
    plt.cla()


def scatter_BTs_fvc(BT, fvc, k1, c1, k2, c2, band=12, edge=True, angle=0):
    """
    To draw the scatter
    :param lst_file:
    :param ndvi_file:
    :return:
    """
    # display the figure
    # reshape to an one-dimension array
    # sub_lst.reshape(-1)
    # sub_ndvi.reshape(-1)

    # scatter

    # ndvi[ndvi<=0]=np.nan
    # ndvi[ndvi>=1]=np.nan

    plt.scatter(fvc, BT, color="cornflowerblue", s=1.)
    if edge:
        # edges
        x = np.array([(0.025 + i * 0.05) for i in range(20)])
        # x = np.array([(0.05 + i * 0.1) for i in range(10)])
        y1 = x * k1 + c1
        y2 = x * k2 + c2

        # dry edge
        plt.plot(x, y1, label="dry edge", color='r')
        # wet edge
        plt.plot(x, y2, label="wet edge", color='b')
        # plt.plot(x, Tmin, label="wet edge", color='b')

    plt.legend()
    plt.xlabel("FVC")
    plt.ylabel("Radiance")
    # plt.ylim(7.55, 11.3)
    # plt.ylim(np.min(BT) - 0.5, np.max(BT) + 0.5)
    # plt.xlim(0, 1.2)
    # plt.savefig("fvc_BTs_edges.png")
    plt.savefig("pics/BTs_fvc_edges_" + str(band) + "_" + str(angle) + ".png")
    # plt.show()
    plt.cla()


def display_LUT():
    LUT = get_LUT(file_LUT)
    list_LST = []
    list_BT = []
    for i in range(len(LUT)):
        list_BT.append(LUT[i])
        list_LST.append(240+i*0.1)

    plt.xlabel("LST")
    plt.ylabel("BT")
    plt.plot(list_LST, list_BT)
    plt.savefig("pics/LUT.png")


def display_lines_0_60(BT_0, BT_60, FVC_0, FVC_60, band=12):
    """
    在同一特征空间内绘制0度与60数据的连线图
    :return:
    """
    # 绘制
    for i in range(BT_0.shape[0]):
    # for i in range(int(BT_0.shape[0] / 2), BT_0.shape[0]):
        # 每个像元绘制0到60的连线
        # plt.annotate("", xy=(FVC_60[i, j], BT_60[i, j]), xytext=(FVC_0[i, j], BT_0[i, j]), )
        plt.plot((FVC_0[i], FVC_60[i]), (BT_0[i], BT_60[i]),
            color=(random.randint(0, 200)/256, random.randint(0, 200)/256, random.randint(0, 200)/256), linewidth=0.5, alpha=0.5)

    # 坐标轴、标签等
    plt.xlabel("FVC")
    plt.ylabel("BT")

    # 存储与显示
    plt.savefig("pics/lines_colorful_" + str(band) + ".png", dpi=500)
    # plt.show()
    plt.cla()


def display_FVCdiff():
    """
    绘制FVC差值图
    :return:
    """
    _, FVC_60 = open_tiff("pics/FVC.tif")
    _, FVC_0 = open_tiff("pics/FVC_0.tif")
    _, is_valid = open_tiff("pics/is_valid.tif")
    diff = (FVC_60 - FVC_0) * is_valid
    write_tiff(diff, "FVC_diff")


def display_BTsv_diff():
    """
    求一定波段的植被/土壤组分辐亮度及其差值，作箱线图
    :return:
    """
    _, LSTv = open_tiff("pics/LSTv_all.tif")
    _, LSTs = open_tiff("pics/LSTs_all.tif")

    diff_all = []

    for i in range(10, 15):
        # 每个波段计算发射率v/s乘BTs/BTv的结果
        _, SEv = open_tiff("pics/SEv_aver_" + str(i) + ".tif")
        _, SEs = open_tiff("pics/SEs_aver_" + str(i) + ".tif")
        BTv = SEv * LSTv
        BTs = SEs * LSTs
        diff = BTs - BTv
        # write_tiff(diff, "diff_sv_" + str(i))
        # write_tiff(BTv, "BTv_all_" + str(i))
        # write_tiff(BTs, "BTs_all_" + str(i))

        diff = diff[diff < 20]
        diff = diff[diff > -20]
        diff = diff[diff != 0]
        print(diff.shape)
        diff_all.append(diff.tolist())
        # display_hist(diff, "diff_sv_" + str(i))   # 当前波段的差值直方图，效果不好（主要在0左右）

    # 箱线图：效果也不好
    fig, ax = plt.subplots()
    ax.boxplot(diff_all, meanline=True, showfliers=False)
    ax.set_xticklabels([str(x) for x in range(10, 15)])
    ax.set_xlabel("bands")
    ax.set_ylabel("BTs-BTv")
    fig.savefig("pics/boxplot_BTsv.png", dpi=400)

    return


def result_diff():
    """
    结果分析部分，差值等txt导出
    :return:
    """
    # 组分温度
    _, LSTv = open_tiff("pics/LSTv_up_noise.tif")
    _, LSTs = open_tiff("pics/LSTs_up_noise.tif")
    # 覆盖度
    _, FVC0 = open_tiff("pics/FVC_0_up.tif")
    _, FVC60 = open_tiff("pics/FVC_60_up.tif")
    diff_clst = LSTs - LSTv     # 组分温度差值
    diff_fvc = FVC60 - FVC0

    # 对应文件
    file_clst = open("pics/diff_CLST.txt", 'w')
    file_fvc = open("pics/diff_FVC.txt", 'w')

    shape = diff_clst.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            if FVC0[i, j] != 0:
                file_clst.write(str(diff_clst[i, j]) + ",")
                file_fvc.write(str(diff_fvc[i, j]) + ",")

    file_clst.close()
    file_fvc.close()

    # 每个波段输出地表温度差值
    for band in range(10, 15):
        # 地表温度结果
        _, LST_0_space = open_tiff("pics/LST_space_0_final_" + str(band) + ".tif")
        _, LST_0 = open_tiff("pics/LST_0_final_" + str(band) + ".tif")
        _, LST_60 = open_tiff("pics/LST_final_" + str(band) + ".tif")

        # 差值s
        diff = LST_0_space - LST_0    # 实际结果与理想结果差值
        diff_ori = LST_60 - LST_0     # 模拟/理想的纠正量
        diff_real = LST_0_space - LST_60  # 实际算法纠正量
        file_diff = open("pics/diff_corr_simu_" + str(band) + ".txt", 'w')
        file_ori = open("pics/diff_simu_diff_" + str(band) + ".txt", 'w')
        file_real = open("pics/diff_corr_diff_" + str(band) + ".txt", 'w')
        shape = diff.shape
        for i in range(shape[0]):
            for j in range(shape[1]):
                if LST_0_space[i, j] != 0:
                    file_diff.write(str(diff[i, j]) + ",")
                    file_ori.write(str(diff_ori[i, j]) + ",")
                    file_real.write(str(diff_real[i, j]) + ",")

        file_diff.close()
        file_ori.close()
        file_real.close()


# </editor-fold>

# ****************************************** 综合 ******************************************

# 适用于hdf格式的ASTER温度文件
def main_hdf():
    """
    simulation experiment of angular normalization
    打开MODIS、ASTER、CI文件
	    从MODIS数据获取LAI（500m）
    根据位置匹配三种数据
    对ASTER每个像元分类（30m，veg/soil）
    根据分类结果计算MODIS像元内的平均Ts，Tv
    根据平均Ts、Tv及fvc计算B(T)模拟值（1km，原始角度）
    根据模拟数据及fvc建立fvc-B(T)空间（1km，原始角度）
    结合垂直fvc与特征空间计算垂直B(T)
    :return:
    """
    # <editor-fold> 打开ASTER与MODIS相应文件，获取数据
    # ASTER
    # 温度
    sds_aster, _ = open_gdal(file_LST_ASTER_hdf)
    lst_aster = sds_aster[3].ReadAsArray() * 0.1
    lst_aster[lst_aster < 250] = 250
    # display(lst_aster, "LST_ASTER")
    lat_aster, lon_aster = get_aster_lat_lon(sds_aster)
    # display(lat_aster, "Lat_ASTER")
    # display(lon_aster, "Lon_ASTER")
    # 反射率（red, nir）
    sds_ref_aster, _ = open_gdal(file_refl_ASTER)
    ref_vis_aster = sds_ref_aster[0].ReadAsArray() * 0.001
    ref_red_aster = sds_ref_aster[1].ReadAsArray() * 0.001
    ref_nir_aster = sds_ref_aster[2].ReadAsArray() * 0.001
    # 发射率，分辨率与温度相同
    # 5个波段
    sds_se_aster, _ = open_gdal(file_SE_ASTER)
    SE_10 = sds_se_aster[0].ReadAsArray() * 0.001
    SE_11 = sds_se_aster[1].ReadAsArray() * 0.001
    SE_12 = sds_se_aster[2].ReadAsArray() * 0.001
    SE_13 = sds_se_aster[3].ReadAsArray() * 0.001
    SE_14 = sds_se_aster[4].ReadAsArray() * 0.001

    # MODIS
    # MODIS的hdf中有坐标信息，因此用到metadata
    ds_LAI, LAI = open_tiff(file_MOD15)

    # CI
    ds_CI, CI = open_tiff(file_CI)

    # print(CI.shape)
    # print(LAI.shape)

    # </editor-fold>

    # <editor-fold> 预处理：scale，裁剪等
    # 获取ASTER数据外包矩形的坐标
    minLat_ASTER = np.min(lat_aster)
    maxLat_ASTER = np.max(lat_aster)
    minLon_ASTER = np.min(lon_aster)
    maxLon_ASTER = np.max(lon_aster)
    print(minLat_ASTER, minLon_ASTER, maxLat_ASTER, maxLon_ASTER)

    # CI
    # 计算对应的CI横纵坐标范围
    geotrans_CI = ds_CI.GetGeoTransform()
    print(geotrans_CI)
    base_lon = geotrans_CI[0]
    base_lat = geotrans_CI[3]
    inter_lon = geotrans_CI[1]
    inter_lat = geotrans_CI[5]
    min_x = cal_index(base_lon, inter_lon, minLon_ASTER)
    max_x = cal_index(base_lon, inter_lon, maxLon_ASTER)
    min_y = cal_index(base_lat, inter_lat, maxLat_ASTER)  # 这里由于索引大对应纬度小，进行调换
    max_y = cal_index(base_lat, inter_lat, minLat_ASTER)
    # print(min_x, max_x, min_y, max_y)

    # MODIS
    # 计算对应的MODIS横纵坐标范围
    # base为左上角点的坐标，inter为像元分辨率
    geotrans_LAI = ds_LAI.GetGeoTransform()
    print(ds_LAI.GetProjection())
    print(geotrans_LAI)
    base_lon = geotrans_LAI[0]
    base_lat = geotrans_LAI[3]
    inter_lon = geotrans_LAI[1]
    inter_lat = geotrans_LAI[5]
    min_x_2 = cal_index(base_lon, inter_lon, minLon_ASTER)
    max_x_2 = cal_index(base_lon, inter_lon, maxLon_ASTER)
    min_y_2 = cal_index(base_lat, inter_lat, maxLat_ASTER)  # 这里由于索引大对应纬度小，进行调换
    max_y_2 = cal_index(base_lat, inter_lat, minLat_ASTER)
    print(min_x_2, max_x_2, min_y_2, max_y_2)

    # 可能有微小差距（1个像元）
    max_x += ((max_x_2 - min_x_2) - (max_x - min_x))
    max_y += ((max_y_2 - min_y_2) - (max_y - min_y))

    # 对CI、LAI数据进行裁剪、输出文件
    CI = CI[min_y - 1:max_y + 1, min_x - 1:max_x + 1] * 0.001
    LAI = LAI[min_y_2 - 1:max_y_2 + 1, min_x_2 - 1:max_x_2 + 1] * 0.1
    write_tiff(CI, "CI")
    write_tiff(LAI, "LAI")

    # LAI与CI数据应当是可以完全对应的
    # </editor-fold>

    # <editor-fold> 对每个MODIS像元：获取对应的CI值，计算fvc_60与fvc_0；计算其对应ASTER像元的组分发射率与温度，输出结果
    # 60度
    realVZA = generate_angles(LAI.shape)
    # 0度
    theta_0 = 0

    # 计算FVC，G默认为0.5，并导出图像
    FVC_60 = cal_fvc_gap(LAI, CI, realVZA)
    FVC_0 = cal_fvc_gap(LAI, CI, theta_0)
    write_tiff(FVC_60, "FVC")
    write_tiff(FVC_0, "FVC_0")
    print("done FVC calculation")

    # 进行ASTER的像元分类，计算组分温度与发射率
    # 用于存储组分温度与组分发射率的数组
    LSTv_all = np.zeros(LAI.shape, dtype=np.float64)
    LSTs_all = np.zeros(LAI.shape, dtype=np.float64)
    SEs_aver_10 = np.zeros(LAI.shape, dtype=np.float64)
    SEs_aver_11 = np.zeros(LAI.shape, dtype=np.float64)
    SEs_aver_12 = np.zeros(LAI.shape, dtype=np.float64)
    SEs_aver_13 = np.zeros(LAI.shape, dtype=np.float64)
    SEs_aver_14 = np.zeros(LAI.shape, dtype=np.float64)
    SEv_aver_10 = np.zeros(LAI.shape, dtype=np.float64)
    SEv_aver_11 = np.zeros(LAI.shape, dtype=np.float64)
    SEv_aver_12 = np.zeros(LAI.shape, dtype=np.float64)
    SEv_aver_13 = np.zeros(LAI.shape, dtype=np.float64)
    SEv_aver_14 = np.zeros(LAI.shape, dtype=np.float64)
    # 用于存储构造特征空间的数据的列表，数据用[BT,fvc]记录
    is_valid = np.zeros(LAI.shape, dtype=bool)                  # 是否有有效组分（无云/水体的ASTER像元）
    for y_modis in range(LAI.shape[0]):         # 一行
        if y_modis % 10 == 0:
            print(str(y_modis) + " rows")
        for x_modis in range(LAI.shape[1]):
            # 对当前MODIS像元，用于存储对应ASTER像元温度、组分发射率信息的列表
            LSTs = []
            SEv_10 = []
            SEs_10 = []
            SEv_11 = []
            SEs_11 = []
            SEv_12 = []
            SEs_12 = []
            SEv_13 = []
            SEs_13 = []
            SEv_14 = []
            SEs_14 = []
            # 像元中invalid的个数
            invalidNum = 0
            # 当前MODIS像元坐标范围
            cur_minLon = geotrans_LAI[0] + geotrans_LAI[1] * (min_x_2-1+x_modis-0.5)
            cur_maxLon = geotrans_LAI[0] + geotrans_LAI[1] * (min_x_2-1+x_modis+0.5)
            cur_minLat = geotrans_LAI[3] + geotrans_LAI[5] * (min_y_2-1+y_modis+0.5)
            cur_maxLat = geotrans_LAI[3] + geotrans_LAI[5] * (min_y_2-1+y_modis-0.5)
            # 找到对应的ASTER像元
            for y_aster in range(lst_aster.shape[0]):
                # 先对该行的纬度进行判断
                # 当前行都在目标像元下方
                if lat_aster[y_aster, 0] < cur_minLat and lat_aster[y_aster, lat_aster.shape[1]-1] < cur_minLat:
                    break
                # 当前行都在目标像元上方
                elif lat_aster[y_aster, 0] > cur_maxLat and lat_aster[y_aster, lat_aster.shape[1]-1] > cur_maxLat:
                    continue
                # 当前行跟目标像元有交叉
                for x_aster in range(lst_aster.shape[1]):
                    # 去除图像边缘点
                    if lst_aster[y_aster, x_aster] == 250:
                        continue
                    # 当前ASTER像元
                    # 判断纬度是否在范围内
                    if cur_minLat <= lat_aster[y_aster, x_aster] <= cur_maxLat:
                        # 判断经度是否在范围内
                        if cur_minLon <= lon_aster[y_aster, x_aster] <= cur_maxLon:
                            # 判断当前像元是否植被——计算NDVI
                            # 一个ASTER LST像元（90m）包含6*6个反射率像元（15m）
                            cur_vis = ref_vis_aster[y_aster * 6:y_aster * 6 + 5, x_aster * 6:x_aster * 6 + 5]
                            cur_red = ref_red_aster[y_aster * 6:y_aster * 6 + 5, x_aster * 6:x_aster * 6 + 5]
                            cur_nir = ref_nir_aster[y_aster * 6:y_aster * 6 + 5, x_aster * 6:x_aster * 6 + 5]
                            cur_ndvi = np.mean(cal_NDVI(cur_red, cur_nir))
                            # 判断是否为阴影（山/云）或水
                            # 计算nir波段反射率方差
                            cur_std = np.std(cur_nir)
                            # print(cur_std)
                            # 针对ASTER三个波段反射率、nir波段反射率方差进行筛选
                            # 去除水体、阴影、云、FVC异常值（来自CI、LAI的异常像元）
                            if (not (np.mean(cur_vis) < 0.2 and np.mean(cur_nir) < 0.2 and np.mean(cur_red) < 0.2)) and \
                                    (not (np.mean(cur_vis) > 0.17 and np.mean(cur_nir) > 0.28 and np.mean(cur_red) > 0.17))\
                                    and cur_std < 0.05 and FVC_60[y_modis, x_modis] > 0 and FVC_0[y_modis, x_modis] < 0.95:
                                is_valid[y_modis, x_modis] = True
                                # 根据平均NDVI值判断是植被还是土壤
                                LSTs.append(lst_aster[y_aster, x_aster])
                                if cur_ndvi > threshold_NDVI:
                                    SEv_10.append(SE_10[y_aster, x_aster])
                                    SEv_11.append(SE_11[y_aster, x_aster])
                                    SEv_12.append(SE_12[y_aster, x_aster])
                                    SEv_13.append(SE_13[y_aster, x_aster])
                                    SEv_14.append(SE_14[y_aster, x_aster])
                                else:
                                # elif cur_ndvi < threshold_NDVI_min:
                                    SEs_10.append(SE_10[y_aster, x_aster])
                                    SEs_11.append(SE_11[y_aster, x_aster])
                                    SEs_12.append(SE_12[y_aster, x_aster])
                                    SEs_13.append(SE_13[y_aster, x_aster])
                                    SEs_14.append(SE_14[y_aster, x_aster])
                            # 无效像元
                            else:
                                invalidNum += 1
                        # 经度不在范围内
                        else:
                            # 当前行已经匹配完了
                            if lon_aster[y_aster, x_aster] > cur_maxLon:
                                break
                            # 当前行还没匹配到
                            else:
                                continue
                    # 当前纬度不在范围内
                    else:
                        continue

                # 判断无效ASTER像元个数，大于一定量就将整个MODIS像元设为无效
                if invalidNum > 5:
                    is_valid[y_modis, x_modis] = False
                    break

            # 存储当前像元的组分温度(极值)与组分发射率（均值）
            # 组分温度为所有像元的极值，组分发射率为对应组分的均值
            if is_valid[y_modis, x_modis]:
                if len(LSTs) > 0:
                    LSTv_all[y_modis, x_modis] = np.min(LSTs)
                    LSTs_all[y_modis, x_modis] = np.max(LSTs)

                if len(SEv_10) > 0:
                    SEv_aver_10[y_modis, x_modis] = np.mean(SEv_10)
                    SEv_aver_11[y_modis, x_modis] = np.mean(SEv_11)
                    SEv_aver_12[y_modis, x_modis] = np.mean(SEv_12)
                    SEv_aver_13[y_modis, x_modis] = np.mean(SEv_13)
                    SEv_aver_14[y_modis, x_modis] = np.mean(SEv_14)
                if len(SEs_10) > 0:
                    SEs_aver_10[y_modis, x_modis] = np.mean(SEs_10)
                    SEs_aver_11[y_modis, x_modis] = np.mean(SEs_11)
                    SEs_aver_12[y_modis, x_modis] = np.mean(SEs_12)
                    SEs_aver_13[y_modis, x_modis] = np.mean(SEs_13)
                    SEs_aver_14[y_modis, x_modis] = np.mean(SEs_14)

    print("done component temperature calculation")
    # </editor-fold>

    # <editor-fold> 中间结果输出
    # 出图
    write_tiff(LSTv_all, "LSTv_all")
    write_tiff(LSTs_all, "LSTs_all")
    write_tiff(SEs_aver_10, "SEs_aver_10")
    write_tiff(SEs_aver_11, "SEs_aver_11")
    write_tiff(SEs_aver_12, "SEs_aver_12")
    write_tiff(SEs_aver_13, "SEs_aver_13")
    write_tiff(SEs_aver_14, "SEs_aver_14")
    write_tiff(SEv_aver_10, "SEv_aver_10")
    write_tiff(SEv_aver_11, "SEv_aver_11")
    write_tiff(SEv_aver_12, "SEv_aver_12")
    write_tiff(SEv_aver_13, "SEv_aver_13")
    write_tiff(SEv_aver_14, "SEv_aver_14")
    write_tiff(is_valid, "is_valid")

    # </editor-fold>


def add_noise():
    """
    给计算出的组分温度添加噪声
    :return:
    """
    _, LSTv = open_tiff("pics/LSTv_up.tif")
    _, LSTs = open_tiff("pics/LSTs_up.tif")
    # 添加噪声
    noise = np.random.normal(1, 1, LSTs.shape)
    LSTv_new = LSTv + noise
    noise = np.random.normal(1, 1, LSTs.shape)
    LSTs_new = LSTs + noise
    # 有效区域外重新赋值
    LSTv_new[LSTv==0] = 0
    LSTs_new[LSTs==0] = 0
    # 输出结果
    write_tiff(LSTs_new, "LSTs_up_noise")
    write_tiff(LSTv_new, "LSTv_up_noise")


def main_calRadiance(band=12):
    """
    从平均组分温度、组分发射率、植被覆盖度来计算辐亮度及等效发射率
    :return:
    """
    print("radiance calculation for band " + str(band))
    # 打开所需数据文件
    _, LSTv = open_tiff("pics/LSTv_up_noise.tif")
    _, LSTs = open_tiff("pics/LSTs_up_noise.tif")
    _, SEs = open_tiff("pics/SEs_up_" + str(band) + ".tif")
    _, SEv = open_tiff("pics/SEv_up_" + str(band) + ".tif")
    _, FVC_60 = open_tiff("pics/FVC_60_up.tif")
    _, FVC_0 = open_tiff("pics/FVC_0_up.tif")
    _, is_valid = open_tiff("pics/is_valid_up.tif")

    # 存储计算出来的两个角度辐亮度的数组
    BT_0 = np.zeros(LSTs.shape, dtype=np.float64)
    BT_60 = np.zeros(LSTs.shape, dtype=np.float64)
    SE = np.zeros(LSTs.shape, dtype=np.float64)
    SE_0 = np.zeros(LSTs.shape, dtype=np.float64)
    # 遍历每个像元，计算辐亮度与等效发射率
    for y in range(LSTs.shape[0]):
        for x in range(LSTs.shape[1]):
            # 是有效像元才进行计算
            if is_valid[y, x]:
                BTs = lst2BTs(LSTs[y, x], band)
                BTv = lst2BTs(LSTv[y, x], band)
                # 判断发射率，没有的组分就使用平均
                cur_SEs = SEs[y, x] if SEs[y, x] > 0 else SEs_aver[(band-10)*2]
                cur_SEv = SEv[y, x] if SEv[y, x] > 0 else SEs_aver[(band-10)*2+1]
                #
                BT_0[y, x] = FVC_0[y, x] * BTv * cur_SEv + (1 - FVC_0[y, x]) * BTs * cur_SEs
                BT_60[y, x] = FVC_60[y, x] * BTv * cur_SEv + (1 - FVC_60[y, x]) * BTs * cur_SEs
                # if BT_60[y,x] == 0 and BT_0[y, x] != 0:
                #     print(BT_0[y,x])
                #     print(BTs, BTv, FVC_0[y, x], FVC_60[y, x], SEv[y, x], SEs[y, x])
                # 计算等效发射率
                SE[y, x] = FVC_60[y, x] * cur_SEv + (1 - FVC_60[y, x]) * cur_SEs
                SE_0[y, x] = FVC_0[y, x] * cur_SEv + (1 - FVC_0[y, x]) * cur_SEs

            # 其他情况都不考虑

    write_tiff(BT_0, "BT_0_" + str(band))
    write_tiff(BT_60, "BT_60_" + str(band))
    write_tiff(SE, "SE_60_" + str(band))
    write_tiff(SE_0, "SE_0_" + str(band))


def main_space(band=12):
    """
    得到模拟结果后，进行特征空间相关处理
    :return:
    """
    # <editor-fold> 读取数据，自动提取顶点

    print("space construction for band " + str(band))
    # 读取相关数据：某一波段的多角度辐亮度
    ds_BT, BT = open_tiff("pics/BT_60_" + str(band) + ".tif")
    ds_BT_0, BT_0 = open_tiff("pics/BT_0_" + str(band) + ".tif")
    ds_fvc, fvc = open_tiff("pics/FVC_60_up.tif")
    ds_fvc_0, fvc_0 = open_tiff("pics/FVC_0_up.tif")
    ds_valid, is_valid = open_tiff("pics/is_valid_up.tif")

    # 更新is_valid数据
    is_valid[BT <= 0] = 0
    is_valid[fvc <= 0] = 0
    is_valid[BT_0 <= 0] = 0
    is_valid[fvc_0 <= 0] = 0

    # 获取有效值（有效值转换为一维列表）
    # 构建特征空间只使用有效值
    BT_valid = BT[is_valid > 0]
    BT_0_valid = BT_0[is_valid > 0]
    fvc_valid = fvc[is_valid > 0]
    fvc_0_valid = fvc_0[is_valid > 0]
    # 辐亮度转为地表温度
    LST = BTs2lst_real(BT, band, 60)
    LST_0 = BTs2lst_real(BT_0, band, 0)
    LST_valid = LST[is_valid > 0]
    LST_0_valid = LST_0[is_valid > 0]

    # 0-60图绘制
    display_lines_0_60(BT_0_valid, BT_valid, fvc_0_valid, fvc_valid, band)

    # 垂直角度的特征空间
    k1, c1, k2, c2 = getEdges_fvc(BT_0_valid, fvc_0_valid)
    scatter_BTs_fvc(BT_0_valid, fvc_0_valid, k1, c1, k2, c2, band, True, 0)

    # 倾斜方向的特征空间
    k1, c1, k2, c2 = getEdges_fvc(BT_valid, fvc_valid)
    print(k1, c1, k2, c2)
    # 出图
    scatter_BTs_fvc(BT_valid, fvc_valid, k1, c1, k2, c2, band, True, 60)
    # 计算特征空间中的顶点
    point_x, point_y = cal_vertex(k1, c1, k2, c2)
    print(point_x, point_y)

    # </editor-fold>

    # 自动提取
    # point_x = 9
    # point_y = 0.8

    # 寻找最优顶点
    best_x = 0
    best_y = 0
    best_RMSE = 5
    preRMSE = 100
    # 记录RMSE的文件
    # file = open("pics/RMSEs_space" + str(band) + ".txt", 'w')
    # file.write("fvc\tRadiance\tRMSE\n")
    for x in range(502, 505):
        point_x = x / 10
        print("x: " + str(x))
        for y in [-x + delta for delta in range(-130, 300)]:
            if y % 10 == 0:
                print("y: " + str(y))
            point_y = y / 10
            BT_0_space = np.zeros(BT_0.shape, dtype=np.float64)
            for i in range(BT_0.shape[0]):
                for j in range(BT_0.shape[1]):
                    # FVC过大的点直接去除
                    if fvc_0[i, j] > point_x or fvc[i, j] > point_x or is_valid[i, j] <= 0:
                        continue
                    k, c = cal_params(point_y, point_x, BT[i, j], fvc[i, j])
                    BT_0_space[i, j] = k * fvc_0[i, j] + c
            # 转换为地表温度，求地表温度的最小误差
            cur_LST = BTs2lst_real(BT_0_space, band, 0)
            RMSE_LST_space_0 = np.sqrt(metrics.mean_squared_error(cur_LST[is_valid > 0], LST_0_valid))
            # RMSE_BT_space_0 = np.sqrt(metrics.mean_squared_error(BT_0, BT_0_space))
            # file.write("%f\t%f\t%f\n" % (point_x, point_y, RMSE_BT_space_0))
            # 根据fvc_0与特征空间计算垂直方向辐亮度
            if RMSE_LST_space_0 < best_RMSE:
                best_x = point_x
                best_y = point_y
                best_RMSE = RMSE_LST_space_0
            # 如果RMSE大于上一个，则后面都是递增
            if RMSE_LST_space_0 > preRMSE:
                preRMSE = 100
                break
            else:
                preRMSE = RMSE_LST_space_0

    # file.close()
    print("best x: " + str(best_x))
    print("best y: " + str(best_y))
    print("best RMSE: " + str(best_RMSE))
    point_x = best_x
    point_y = best_y

    # <editor-fold> 计算纠正后的Radiance
    BT_0_space = np.zeros(is_valid.shape, dtype=np.float64)
    for i in range(BT_0_space.shape[0]):
        for j in range(BT_0_space.shape[1]):
            # 无效值直接为0
            if is_valid[i, j] == 0:
                BT_0_space[i, j] = 0
                continue
            # FVC过大的点直接去除
            if fvc_0[i, j] > point_x or fvc[i, j] > point_x:
                continue
            k, c = cal_params(point_y, point_x, BT[i, j], fvc[i, j])
            BT_0_space[i, j] = k * fvc_0[i, j] + c
    # </editor-fold>

    # <editor-fold> 结果定量分析
    BT_0_space_valid = BT_0_space[is_valid > 0]
    # 计算结果与模拟结果进行对比
    RMSE_BT_space_0 = np.sqrt(metrics.mean_squared_error(BT_0_valid, BT_0_space_valid))
    display_hist(BT_0_space_valid - BT_0_valid, "Radiance_diff_space_0_" + str(band))
    print("RMSE_Radiance_space_0:\t" + str(RMSE_BT_space_0))
    # 原始数据与模拟结果的对比
    display_hist(BT_0_valid - BT_valid, "Radiance_diff_0_" + str(band))
    RMSE_BT_0 = np.sqrt(metrics.mean_squared_error(BT_valid, BT_0_valid))
    print("RMSE_Radiance_0:\t\t" + str(RMSE_BT_0))

    # 原始数据与特征空间结果的对比【不需要】
    display_hist(BT_0_space_valid - BT_valid, "Radiance_diff_space_" + str(band))
    RMSE_BT_space = np.sqrt(metrics.mean_squared_error(BT_valid, BT_0_space_valid))
    print("RMSE_Radiance_space:\t" + str(RMSE_BT_space))

    # 温度对比

    LST_space_0 = BTs2lst_real(BT_0_space, band, 0)

    LST_space_0_valid = LST_space_0[is_valid > 0]

    display_hist(LST_space_0_valid - LST_0_valid, "BT_diff_space_0_" + str(band))
    RMSE_LST_space_0 = np.sqrt(metrics.mean_squared_error(LST_space_0_valid, LST_0_valid))
    print("RMSE_BT_space_0:\t\t" + str(RMSE_LST_space_0))
    display_hist(LST_0_valid - LST_valid, "BT_diff_0_" + str(band))
    RMSE_LST_0 = np.sqrt(metrics.mean_squared_error(LST_0_valid, LST_valid))
    print("RMSE_BT_0:\t\t" + str(RMSE_LST_0))
    display_hist(LST_space_0_valid - LST_valid, "BT_diff_space_" + str(band))
    RMSE_LST_space = np.sqrt(metrics.mean_squared_error(LST_space_0_valid, LST_valid))
    print("RMSE_BT_space:\t\t" + str(RMSE_LST_space))
    # </editor-fold>

    # 出图
    write_tiff(BT_0_space, "Radiance_0_space_" + str(band))
    write_tiff(LST, "LST_final_" + str(band))
    write_tiff(LST_0, "LST_0_final_" + str(band))
    write_tiff(LST_space_0, "LST_space_0_final_" + str(band))


def analysis_LSTsv():
    """
    对计算出的组分温度进行分析
    :return:
    """
    _, LSTv = open_tiff("pics/LSTv_all.tif")
    _, LSTs = open_tiff("pics/LSTs_all.tif")

    # 为便于出图
    LSTv[LSTv == 0] = -200
    LSTs[LSTs == 0] = -100
    diff_LSTsv = LSTs - LSTv  # 差值
    # 去掉0值的直方图
    display_hist(LSTv[LSTv > 0], "LSTv_all")
    display_hist(LSTs[LSTs > 0], "LSTs_all")

    diff_LSTsv[diff_LSTsv >= 100] = -100
    diff_LSTsv[diff_LSTsv < -100] = -100
    display_hist(diff_LSTsv[diff_LSTsv>-90], "diff_LSTsv")
    write_tiff(diff_LSTsv, "diff_LSTsv.tif")


def writeGeo(source, target):
    """
    给指定文件添加地理信息
    :param source:
    :param target:
    :return:
    """
    _, data = open_tiff(source)
    # 用于参考的MODIS LAI数据
    ds_LAI, LAI = open_tiff(file_MOD15)
    proj = ds_LAI.GetProjection()
    # 写新的文件
    driver = gdal.GetDriverByName("GTiff")
    ds_new = driver.Create(target, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    # 手动添加坐标信息（计算左上角顶点）
    # # 2318
    geoTrans = (112.0475831299942, 0.01, 0.0, 41.02, 0.0, -0.01)
    # 2309
    # geoTrans = (112.2025831299942, 0.01, 0.0, 41.55, 0.0, -0.01)
    # 2327
    # geoTrans = (111.90104305828597, 0.01, 0.0, 40.485, 0.0, -0.01)
    # 赋值
    ds_new.SetProjection(proj)
    ds_new.SetGeoTransform(geoTrans)
    ds_new.GetRasterBand(1).WriteArray(data)
    del ds_new


def addGeoinfo():
    """
    给tif文件添加地理坐标信息
    :return:
    """
    # 添加信息的文件，包括两种植被覆盖度、CI、LAI、两种组分温度、三种亮温
    writeGeo("pics/FVC_0_up.tif", "pics/geo/FVC_0_up_geo.tif")
    writeGeo("pics/FVC_60_up.tif", "pics/geo/FVC_60_up_geo.tif")
    # writeGeo("pics/CI_up.tif", "pics/geo/CI_up_geo.tif")
    # writeGeo("pics/LAI_up.tif", "pics/geo/LAI_up_geo.tif")
    writeGeo("pics/LSTs_up.tif", "pics/geo/LSTs_up_geo.tif")
    writeGeo("pics/LSTv_up.tif", "pics/geo/LSTv_up_geo.tif")
    for band in range(10, 15):
        writeGeo("pics/LST_0_final_" + str(band) + ".tif", "pics/geo/LST_0_geo_" + str(band) + ".tif")
        writeGeo("pics/LST_final_" + str(band) + ".tif", "pics/geo/LST_60_geo_" + str(band) + ".tif")
        writeGeo("pics/LST_space_0_final_" + str(band) + ".tif", "pics/geo/LST_0_space_geo_" + str(band) + ".tif")


def test():
    _, FVC_0 = open_tiff("pics/LAI.tif")

    # 上采样操作
    shape = FVC_0.shape
    print(shape)
    new_shape = (int(shape[0] / 2), int(shape[1] / 2))
    new_FVC0 = np.zeros(new_shape, dtype=np.float64)
    new_valid = np.zeros(new_shape, dtype=np.float64)
    for i in range(new_shape[0]):
        for j in range(new_shape[1]):
            # 当前范围存在有效值
            # 其他数据取平均值
            new_FVC0[i, j] = np.mean((FVC_0[i*2:i*2+2, j*2:j*2+2]))
            new_valid[i, j] = 1

    # 存储上采样后的数据
    write_tiff(new_FVC0, "LAI_up")


def sensitivity_overall():
    """
    敏感性分析-计算FVC随各变量的变化
    给定LAI，VZA为横坐标，FVC为纵坐标，每个omega值对应一条曲线
    :return:
    """
    theta_list = range(0, 61, 5)        # VZA
    LAI_list = [0.1, 0.2, 0.3, 0.4, 0.5, 1, 2, 3, 4, 5, 6]  # LAI
    omega_list = range(1, 11, 1)        # clumping index
    G = 0.5     # G暂时取定值（spherical leaf angle distribution）
    # 各种情况下
    for LAI in LAI_list:
        # 一定LAI
        fig, ax = plt.subplots()    # 当前LAI对应的图
        for omega in omega_list:
            omega = omega / 10
            # 一条曲线
            FVC_values = []
            for theta in theta_list:
                theta = theta * math.pi / 180
                # 根据间隙率公式计算FVC
                FVC = 1 - math.exp(-LAI * G * omega / math.cos(theta))
                FVC_values.append(FVC)
            # 绘制当前omega的曲线
            # 给定LAI，VZA为横坐标，FVC为纵坐标，不同omega对应不同曲线
            ax.plot(theta_list, FVC_values, label='omega = ' + str(omega))

        # 绘制完所有图后出图
        ax.legend()
        ax.set_xlabel('VZA')
        ax.set_ylabel('FVC')
        # fig.show()
        fig.savefig('pics/sensitivity/LAI_' + str(LAI) + '.png', dpi=400)


def sensitivity_VZA():
    """
    探究角度效应对FVC的影响
    箱线图，横坐标为不同LAI，纵坐标为FVC，计算各种omega下最大最小角度对应的FVC差
    :return:
    """
    # 生成数据
    LAI_list = [0.1, 0.2, 0.5, 1, 2, 3, 4, 5, 6]
    omega_list = range(1, 11, 1)
    G = 0.5
    data_all = []               # 存储所有数据的列表
    # 一个LAI值对应一个箱线图
    for LAI in LAI_list:
        # 当前LAI的所有数据
        data_cur = []
        for omega in omega_list:
            # 一个omega计算一个FVC差值
            omega = omega / 10
            # 最小与最大VZA分别为0与60
            theta_0 = 0
            theta_60 = math.pi / 3
            FVC_0 = 1 - math.exp(-LAI * G * omega / math.cos(theta_0))
            FVC_60 = 1 - math.exp(-LAI * G * omega / math.cos(theta_60))
            data_cur.append(FVC_60 - FVC_0)
        data_all.append(data_cur)       # 将当前LAI的数据加入总数据集

    # 生成所有数据后，绘制箱线图
    fig, ax = plt.subplots()
    ax.boxplot(data_all)
    ax.set_xticklabels([str(x) for x in LAI_list])
    ax.set_xlabel('LAI')
    ax.set_ylabel('FVC difference')
    fig.savefig('pics/sensitivity/boxplot_VZA60.png', dpi=400)


if __name__ == '__main__':
    # test()
    # get_mean_SE()
    # display_BTsv_diff()

    # 全流程
    # main_hdf()
    # up_sample()
    # add_noise()
    for i in range(11, 15):
        main_calRadiance(i)
        main_space(i)
    result_diff()
    addGeoinfo()

    # calRMSE_new("pics/BT_space_0_final_14.tif", "pics/BT_final_14.tif", "pics/VZA_up.tif", "14")