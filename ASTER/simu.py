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


# ****************************************** 一些声明 **************************************
# ASTER
# 0906
file_LST_ASTER_tiff = "data/ASTER/AST_08_00309042019034740_20210822003637_24184.SurfaceKineticTemperature.KineticTemperature.tif"
# file_LST_ASTER_hdf = "data/ASTER/AST_08_00309042019034740_20210822021707_8909.hdf"
# file_refl_ASTER = "data/ASTER/AST_07_00309042019034740_20210822003647_8266.hdf"
# 0828
# file_LST_ASTER_hdf = "data/ASTER/AST_08_00308282019034132_20210825125717_3112.hdf"
# file_refl_ASTER = "data/ASTER/AST_07_00308282019034132_20210825125751_31909.hdf"
file_LST_ASTER_hdf = "data/ASTER/AST_08_00308252019030937_20211025205719_32091.hdf"
# file_LST_ASTER_hdf = "data/ASTER/AST_08_00308252019030844_20211025205749_4200.hdf"
file_refl_ASTER = "data/ASTER/AST_07_00308252019030937_20211025201308_7969.hdf"
# file_refl_ASTER = "data/ASTER/AST_07_00308252019030844_20211025201358_16022.hdf"
# MOD09
file_MOD09_1 = "data/MODIS/MOD09GA.sur_refl_b01_1.tif"
file_MOD09_2 = "data/MODIS/MOD09GA.sur_refl_b02_1.tif"
file_MOD09_SZA = "data/MODIS/MOD09GA.SolarZenith_1.tif"
file_MOD09_SAA = "data/MODIS/MOD09GA.SolarAzimuth_1.tif"
file_MOD09_VZA = "data/MODIS/MOD09GA.SensorZenith_1.tif"
# MOD15
file_MOD15 = "data/MODIS/LAI_2019233h26v4.Lai_500m.tif"
# MCD43A1
file_MCD43A1_B1_1 = "data/MODIS/MCD43A1.BRDF_Albedo_Parameters_Band1.Num_Parameters_01.tif"
file_MCD43A1_B1_2 = "data/MODIS/MCD43A1.BRDF_Albedo_Parameters_Band1.Num_Parameters_02.tif"
file_MCD43A1_B1_3 = "data/MODIS/MCD43A1.BRDF_Albedo_Parameters_Band1.Num_Parameters_03.tif"
file_MCD43A1_B2_1 = "data/MODIS/MCD43A1.BRDF_Albedo_Parameters_Band2.Num_Parameters_01.tif"
file_MCD43A1_B2_2 = "data/MODIS/MCD43A1.BRDF_Albedo_Parameters_Band2.Num_Parameters_02.tif"
file_MCD43A1_B2_3 = "data/MODIS/MCD43A1.BRDF_Albedo_Parameters_Band2.Num_Parameters_03.tif"
# 聚集指数
# DOY233即233-240
file_CI = "data/CI/CI_6.tif"
# 查找表
file_LUT = "LUT.txt"
# 判断植被/土壤的NDVI阈值
threshold_NDVI = 0.45
# ASTER图像区域

# ****************************************** 文件操作 **************************************


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


# ****************************************** 计算函数 **************************************


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
    # hist, edges = np.histogram(result, 100)
    # print(hist)
    # print(edges)

    # display(result, "NDVI")
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
    return 1-np.exp(LAI * omega * G / np.cos(theta*math.pi/180))


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
    # 遍历所有像元计算
    LSTs = []
    LSTv = []
    num_veg = 0
    num_soil = 0
    for y in range(lst_aster.shape[0]):
        for x in range(lst_aster.shape[1]):
            # 去除边缘无数据点与云
            if lst_aster[y, x] > 285:
                ndvi = cal_NDVI(ref_red_aster[y, x], ref_nir_aster[y, x])
                if ndvi > threshold_NDVI:
                    num_veg += 1
                    LSTv.append(lst_aster[y, x])
                else:
                    num_soil += 1
                    LSTs.append(lst_aster[y, x])
    print(np.mean(LSTs))
    print(np.mean(LSTv))
    print(num_soil)
    print(num_veg)


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
    # average and standard deviation of all lsts
    LST_aver = BT.mean()
    LST_std = np.std(BT)
    print("BT_aver: " + str(LST_aver))
    print("BT_std: " + str(LST_std))

    # divide the NDVI into intervals, 10 * 8 subintervals
    interval_num = 30
    subinterval_num = 6
    # do the statics
    Ts = [[[] for j in range(subinterval_num)] for i in range(interval_num)]
    for i in range(BT.shape[0]):
        for j in range(BT.shape[1]):
            # 去除异常值
            if fvc[i, j] <= 0 or fvc[i, j] >= 1:
                continue
            if BT[i, j] == 0 or BT[i, j] >= LST_aver + 3.25 * LST_std or BT[i, j] <= LST_aver - 2.5 * LST_std:
                continue
            index = int(fvc[i, j] * interval_num * subinterval_num)
            Ts[int(index / subinterval_num)][index % subinterval_num].append(BT[i, j])
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
                if x > mean_all + 3.25 * dev_all or x < mean_all - 3 * dev_all:
                    continue
                # LST
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

        if len(maxTs) == 0 and len(Tmax_aver) > 0:
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

    # print(Tmax_aver)
    # print(Tmin_aver)

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


def calRMSE_new(lst: np.ndarray, lst_0: np.ndarray, VZA: np.ndarray, title: str):
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
        try:
            index = int(VZA[i] + 0.5) - minVZA
            lst_intervals[index].append(lst[i])
            lst0_intervals[index].append(lst_0[i])
            diff_intervals[index].append(np.abs(lst[i] - lst_0[i]))
        except Exception as e:
            print(e)
            print(VZA[i])
            print(index)

    RMSEs = []
    nums = []
    for i in range(len(lst_intervals)):
        # print(i)
        # print(len(lst_intervals[i]))
        if len(lst_intervals[i]) == 0:
            RMSEs.append(RMSEs[-1])
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
    plt.ylabel("pixel num")
    plt.savefig("pics/pixelNum.png")
    plt.show()


# ****************************************** 其他函数 **************************************


def lst2BTs(lst):
    """
    transform lst data into BTs data
    :param lst:
    :return:
    """
    LUT = get_LUT(file_LUT)
    if type(lst) == np.ndarray:
        shape = lst.shape
        BTs = np.zeros(shape, dtype=np.float64)
        for i in range(shape[0]):
            for j in range(shape[1]):
                index = int((lst[i, j] - 240) * 10)
                BTs[i, j] = LUT[index]
    else:
        index = int((lst - 240) * 10)
        BTs = LUT[index]
    return BTs


def BTs2lst(BTs):
    """
    transform BTs data into lst data
    :param BTs:
    :return:
    """
    LUT = get_LUT(file_LUT)
    if type(BTs) == np.ndarray:
        shape = BTs.shape
        lst = np.zeros(shape, dtype=np.float64)
        for i in range(shape[0]):
            for j in range(shape[1]):
                for k in range(len(LUT)):
                    if LUT[k] > BTs[i, j]:
                        continue
                    # LUT中的BTs第一次大于BTs
                    if k == 0:
                        lst[i, j] = 240
                    # 实际BTs更接近后一个
                    if BTs[i, j] * 2 > LUT[k - 1] + LUT[k]:
                        lst[i, j] = 240 + 0.1 * k
                    else:
                        lst[i, j] = 240 + 0.1 * (k - 1)
    else:
        for k in range(len(LUT)):
            if LUT[k] > BTs:
                continue
            # LUT中的BTs第一次大于BTs
            if k == 0:
                lst = 240
            # 实际BTs更接近后一个
            if BTs * 2 > LUT[k - 1] + LUT[k]:
                lst = 240 + 0.1 * k
            else:
                lst = 240 + 0.1 * (k - 1)
    return lst


def get_aster_lat_lon(sds):
    """
    获取一个ASTER数据集的所有像元坐标
    :param sds:
    :return: lat与lon数组（ndarray）
    """
    # 获取11*11的坐标数据
    lat_11 = sds[6].ReadAsArray()
    lon_11 = sds[7].ReadAsArray()
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
            offset_y = y - 83 * index_y
            # lat
            # x方向插值
            lat_up = lat_11[index_y,index_x] + (lat_11[index_y, index_x+1] - lat_11[index_y, index_x]) * (offset_x * 2 + 1) / 166
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


# ****************************************** 绘制图像 **************************************

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
    plt.hist(data, bins=100, density=True)
    plt.xlabel("difference")
    plt.ylabel("frequency")
    plt.savefig("pics/" + title + "_hist.png")
    plt.show()


def scatter_BTs_fvc(BT, fvc, k1, c1, k2, c2):
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
    plt.ylim(6.2, 9)
    # plt.ylim(np.min(BT) - 0.5, np.max(BT) + 0.5)
    # plt.xlim(0, 3)
    # plt.savefig("fvc_BTs_edges.png")
    plt.savefig("pics/BTs_fvc_edges.png")
    plt.show()


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

    # MODIS
    # MODIS的hdf中有坐标信息，因此用到metadata
    ds_LAI, LAI = open_tiff(file_MOD15)

    # CI
    ds_CI, CI = open_tiff(file_CI)

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
    # print(geotrans)
    # CI的坐标单位为m
    base_lon = geotrans_CI[0]
    base_lat = geotrans_CI[3]
    inter_lon = geotrans_CI[1]
    inter_lat = geotrans_CI[5]
    min_x = cal_index(base_lon, inter_lon, minLon_ASTER)
    max_x = cal_index(base_lon, inter_lon, maxLon_ASTER)
    min_y = cal_index(base_lat, inter_lat, maxLat_ASTER)  # 这里由于索引大对应纬度小，进行调换
    max_y = cal_index(base_lat, inter_lat, minLat_ASTER)
    # print(min_x, max_x, min_y, max_y)
    # 进而对CI数据进行裁剪
    CI = CI[min_y - 1:max_y + 1, min_x - 1:max_x + 1] * 0.001
    # print(CI.shape)
    write_tiff(CI, "CI")

    # MODIS
    # 计算对应的MODIS横纵坐标范围
    # base为左上角点的坐标，inter为像元分辨率
    geotrans_LAI = ds_LAI.GetGeoTransform()
    base_lon = geotrans_LAI[0]
    base_lat = geotrans_LAI[3]
    inter_lon = geotrans_LAI[1]
    inter_lat = geotrans_LAI[5]
    min_x = cal_index(base_lon, inter_lon, minLon_ASTER)
    max_x = cal_index(base_lon, inter_lon, maxLon_ASTER)
    min_y = cal_index(base_lat, inter_lat, maxLat_ASTER)  # 这里由于索引大对应纬度小，进行调换
    max_y = cal_index(base_lat, inter_lat, minLat_ASTER)
    print(min_x, max_x, min_y, max_y)
    # 进而对MODIS数据进行裁剪
    LAI = LAI[min_y - 1:max_y + 1, min_x - 1:max_x + 1]
    print(LAI.shape)
    write_tiff(LAI, "LAI")

    # LAI与CI数据应当是可以完全对应的
    # </editor-fold>

    # <editor-fold> 对每个MODIS像元：获取对应的CI值，计算fvc_60与fvc_0；计算其对应ASTER像元的平均LSTs, LSTv，进而计算辐亮度
    # 60度
    theta_60 = np.ones(LAI.shape) * 60
    # 0度
    theta_0 = np.zeros(LAI.shape)

    # 计算FVC，G默认为0.5
    FVC_60 = cal_fvc_gap(LAI, CI, theta_60)
    FVC_0 = cal_fvc_gap(LAI, CI, theta_0)

    # 用于存储植被覆盖度与辐亮度的数组
    BT_60 = np.zeros(LAI.shape, dtype=np.float64)
    BT_0 = np.zeros(LAI.shape, dtype=np.float64)
    # 用于存储构造特征空间的数据的列表，数据用[BT,fvc]记录
    space_data_list = []
    is_valid = np.zeros(LAI.shape, dtype=bool)
    for y_modis in range(LAI.shape[0]):     # 一行
        for x_modis in range(LAI.shape[1]):
            # 对当前MODIS像元
            LSTv = []
            LSTs = []
            # 当前MODIS像元坐标范围
            cur_minLon = geotrans_LAI[0] + geotrans_LAI[1] * (min_x-1+x_modis-0.5)
            cur_maxLon = geotrans_LAI[0] + geotrans_LAI[1] * (min_x-1+x_modis+0.5)
            cur_minLat = geotrans_LAI[3] + geotrans_LAI[5] * (min_y-1+y_modis+0.5)
            cur_maxLat = geotrans_LAI[3] + geotrans_LAI[5] * (min_y-1+y_modis-0.5)
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
                            # 根据平均NDVI值判断是植被还是土壤
                            if cur_ndvi > threshold_NDVI:
                                LSTv.append(lst_aster[y_aster, x_aster])
                            else:
                                LSTs.append(lst_aster[y_aster, x_aster])
                            # 判断是否为阴影（山/云）或水
                            # 计算nir波段反射率方差
                            cur_std = np.std(cur_nir)
                            # print(cur_std)
                            # 针对ASTER三个波段反射率、nir波段反射率方差进行筛选
                            # if (not (np.mean(cur_vis) < 0.1 and np.mean(cur_nir) < 0.15 and np.mean(cur_red) < 0.1)) and \
                            if (not (np.mean(cur_vis) < 0.1 and np.mean(cur_nir) < 0.19 and np.mean(cur_red) < 0.1)) and \
                                    (not (np.mean(cur_vis) > 0.2 and np.mean(cur_nir) > 0.2 and np.mean(cur_red) > 0.2))\
                                    and cur_std < 0.04:
                                is_valid[y_modis, x_modis] = True
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

            # 全图没找到合适的ASTER像元则做特殊处理
            if len(LSTs) == 0:
                if len(LSTv) == 0:
                    # 一个都没有，即不在研究区内，生成特征空间时需去除
                    BT_60[y_modis, x_modis] = 0
                    BT_0[y_modis, x_modis] = 0
                    continue
                # 只是没有土壤像元，给定值
                else:
                    LSTs.append(317)
            # 没有植被像元
            if len(LSTv) == 0:
                LSTv.append(304)

            # 对平均温度进行判断（对水与大部分云的阴影进一步去除）
            if np.mean(LSTs) <= 300:
            # if np.mean(LSTs) <= 290:
                is_valid[y_modis, x_modis] = False

            # 对当前MODIS像元计算辐亮度
            try:
                cur_BTv = lst2BTs(np.mean(LSTv))
                cur_BTs = lst2BTs(np.mean(LSTs))
            except Exception as e:
                print(LSTv)
                print(LSTs)
                cur_BTv = 0
                cur_BTs = 0
                print(e)
            # 原始角度
            BT_60[y_modis, x_modis] = FVC_60[y_modis, x_modis] * cur_BTv + (1 - FVC_60[y_modis, x_modis]) * cur_BTs
            # 垂直方向
            BT_0[y_modis, x_modis] = FVC_0[y_modis, x_modis] * cur_BTv + (1 - FVC_0[y_modis, x_modis]) * cur_BTs

            # 如果是云/水像元则不参与特征空间的构造，否则存入数组
            if is_valid[y_modis, x_modis]:
                space_data_list.append([BT_60[y_modis, x_modis], FVC_60[y_modis, x_modis]])

    # </editor-fold>

    # <editor-fold> 建立特征空间，计算根据特征空间得到的的radiance
    # 从数组中获取用于构建特征空间的数据，并转换为二维ndarray
    print(len(space_data_list))
    BT_space = [x[0] for x in space_data_list]
    BT_space = np.asarray(BT_space)
    BT_space = BT_space.reshape((1, BT_space.shape[0]))
    fvc_space = [x[1] for x in space_data_list]
    fvc_space = np.asarray(fvc_space)
    fvc_space = fvc_space.reshape((1, fvc_space.shape[0]))

    # 出图
    write_tiff(FVC_60, "FVC")
    write_tiff(FVC_0, "FVC_0")
    write_tiff(fvc_space, "FVC_space")
    write_tiff(BT_60, "BT")
    write_tiff(BT_0, "BT_0")
    write_tiff(BT_space, "BT_space")
    write_tiff(is_valid, "is_valid")

    # 建立特征空间
    main_space()
    # </editor-fold>


def main_space():
    """
    得到模拟结果后，进行特征空间相关处理
    :return:
    """
    # 读取相关数据
    ds_BT, BT = open_tiff("pics/BT.tif")
    ds_BT_space, BT_space = open_tiff("pics/BT_space.tif")
    ds_BT_0, BT_0 = open_tiff("pics/BT_0.tif")
    ds_fvc, fvc = open_tiff("pics/FVC.tif")
    ds_fvc_space, fvc_space = open_tiff("pics/FVC_space.tif")
    ds_fvc_0, fvc_0 = open_tiff("pics/FVC_0.tif")
    ds_VZA, VZA = open_tiff("pics/VZA.tif")
    ds_valid, is_valid = open_tiff("pics/is_valid.tif")

    # 生成特征空间
    k1, c1, k2, c2 = getEdges_fvc(BT_space, fvc_space)
    # 出图
    scatter_BTs_fvc(BT_space, fvc_space, k1, c1, k2, c2)
    # 计算特征空间中的顶点
    point_x, point_y = cal_vertex(k1, c1, k2, c2)
    print(point_x, point_y)

    # 根据fvc_0与特征空间计算垂直方向辐亮度
    BT_0_space = np.zeros(BT_0.shape, dtype=np.float64)
    for i in range(BT_0.shape[0]):
        for j in range(BT_0.shape[1]):
            # FVC过大的点直接去除
            if fvc_0[i, j] > point_x or fvc[i, j] > point_x:
                is_valid[i, j] = 0
                continue
            k, c = cal_params(point_y, point_x, BT[i, j], fvc[i, j])
            BT_0_space[i, j] = k * fvc_0[i, j] + c
            if is_valid[i, j] != 0 and np.abs(BT_0_space[i, j] - BT[i, j]) > 0.5:
                print("fvc:\t" + str(fvc[i, j]))
                print("fvc_0:\t" + str(fvc_0[i, j]))
                print("BT:\t" + str(BT[i, j]))
                print("BT_0_spcae:\t" + str(BT_0_space[i, j]))

    # 获取有效数据
    BT_0_space_valid = BT_0_space * is_valid
    BT_valid = BT * is_valid
    BT_0_valid = BT_0 * is_valid
    # <editor-fold> 结果定量分析

    # 计算结果与模拟结果进行对比
    display_hist(BT_0_space_valid - BT_0_valid, "BT_diff_space_0")
    RMSE_BT_space_0 = np.sqrt(metrics.mean_squared_error(BT_0_valid, BT_0_space_valid))
    print("RMSE_BT_space_0:" + str(RMSE_BT_space_0))
    # 根据角度划分的折线图
    calRMSE(BT_0_valid, BT_0_space_valid, VZA, "BT_space_0")

    # 原始数据与模拟结果的对比
    display_hist(BT_0_valid - BT_valid, "BT_diff_0")
    RMSE_BT_0 = np.sqrt(metrics.mean_squared_error(BT_valid, BT_0_valid))
    print("RMSE_BT_0:" + str(RMSE_BT_0))
    calRMSE(BT_valid, BT_0_valid, VZA, "BT_0")

    # 原始数据与特征空间结果的对比
    display_hist(BT_0_space_valid - BT_valid, "BT_diff_space")
    RMSE_BT_space = np.sqrt(metrics.mean_squared_error(BT_valid, BT_0_space_valid))
    print("RMSE_BT_space:" + str(RMSE_BT_space))
    calRMSE(BT_valid, BT_0_space_valid, VZA, "BT_space")

    # </editor-fold>

    # 温度对比
    LST = BTs2lst(BT_valid)
    LST_0 = BTs2lst(BT_0_valid)
    LST_space_0 = BTs2lst(BT_0_space_valid)

    calRMSE(LST_0, LST_space_0, VZA, "LST_space_0")
    calRMSE(LST_0, LST, VZA, "LST_0")
    calRMSE(LST_space_0, LST, VZA, "LST_space")

    # 出图
    write_tiff(BT_0_space_valid, "BT_0_space_valid")
    write_tiff((BT_0_space_valid-BT_0_valid), "BT_diff_0_space")
    write_tiff(LST, "LST_valid")
    write_tiff(LST_0, "LST_0_valid")
    write_tiff(LST_space_0, "LST_space_0_valid")


def test():
    ds, CI = open_tiff(file_CI)
    print(ds.GetGeoTransform())
    print(help(ds))
    print(ds.GetProjection())
    # GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563,AUTHORITY["EPSG","7030"]],
    # AUTHORITY["EPSG","6326"]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433,AUTHORITY["EPSG","9122"]],
    # AXIS["Latitude",NORTH],AXIS["Longitude",EAST],AUTHORITY["EPSG","4326"]]


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
    # cal_mean_LSTvs()
    # main_hdf()
    # main_space()
    cal_mean_LSTvs()
    # display_LUT()
    # sensitivity_VZA()