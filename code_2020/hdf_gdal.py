import pandas as pd
import numpy as np
from enum import Enum
from scipy import stats, misc
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn import metrics
from PIL import Image
from osgeo import gdal, gdalconst
from code_2020.kernals import Ross_thick, LI_SparseR
import math


# ************************************** definitions ********************************


class FileType(Enum):
    Reflectance = 0
    LST = 1

# my colors
color_darkBlue = (80, 140, 170)

# ## file paths
# LUT
file_LUT = "LUT.txt"
# # hdf
lst_fileName = "MOD11_L2.0812.0340.hdf"
reflectance_fileName = "data/MOD09.0812.0340.hdf"
BRDF_fileName = "MCD43A1.hdf"
# # tiff
# MOD09
file_MOD09_1 = "/MOD09/MOD09GA.sur_refl_b01_1.tif"
file_MOD09_2 = "/MOD09/MOD09GA.sur_refl_b02_1.tif"
file_MOD09_SZA = "/MOD09/MOD09GA.SolarZenith_1.tif"
file_MOD09_SAA = "/MOD09/MOD09GA.SolarAzimuth_1.tif"
file_MOD09_VZA = "/MOD09/MOD09GA.SensorZenith_1.tif"
file_MOD09_VAA =  "/MOD09/MOD09GA.SensorAzimuth_1.tif"
# MOD11
file_MOD11 = "/MOD11/MOD11_L2.LST.tif"
# MCD43A1
file_MCD43A1_B1_1 = "/MCD43/MCD43A1.BRDF_Albedo_Parameters_Band1.Num_Parameters_01.tif"
file_MCD43A1_B1_2 = "/MCD43/MCD43A1.BRDF_Albedo_Parameters_Band1.Num_Parameters_02.tif"
file_MCD43A1_B1_3 = "/MCD43/MCD43A1.BRDF_Albedo_Parameters_Band1.Num_Parameters_03.tif"
file_MCD43A1_B2_1 = "/MCD43/MCD43A1.BRDF_Albedo_Parameters_Band2.Num_Parameters_01.tif"
file_MCD43A1_B2_2 = "/MCD43/MCD43A1.BRDF_Albedo_Parameters_Band2.Num_Parameters_02.tif"
file_MCD43A1_B2_3 = "/MCD43/MCD43A1.BRDF_Albedo_Parameters_Band2.Num_Parameters_03.tif"

# constants
h = 6.62606896 * 1e-34      # Planck constant
c = 299792458               # speed of light
k = 1.3806 * 1e-23          # Boltzmann constant

# NDVIs = 0.156
# NDVIv = 0.86

# bbox info
minLon = 105.78432759485935
maxLon = 111.88621521850175
minLat = 36.95825861049653
maxLat = 41.45847714860637


# ************************************** simple calculations ****************************


def cal_Planck(Ts, wv):
    """
    Func: apply the Planck's law and return the radiance
    :param Ts:
    :param wv: unit: m
    :return:
    """
    # flux density
    # result = 2 * np.pi * h * c * c / (wv ** 5) / (np.exp(h * c / (wv * k * Ts)))

    # radiance
    result = 2 * h * c * c / ((wv ** 5) * 1e6 * (np.exp(h * c / (wv * k * Ts)) - 1))
    return result


def cal_NDVI(data_red: np.ndarray, data_nir: np.ndarray):
    """
    To calculate the NDVI of each pixel and return the result.
    :param data_red:
    :param data_nir:
    :return:
    """
    result = ((data_nir - data_red) / (data_nir + data_red))
    # 进行异常值处理
    result[np.isnan(result)] = 0
    result[result > 1] = 1
    result[result < 0] = 0
    # check the histogram
    hist, edges = np.histogram(result, 100)
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


# ************************************** read files ************************************


def open_1(fileName):
    # 打开文件
    df = pd.read_hdf(fileName)
    # 表头信息
    head = df.columns
    print(head)


def open_gdal(fileName):
    """
    Func: To open a gdal file
    :param fileName:
    :return: SDS
    """
    hdf = gdal.Open(fileName)
    # subDataset
    subdatasets = hdf.GetSubDatasets()
    for subDataset in subdatasets:
        print(subDataset)

    # SDSs
    sdsdict = hdf.GetMetadata('SUBDATASETS')
    sdslist = [sdsdict[k] for k in sdsdict.keys() if '_NAME' in k]
    sds = []
    for n in sdslist:
        sds.append(gdal.Open(n))
    print(len(sds))

    return sds


def get_LUT(fileName):
    """
    Func: to get the lookup table and save in a list
    :param fileName: file name of the LUT text file
    :return: the list of LUT
    """
    with open(fileName, "r") as file:
        lines = file.readlines()
        LUTlist = [float(lines[i].split()[1]) for i in range(1, len(lines))]
        print(LUTlist)
    return LUTlist


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


def write_tiff(data: np.ndarray, filename: str, date="0812"):
    """

    :param data:
    :param filename:
    :return:
    """
    Image.fromarray(data).save("pics" + date + "/" + filename + '.tif')
    # misc.imsave(filename + ".tif", data)


# ********************************** handle files of different types ***********************


def handle_Reflectance(fileName):
    """

    :param fileName:
    :return:
    """
    sds = open_gdal(fileName)
    # MOD09
    # data of bands
    red = sds[16].ReadAsArray() * 0.0001
    nir = sds[17].ReadAsArray() * 0.0001
    blue = sds[18].ReadAsArray() * 0.0001
    green = sds[19].ReadAsArray() * 0.0001

    # calculate the NDVI
    # cal_NDVI(red, nir)

    # true color map
    show_trueColor(red, blue, green)


def handle_LST(fileName):
    """

    :param fileName:
    :return:
    """
    sds = open_gdal(fileName)
    # MOD11_L2
    lst = sds[0].ReadAsArray() * 0.02
    lst[lst < 300] = 300
    # check the histogram
    hist, edges = np.histogram(lst, 100)
    print(hist)
    print(edges)
    display(lst, "LST")


def handle_BRDF(fileName):
    sds = open_gdal(fileName)
    BRDF_red = sds[10].ReadAsArray() * 0.001
    # TODO: NIR波段使用BRDF产品给出的NIR还是与反射率一致的band2？
    # BRDF_nir = sds[18].ReadAsArray() * 0.001
    BRDF_nir = sds[11].ReadAsArray() * 0.001

    print(BRDF_red)
    print(BRDF_nir)


# ************************************** algorithms *********************************


def changeResolution(brdf: np.ndarray):
    """
    Func: to transform the 500m BRDF data into 1km data
    :param brdf:
    :return: ndarray of double resolution
    """
    data = [[] for i in range(int(brdf.shape[0]/2))]
    # x: height
    for x in range(int(brdf.shape[0]/2)):
        # y: width
        for y in range(int(brdf.shape[1]/2)):
            data[x].append(np.mean(brdf[2*x,2*y], brdf[2*x+1, 2*y+1], brdf[2*x, 2*y+1], brdf[2*x, 2*y+1]))
    return np.array(data)


def bilinearInterpolation(data: np.ndarray):
    """
    To do linear interpolation to a set of angle data and
    :param data:
    :return:
    """
    # 原始与目标图像大小
    width_original = data.shape[1]
    height_original = data.shape[0]
    width_target = 3137
    height_target = 2132
    # 结果图像数组
    result = np.zeros(shape=(height_target, width_target), dtype=np.float64)

    # 进行双线性插值
    for i in range(height_target):
        for j in range(width_target):
            row = (i / height_target) * height_original
            col = (j / width_target) * width_original
            row_int = int(row)
            col_int = int(col)
            u = row - row_int
            v = col - col_int
            if row_int == height_original-1 or col_int == width_original-1:
                row_int -= 1
                col_int -= 1
            result[i, j] = (1-u) * (1-v) * data[row_int, col_int] + (1-u) * v * data[row_int, col_int+1] + u * (1-v) * \
                            data[row_int+1, col_int] + u * v * data[row_int+1, col_int+1]

    return result


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


def getEdges_fvc(lst: np.ndarray, ndvi: np.ndarray):
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
    interval_num = 30
    subinterval_num = 6
    # do the statics
    Ts = [[[] for j in range(subinterval_num)] for i in range(interval_num)]
    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            if ndvi[i, j] <= 0 or ndvi[i, j] >= 1:
                continue
            if lst[i, j] >= LST_aver + 3.25 * LST_std or lst[i, j] <= LST_aver - 2.5 * LST_std:
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

        if len(maxTs) == 0:
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


def cal_ref_BRDF(SZA, SAA, VZA, VAA, iso, vol, geo):
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

    print(np.min(result_vol))
    print(np.min(result_geo))
    print(np.min(iso))
    print(np.max(result_vol))
    print(np.max(result_geo))
    print(np.max(iso))
    # display(result_vol, "VOL_4")
    # display(result_geo, "GEO_4")
    # display(np.array(iso), "ISO_2")

    # display(result, "BRDF_band1_2")
    return np.array(result)


def calParams(ndvi1, lst1, ndvi2, lst2, LUT):
    """
    Calculate the parameters in fvc-BTs
    :param ndvi1:
    :param lst1:
    :param ndvi2:
    :param lst2:
    :param LUT
    :return: const and slope
    """
    fvc1 = ((ndvi1 - NDVIs) / (NDVIv - NDVIs)) ** 2
    fvc2 = ((ndvi2 - NDVIs) / (NDVIv - NDVIs)) ** 2
    BTs1 = LUT[int((lst1 - 240) * 10)]
    BTs2 = LUT[int((lst2 - 240) * 10)]
    slope = (BTs1 - BTs2) / (fvc1 - fvc2)
    const = BTs1 - fvc1 * slope
    return slope, const


def lst2BTs(lst: np.ndarray):
    """
    transform lst data into BTs data
    :param lst:
    :return:
    """
    LUT = get_LUT(file_LUT)
    shape = lst.shape
    BTs = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            index = int((lst[i, j] - 240) * 10)
            BTs[i, j] = LUT[index]

    return BTs


def BTs2lst(BTs: np.ndarray):
    """
    transform BTs data into lst data
    :param BTs:
    :return:
    """
    LUT = get_LUT(file_LUT)
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

    return lst


def calRMSE(lst: np.ndarray, lst_0: np.ndarray, VZA: np.ndarray, dateStr: str):
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
            if lst[i, j] > 270:
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
        print(i)
        print(len(lst_intervals[i]))
        if len(lst_intervals[i]) == 0:
            RMSEs.append(RMSEs[-1])
        else:
            RMSE = np.sqrt(metrics.mean_squared_error(np.array(lst0_intervals[i]), np.array(lst_intervals[i])))
            print(RMSE)
            RMSEs.append(RMSE)

    x = range(minVZA, maxVZA + 1, 1)
    print(x)
    plt.plot(x, RMSEs)
    plt.xlabel("VZA")
    plt.ylabel("RMSE")
    plt.savefig("pics" + dateStr +  "/RMSE_VZA_1.png")
    plt.show()

    # # 以5为区间长度，根据原始VZA分为若干个区间
    # interval_num = math.ceil(maxVZA/3)
    # print(interval_num)
    # lst_intervals = [[] for i in range(interval_num)]
    # lst0_intervals = [[] for i in range(interval_num)]
    # print(lst_intervals[0])
    # for i in range(lst.shape[0]):
    #     for j in range(lst.shape[1]):
    #         try:
    #             index = int(VZA[i, j] / 3)
    #             lst_intervals[index].append(lst[i, j])
    #             lst0_intervals[index].append(lst_0[i, j])
    #         except Exception as e:
    #             print(e)
    #             print(VZA[i, j])
    #             print(index)
    #
    #
    # # 绘制折线图
    # # x = [2.5+5*i for i in range(len(lst_intervals))]
    # x = [1.5+3*i for i in range(len(lst_intervals))]


def cal_NDVIvs(ndvi_55: np.ndarray, ndvi_60: np.ndarray):
    """
    根据multi-angle算法计算每个像元的NDVIv、NDVIs，输入为55与60度的NDVI，对每个像元进行计算
    :param ndvi_55:
    :param ndvi_60:
    :return:
    """

    pass


def cal_FVC_multi_angle(ref1_55: np.ndarray, ref1_60: np.ndarray, ref2_55: np.ndarray, ref2_60: np.ndarray,
                        ndvi_0: np.ndarray):
    """
    根据两个波段55与60度的反射率，计算55与60度的NDVI，根据multi-angle算法计算每个像元的NDVIv、NDVIs，得到FVC
    :param ref1_55:
    :param ref1_60:
    :param ref2_55:
    :param ref2_60:
    :return:
    """
    ndvi_55 = cal_NDVI(ref1_55, ref2_55)
    ndvi_60 = cal_NDVI(ref1_60, ref2_60)
    NDVIv_cal, NDVIs_cal = cal_NDVIvs(ndvi_55, ndvi_60)
    fvc = cal_fvc(ndvi_0, NDVIv=NDVIv_cal, NDVIs=NDVIs_cal)
    return fvc


# ************************************** display **************************************


def show_trueColor(red, blue, green):
    """
    To display the true color map of the given data
    :return:
    """
    red_ = np.expand_dims(red, 0)
    blue_ = np.expand_dims(blue, 0)
    green_ = np.expand_dims(green, 0)

    # check the histogram
    hist, edges = np.histogram(red, 100)
    print(hist)
    print(edges)

    # true color array
    trueColor = np.concatenate([red_, green_, blue_])
    # 转换为matplotlib中的数据格式  即：(C, H, W) -> (H, W, C)
    trueColor = trueColor.transpose((1, 2, 0))
    display(trueColor, "trueColor")


def display(data, title, date="0812", cmap=None):
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
    # 绘制区域矩形
    # axis = plt.gca()
    # axis.add_patch(patches.Rectangle((420, 630), 490, 470, edgecolor='r', facecolor='none'))
    # axis.add_patch(patches.Rectangle((1170, 650), 610, 450, edgecolor='r', facecolor='none'))

    # 去掉坐标轴刻度
    plt.xticks([])
    plt.yticks([])
    plt.savefig("pics" + date + "/" + title + ".png", dpi=300)
    plt.show()

    # PIL: only 1 band
    # im = Image.fromarray(data)
    # im.show()
    # im.save("band_16.png")


def display_lst(data, title, date="0812"):
    plt.imshow(data, cmap=plt.get_cmap('bwr'))

    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    plt.savefig("pics" + date + "/" + title + ".png")
    plt.show()


def display_hist(data, title, date="0812"):
    data = data.reshape(-1)
    print(data.shape)
    plt.hist(data, bins=100, density=True)
    plt.xlabel("difference")
    plt.ylabel("frequency")
    plt.savefig("pics" + date + "/" + title + "_hist.png")
    plt.show()


def scatter_LST_NDVI(lst, ndvi, k1, c1, k2, c2, date):
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

    # # 特殊值处理：温度
    # lst = lst.reshape(-1)
    # ndvi = ndvi.reshape(-1)
    # for i in range(len(lst)):
    #     if lst < 270:

    plt.scatter(ndvi, lst, color="cornflowerblue", s=1.)
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
    plt.xlabel("NDVI")
    plt.ylabel("LST")
    # plt.ylim(275, 340)
    plt.ylim(np.min(lst) - 5, np.max(lst) + 5)
    # plt.xlim(0, 3)
    # plt.savefig("fvc_BTs_edges.png")
    plt.savefig("pics" + date + "/" + "NDVI_LST_edges.png")
    plt.show()


def scatter_BTs_fvc_2(lst, ndvi, k1, c1, k2, c2, date):
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

    plt.scatter(ndvi, lst, color="cornflowerblue", s=1.)
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
    # plt.ylim(4.5, 10.5)
    plt.ylim(np.min(lst) - 0.5, np.max(lst) + 0.5)
    # plt.xlim(0, 3)
    # plt.savefig("fvc_BTs_edges.png")
    plt.savefig("pics" + date + "/" + "BTs_fvc_edges.png")
    plt.show()


def scatter_BTs_fvc(lst, ndvi):
    """
    To draw the scatter
    :param lst_file:
    :param ndvi_file:
    :return:
    """
    # display the figure
    # get a subset of data
    # sub_lst = lst[420:910, 630:1100]
    # sub_ndvi = ndvi[420:910, 630:1100]
    # reshape to an one-dimension array
    sub_lst = lst.reshape(-1)
    sub_ndvi = ndvi.reshape(-1)

    # calculate fvc
    fvc = cal_fvc(sub_ndvi)
    # calculate BTs
    LUT = get_LUT(file_LUT)
    BTs = []
    print(len(sub_lst))
    print(sub_lst.shape)
    for i in range(len(sub_lst)):
        index = int((sub_lst[i] - 240) * 10)
        BTs.append(LUT[index])

    # scatter
    plt.scatter(fvc, BTs, color="royalblue", s=1.)

    plt.xlabel("fv")
    plt.ylabel("B(Ts)")
    plt.ylim(4.5, 10.5)
    # plt.xlim(0, 1.22)
    plt.savefig("fvc-BTs.png")
    plt.show()


def main():
    ### obtain data
    # open the files and get the data
    ds_lst, lst = open_tiff(file_MOD11)
    sds_reflectance = open_gdal(reflectance_fileName)
    sds_BRDF = open_gdal(BRDF_fileName)
    # LST data
    # lst = sds_lst[0].ReadAsArray() * 0.02
    lst = lst * 0.02
    lst[lst < 270] = 270
    # NDVI data
    red = sds_reflectance[16].ReadAsArray() * 0.0001
    display(red, "ref_redband")
    nir = sds_reflectance[17].ReadAsArray() * 0.0001
    ndvi = cal_NDVI(red, nir)

    ### space of BTs-fvc
    # scatter_BTs_fvc(lst, ndvi)

    ### edges
    ## method 1
    # get the dry edge: LST = k * NDVI + c
    k1, c1, k2, c2 = getEdges(lst, ndvi)
    # wet edge: LST = k + c

    ## method 2
    # k1, c1, c2 = getEdges_2(lst, ndvi)
    # k2 = 0

    # plot the dots and edges
    scatter_LST_NDVI(lst, ndvi, k1, c1, k2, c2)

    # get the coordinate of the crossover point of the two edges
    point_NDVI, point_LST = cal_vertex(k1, c1, k2, c2)

    ### calculate FVC_0 from BRDF


# def getNDVItif():
#     ds_ndvi, ndvi = open_tiff(file_MOD13Q1)
#     ndvi = ndvi[854:1304, 2496:3106] * 0.0001
#
#     write_tiff(ndvi, "RawNDVI")
#     display(ndvi, "RawNDVI")


def main_new():
    ### obtain data
    # open the files and get the data
    ds_lst, lst = open_tiff(file_MOD11)
    ds_ref_1, ref_1 = open_tiff(file_MOD09_1)
    ds_ref_2, ref_2 = open_tiff(file_MOD09_2)
    ds_SZA, SZA = open_tiff(file_MOD09_SZA)
    ds_SAA, SAA = open_tiff(file_MOD09_SAA)
    ds_VZA, VZA = open_tiff(file_MOD09_VZA)
    ds_brdf_B1_1, brdf_B1_1 = open_tiff(file_MCD43A1_B1_1)
    ds_brdf_B1_2, brdf_B1_2 = open_tiff(file_MCD43A1_B1_2)
    ds_brdf_B1_3, brdf_B1_3 = open_tiff(file_MCD43A1_B1_3)
    ds_brdf_B2_1, brdf_B2_1 = open_tiff(file_MCD43A1_B2_1)
    ds_brdf_B2_2, brdf_B2_2 = open_tiff(file_MCD43A1_B2_2)
    ds_brdf_B2_3, brdf_B2_3 = open_tiff(file_MCD43A1_B2_3)
    LUT = get_LUT(file_LUT)
    # LST data
    lst = lst * 0.02
    lst[lst < 270] = 270
    geotrans_lst = ds_lst.GetGeoTransform()
    print(geotrans_lst)
    # NDVI data
    ref_1 = ref_1 * 0.0001
    ref_2 = ref_2 * 0.0001
    ndvi = cal_NDVI(ref_1, ref_2)

    # georeans_ref = ds_ref_1.GetGeoTransform()
    # print(georeans_ref)
    # # BRDF
    # georeans_brdf = ds_brdf_B1_1.GetGeoTransform()
    # print(georeans_brdf)
    # print(brdf_B1_1.shape)

    # # lst图像中的bbox对应像元位置信息
    # lst_left = 1170
    # lst_right = 1780
    # lst_top = 650
    # lst_bottom = 1100
    # # 整个区域的经纬度bbox信息；x经度，y纬度
    # minx = geotrans_lst[0] + lst_left * geotrans_lst[1]
    # maxx = geotrans_lst[0] + lst_right * geotrans_lst[1]
    # maxy = geotrans_lst[3] + lst_top * geotrans_lst[5]
    # miny = geotrans_lst[3] + lst_bottom * geotrans_lst[5]
    # # brdf及反射率图像中的bbox信息
    # ref_left = (minx - georeans_ref[0]) / georeans_ref[1]
    # ref_right = (maxx - georeans_ref[0]) / georeans_ref[1]
    # ref_top = (maxy - georeans_ref[3]) / georeans_ref[5]
    # ref_bottom = (miny - georeans_ref[3]) / georeans_ref[5]

    # print(minx, maxx, maxy, miny)

    # 对数据进行裁剪
    # print(lst.shape)
    lst = lst[650:1100, 1170:1780]
    ref_1 = ref_1[854:1304, 2496:3106]
    ref_2 = ref_2[854:1304, 2496:3106]
    ndvi = ndvi[854:1304, 2496:3106]
    SZA = SZA[854:1304, 2496:3106] * 0.01
    SAA = SAA[854:1304, 2496:3106] * 0.01
    VZA = VZA[854:1304, 2496:3106] * 0.01
    brdf_B1_1 = brdf_B1_1[854:1304, 2496:3106] * 0.001
    brdf_B1_2 = brdf_B1_2[854:1304, 2496:3106] * 0.001
    brdf_B1_3 = brdf_B1_3[854:1304, 2496:3106] * 0.001
    brdf_B2_1 = brdf_B2_1[854:1304, 2496:3106] * 0.001
    brdf_B2_2 = brdf_B2_2[854:1304, 2496:3106] * 0.001
    brdf_B2_3 = brdf_B2_3[854:1304, 2496:3106] * 0.001

    print("ndvi: " + str(np.mean(ndvi)))
    display(ndvi, "NDVI")

    fvc = cal_fvc(ndvi)
    print("fvc: " + str(np.mean(fvc)))
    display(fvc, "fvc")
    # display(VZA, "VZA")

    BTs = lst2BTs(lst)

    ref_1[ref_1 > 1] = 1
    ref_2[ref_2 > 1] = 1
    ref_1[ref_1 < 0] = 0
    ref_2[ref_2 < 0] = 0

    print("ref1: " + str(np.mean(ref_1)))
    print("ref2: " + str(np.mean(ref_2)))
    display(ref_1, "band1")
    display(ref_2, "band2")
    print("lst: " + str(np.mean(lst)))
    display_lst(lst, "LST")
    # display_hist(lst, "LST")
    # display(SZA, "SZA")
    # display(VZA, "VZA")

    # 保存lst
    write_tiff(lst, "LST")

    ### space of BTs-fvc
    # scatter_BTs_fvc(lst, ndvi)

    ### edges
    # ndvi-lst
    k1, c1, k2, c2 = getEdges(lst, ndvi)

    # # plot the dots and edges
    # scatter_LST_NDVI(lst, ndvi, k1, c1, k2, c2)

    # get the coordinate of the crossover point of the two edges
    point_NDVI, point_LST = cal_vertex(k1, c1, k2, c2)

    # # BTs-fvc edges
    # k1_, c1_, k2_, c2_ = getEdges_fvc(BTs, fvc)
    # # plot
    # scatter_BTs_fvc_2(BTs, fvc, k1_, c1_, k2_, c2_)

    ### calculate FVC_0 from BRDF
    VZA_0 = np.zeros(lst.shape)
    VAA_0 = np.zeros(lst.shape)
    ref_1_brdf = cal_ref_BRDF(SZA, SAA, VZA_0, VAA_0, brdf_B1_1, brdf_B1_2, brdf_B1_3)
    ref_2_brdf = cal_ref_BRDF(SZA, SAA, VZA_0, VAA_0, brdf_B2_1, brdf_B2_2, brdf_B2_3)
    ref_1_brdf[ref_1_brdf > 0.8] = 0.8
    ref_1_brdf[ref_1_brdf < 0] = 0
    ref_2_brdf[ref_2_brdf > 0.8] = 0.8
    ref_2_brdf[ref_2_brdf < 0] = 0

    print("ref_1_0: " + str(np.mean(ref_1_brdf)))
    print("ref_2_0: " + str(np.mean(ref_2_brdf)))

    display(ref_1_brdf, "BRDF_band1")
    display(ref_2_brdf, "BRDF_band2")

    # 保存ref
    write_tiff(ref_1, "ref_1")
    write_tiff(ref_2, "ref_2")

    # 保存ref_brdf
    write_tiff(ref_1_brdf, "ref_brdf_1")
    write_tiff(ref_2_brdf, "ref_brdf_2")
    # 保存ref_diff
    ref_1_diff = ref_1_brdf - ref_1
    write_tiff(ref_1_diff, "ref_1_diff")
    display(ref_1_diff, "ref_1_diff")
    ref_2_diff = ref_2_brdf - ref_2
    write_tiff(ref_2_diff, "ref_2_diff")
    display(ref_2_diff, "ref_2_diff")

    ndvi_brdf = cal_NDVI(ref_1_brdf, ref_2_brdf)
    ndvi_diff = ndvi_brdf - ndvi
    RMSE_ndvi = np.sqrt(metrics.mean_squared_error(ndvi_diff, ndvi))
    print(RMSE_ndvi)
    print("RMSE_ndvi")
    write_tiff(ndvi, "NDVI")
    write_tiff(ndvi_brdf, "NDVI_BRDF")
    print("NDVI_0: " + str(np.mean(ndvi_brdf)))
    display(ndvi_brdf, "NDVI_BRDF")
    write_tiff(ndvi_diff, "NDVI_diff")
    display(ndvi_diff, "NDVI_diff")
    display_hist(ndvi_diff, "NDVI_diff")

    fvc_brdf = cal_fvc(ndvi_brdf)
    fvc_diff = fvc_brdf - fvc
    RMSE_fvc = np.sqrt(metrics.mean_squared_error(fvc_brdf, fvc))
    print("RMSE_fvc")
    print(RMSE_fvc)
    write_tiff(fvc, "FVC")
    print("fvc_0: " + str(np.mean(fvc_brdf)))
    write_tiff(fvc_brdf, "FVC_BRDF")
    display(fvc_brdf, "FVC_BRDF")
    write_tiff(fvc_diff, "FVC_diff")
    display(fvc_diff, "FVC_diff")
    display_hist(fvc_diff, "FVC_diff")

    ### get LST_0
    # 每个像元：计算FVC-BTs图像中的对应参数，然后计算得到校正后的BTs
    BTs_brdf = np.zeros(lst.shape, dtype=np.float64)
    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            k, const = calParams(ndvi[i, j], lst[i, j], point_NDVI, point_LST, LUT)
            # slope_ndvi_lst = (lst[i, j] - point_LST) / (ndvi[i, j] - point_NDVI) if ndvi[i, j] != point_NDVI else 0
            # if slope_ndvi_lst == 0:
            #     continue
            BTs_brdf[i, j] = k * fvc_brdf[i, j] + const


    # 计算校正后的LST
    lst_0 = BTs2lst(BTs_brdf)
    lst_0[lst_0 < 270] = 270
    write_tiff(lst_0, "LST_0")
    print("lst_0: " + str(np.mean(lst_0)))
    display_lst(lst_0, "LST_0")
    display_hist(lst_0, "LST_0")

    # 差值
    diff = lst_0 - lst
    write_tiff(diff, "LST_diff")
    display_lst(diff, "LST_diff")
    display_hist(diff, "LST_diff")

    # 计算去云的LST
    lst_all = np.zeros(lst.shape, dtype=np.float64)
    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            if lst_0[i, j] > 285:
                lst_all[i, j] = lst_0[i, j]
            else:
                lst_all[i, j] = lst


    # RMSE
    RMSE = np.sqrt(metrics.mean_squared_error(lst, lst_0))
    print(RMSE)

    # 根据原始观测数据划分角度区间，分别算RMSE并绘制折线图
    calRMSE(lst, lst_0, VZA)


def main_new2(dateStr):
    ### obtain data
    # open the files and get the data
    ds_lst, lst = open_tiff("data" + dateStr + file_MOD11)
    ds_ref_1, ref_1 = open_tiff("data" + dateStr + file_MOD09_1)
    ds_ref_2, ref_2 = open_tiff("data" + dateStr + file_MOD09_2)
    ds_SZA, SZA = open_tiff("data" + dateStr + file_MOD09_SZA)
    ds_SAA, SAA = open_tiff("data" + dateStr + file_MOD09_SAA)
    ds_VZA, VZA = open_tiff("data" + dateStr + file_MOD09_VZA)
    ds_brdf_B1_1, brdf_B1_1 = open_tiff("data" + dateStr + file_MCD43A1_B1_1)
    ds_brdf_B1_2, brdf_B1_2 = open_tiff("data" + dateStr + file_MCD43A1_B1_2)
    ds_brdf_B1_3, brdf_B1_3 = open_tiff("data" + dateStr + file_MCD43A1_B1_3)
    ds_brdf_B2_1, brdf_B2_1 = open_tiff("data" + dateStr + file_MCD43A1_B2_1)
    ds_brdf_B2_2, brdf_B2_2 = open_tiff("data" + dateStr + file_MCD43A1_B2_2)
    ds_brdf_B2_3, brdf_B2_3 = open_tiff("data" + dateStr + file_MCD43A1_B2_3)
    LUT = get_LUT(file_LUT)
    # LST data
    # if dateStr == "0812":
    lst = lst * 0.02
    lst[lst < 260] = 260
    geotrans_lst = ds_lst.GetGeoTransform()
    geotrans_ref1 = ds_ref_1.GetGeoTransform()
    proj_lst = ds_lst.GetProjection()
    proj_ref1 = ds_ref_1.GetProjection()
    print(geotrans_lst)
    print(geotrans_ref1)
    print(proj_lst)
    print(proj_ref1)
    # NDVI data
    ref_1 = ref_1 * 0.0001
    ref_2 = ref_2 * 0.0001
    ndvi = cal_NDVI(ref_1, ref_2)

    # georeans_ref = ds_ref_1.GetGeoTransform()
    # print(georeans_ref)
    # # BRDF
    # georeans_brdf = ds_brdf_B1_1.GetGeoTransform()
    # print(georeans_brdf)
    # print(brdf_B1_1.shape)

    # # lst图像中的bbox对应像元位置信息
    # lst_left = 1170
    # lst_right = 1780
    # lst_top = 650
    # lst_bottom = 1100
    # 整个区域的经纬度bbox信息；x经度，y纬度
    # # brdf及反射率图像中的bbox信息
    # ref_left = (minx - georeans_ref[0]) / georeans_ref[1]
    # ref_right = (maxx - georeans_ref[0]) / georeans_ref[1]
    # ref_top = (maxy - georeans_ref[3]) / georeans_ref[5]
    # ref_bottom = (miny - georeans_ref[3]) / georeans_ref[5]

    display_lst(lst, "LST_all", dateStr)

    # 对数据进行裁剪
    # print(lst.shape)
    ref_1 = ref_1[854:1304, 2496:3106]
    ref_2 = ref_2[854:1304, 2496:3106]
    ndvi = ndvi[854:1304, 2496:3106]
    SZA = SZA[854:1304, 2496:3106] * 0.01
    SAA = SAA[854:1304, 2496:3106] * 0.01
    VZA = VZA[854:1304, 2496:3106] * 0.01
    brdf_B1_1 = brdf_B1_1[854:1304, 2496:3106] * 0.001
    brdf_B1_2 = brdf_B1_2[854:1304, 2496:3106] * 0.001
    brdf_B1_3 = brdf_B1_3[854:1304, 2496:3106] * 0.001
    brdf_B2_1 = brdf_B2_1[854:1304, 2496:3106] * 0.001
    brdf_B2_2 = brdf_B2_2[854:1304, 2496:3106] * 0.001
    brdf_B2_3 = brdf_B2_3[854:1304, 2496:3106] * 0.001

    lst_left = (minLon - geotrans_lst[0]) / geotrans_lst[1]
    lst_right = (maxLon - geotrans_lst[0]) / geotrans_lst[1]
    lst_top = (maxLat - geotrans_lst[3]) / geotrans_lst[5]
    lst_bottom = (minLat - geotrans_lst[3]) / geotrans_lst[5]
    print(lst_left, lst_right, lst_top, lst_bottom)
    minx = geotrans_lst[0] + lst_left * geotrans_lst[1]
    maxx = geotrans_lst[0] + lst_right * geotrans_lst[1]
    maxy = geotrans_lst[3] + lst_top * geotrans_lst[5]
    miny = geotrans_lst[3] + lst_bottom * geotrans_lst[5]
    print(minx, maxx, maxy, miny)
    # 确保区域大小为(450, 610)
    lst_bottom += 450 - (int(lst_bottom+0.5) - int(lst_top+0.5))
    lst_right += 610 - (int(lst_right+0.5) - int(lst_left+0.5))

    lst = lst[int(lst_top+0.5):int(lst_bottom+0.5), int(lst_left+0.5):int(lst_right+0.5)]
    print(lst.shape)

    print("ndvi: " + str(np.mean(ndvi)))
    display(ndvi, "NDVI", dateStr)
    display_hist(ndvi, "NDVI", dateStr)

    fvc = cal_fvc(ndvi)
    print("fvc: " + str(np.mean(fvc)))
    display(fvc, "fvc", dateStr)
    display_hist(fvc, "fvc", dateStr)
    display(VZA, "VZA", dateStr)

    BTs = lst2BTs(lst)

    ref_1[ref_1 > 1] = 1
    ref_2[ref_2 > 1] = 1
    ref_1[ref_1 < 0] = 0
    ref_2[ref_2 < 0] = 0

    print("ref1: " + str(np.mean(ref_1)))
    print("ref2: " + str(np.mean(ref_2)))
    display(ref_1, "band1", dateStr)
    display(ref_2, "band2", dateStr)
    print("lst: " + str(np.mean(lst)))
    display_lst(lst, "LST", dateStr)
    display_hist(lst, "LST_hist", dateStr)
    # display_hist(lst, "LST")
    # display(SZA, "SZA")
    # display(VZA, "VZA")

    # 保存lst
    write_tiff(lst, "LST", dateStr)

    ### space of BTs-fvc
    # scatter_BTs_fvc(lst, ndvi)

    ### edges
    # ndvi-lst
    k1, c1, k2, c2 = getEdges(lst, ndvi)

    # plot the dots and edges
    scatter_LST_NDVI(lst, ndvi, k1, c1, k2, c2, dateStr)

    # get the coordinate of the crossover point of the two edges
    point_NDVI, point_LST = cal_vertex(k1, c1, k2, c2)

    # BTs-fvc edges
    k1_, c1_, k2_, c2_ = getEdges_fvc(BTs, fvc)
    # plot
    scatter_BTs_fvc_2(BTs, fvc, k1_, c1_, k2_, c2_, dateStr)

    ### calculate FVC_0 from BRDF
    VZA_0 = np.zeros(lst.shape)
    VAA_0 = np.zeros(lst.shape)
    ref_1_brdf = cal_ref_BRDF(SZA, SAA, VZA_0, VAA_0, brdf_B1_1, brdf_B1_2, brdf_B1_3)
    ref_2_brdf = cal_ref_BRDF(SZA, SAA, VZA_0, VAA_0, brdf_B2_1, brdf_B2_2, brdf_B2_3)
    ref_1_brdf[ref_1_brdf > 0.8] = 0.8
    ref_1_brdf[ref_1_brdf < 0] = 0
    ref_2_brdf[ref_2_brdf > 0.8] = 0.8
    ref_2_brdf[ref_2_brdf < 0] = 0

    print("ref_1_0: " + str(np.mean(ref_1_brdf)))
    print("ref_2_0: " + str(np.mean(ref_2_brdf)))

    display(ref_1_brdf, "BRDF_band1", dateStr)
    display(ref_2_brdf, "BRDF_band2", dateStr)

    # 保存ref
    write_tiff(ref_1, "ref_1", dateStr)
    write_tiff(ref_2, "ref_2", dateStr)

    # 保存ref_brdf
    write_tiff(ref_1_brdf, "ref_brdf_1", dateStr)
    write_tiff(ref_2_brdf, "ref_brdf_2", dateStr)
    # 保存ref_diff
    ref_1_diff = ref_1_brdf - ref_1
    write_tiff(ref_1_diff, "ref_1_diff", dateStr)
    display(ref_1_diff, "ref_1_diff", dateStr)
    ref_2_diff = ref_2_brdf - ref_2
    write_tiff(ref_2_diff, "ref_2_diff", dateStr)
    display(ref_2_diff, "ref_2_diff", dateStr)

    ndvi_brdf = cal_NDVI(ref_1_brdf, ref_2_brdf)
    ndvi_diff = ndvi_brdf - ndvi
    RMSE_ndvi = np.sqrt(metrics.mean_squared_error(ndvi_diff, ndvi))
    print(RMSE_ndvi)
    print("RMSE_ndvi")
    write_tiff(ndvi, "NDVI")
    write_tiff(ndvi_brdf, "NDVI_BRDF")
    print("NDVI_0: " + str(np.mean(ndvi_brdf)))
    display(ndvi_brdf, "NDVI_BRDF", dateStr)
    write_tiff(ndvi_diff, "NDVI_diff", dateStr)
    display(ndvi_diff, "NDVI_diff", dateStr)
    display_hist(ndvi_diff, "NDVI_diff", dateStr)

    fvc_brdf = cal_fvc(ndvi_brdf)
    print(np.max(fvc))
    print(np.min(fvc))
    print(np.all(np.isfinite(fvc)))
    fvc_diff = fvc_brdf - fvc
    # 针对fvc_brdf中的空值进行处理
    fvc_diff[np.isnan(fvc_diff)] = 0
    fvc_brdf[np.isnan(fvc_brdf)] = 0
    write_tiff(fvc, "FVC")
    print("fvc_0: " + str(np.mean(fvc_brdf)))
    write_tiff(fvc_brdf, "FVC_BRDF")
    display(fvc_brdf, "FVC_BRDF", dateStr)
    display_hist(fvc_brdf, "FVC_BRDF", dateStr)
    write_tiff(fvc_diff, "FVC_diff")
    display(fvc_diff, "FVC_diff", dateStr)
    display_hist(fvc_diff, "FVC_diff")
    RMSE_fvc = np.sqrt(metrics.mean_squared_error(fvc_brdf, fvc))
    print("RMSE_fvc")
    print(RMSE_fvc)

    ### get LST_0
    # 每个像元：计算FVC-BTs图像中的对应参数，然后计算得到校正后的BTs
    BTs_brdf = np.zeros(lst.shape, dtype=np.float64)
    for i in range(lst.shape[0]):
        for j in range(lst.shape[1]):
            k, const = calParams(ndvi[i, j], lst[i, j], point_NDVI, point_LST, LUT)
            # slope_ndvi_lst = (lst[i, j] - point_LST) / (ndvi[i, j] - point_NDVI) if ndvi[i, j] != point_NDVI else 0
            # if slope_ndvi_lst == 0:
            #     continue
            BTs_brdf[i, j] = k * fvc_brdf[i, j] + const


    # 计算校正后的LST
    lst_0 = BTs2lst(BTs_brdf)
    lst_0[lst_0 < 260] = 260
    write_tiff(lst_0, "LST_0", dateStr)
    print("lst_0: " + str(np.mean(lst_0)))
    display_lst(lst_0, "LST_0", dateStr)
    display_hist(lst_0, "LST_0", dateStr)

    # 差值
    diff = lst_0 - lst
    write_tiff(diff, "LST_diff", dateStr)
    display_lst(diff, "LST_diff", dateStr)
    display_hist(diff, "LST_diff", dateStr)

    # # 计算去云的LST
    # lst_all = np.zeros(lst.shape, dtype=np.float64)
    # for i in range(lst.shape[0]):
    #     for j in range(lst.shape[1]):
    #         if lst_0[i, j] > 285:
    #             lst_all[i, j] = lst_0[i, j]
    #         else:
    #             lst_all[i, j] = lst[i, j]


    # RMSE
    RMSE = np.sqrt(metrics.mean_squared_error(lst, lst_0))
    print(RMSE)

    # 根据原始观测数据划分角度区间，分别算RMSE并绘制折线图
    calRMSE(lst, lst_0, VZA, dateStr)


def testMOD06():
    sds_cloud = open_gdal("data/MOD06_L2.A2019224.0340.061.2019224133404.hdf")

    fraction = sds_cloud[34].ReadAsArray() * 0.01

    # display(fraction, "cloud_fraction", cmap='jet')

    sds_temp = open_gdal("data/MOD11_L2.0812.0340.hdf")


def testAngle(dateStr):
    """
    用于手动看角度数据
    :return:
    """
    ds_SZA, SZA = open_tiff("data" + dateStr + file_MOD09_SZA)
    ds_SAA, SAA = open_tiff("data" + dateStr + file_MOD09_SAA)
    ds_VZA, VZA = open_tiff("data" + dateStr + file_MOD09_VZA)

    SZA = SZA[854:1304, 2496:3106] * 0.01
    SAA = SAA[854:1304, 2496:3106] * 0.01
    VZA = VZA[854:1304, 2496:3106] * 0.01

    display(SZA, "SZA", dateStr)
    display(SAA, "SAA", dateStr)
    display(VZA, "VZA", dateStr)


def test_reproj():
    file_lst = "data0831/MOD11_L2.LST.hdf"
    hdf = gdal.Open(file_lst)
    datasets = hdf.GetSubDatasets()
    for x in datasets:
        print(x)
    # 查看dataset的信息
    info = gdal.Info(datasets[0][0])
    print(info)
    gdal.Warp("MOD11_L2.LST.tif", datasets[9][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)


def testTiff():
    ds_lst, lst = open_tiff("MOD11_L2.LST.tif")
    lst = lst * 0.02
    lst[lst < 260] = 260
    display_lst(lst, "lst_new", "0807")


def tiff_proj():
    ds_lst, lst = open_tiff("GF5_VIMS_N40.2_E116.6_20190318_010262_L10000126263_TES_LST&Emiss.tif")
    print(ds_lst.GetProjection())
    print(ds_lst.GetGeoTransform())


if __name__ == '__main__':
    # main()

    # testAngle("0807")

    # main_new()

    # main_new2("0807")

    tiff_proj()

    # test_reproj()

    # testTiff()

    # getNDVItif()

    # # BRDF
    # open_gdal(BRDF_fileName)
