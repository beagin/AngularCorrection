"""
get TVDI for Huabei plain

"""
from ASTER.simu import *
import os
from sklearn import linear_model

# 文件路径
# file_NDVI = "temp/MOD13_mask.tif"
# file_LST1 = "temp/MERSI_SW_LST.tif"
# file_LST2 = "temp/190921add1.tif"


def getEdges_manu(lst: np.ndarray, ndvi: np.ndarray):
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
                if x > mean_all + 2 * dev_all or x < mean_all - 3 * dev_all:
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

    Tmax_aver = [330.3596, 330.90103, 329.29767, 3.4051, 331.74847, 331.5992, 330.05356, 329.78018, 328.21152, 326.40527, 325.42807, 324.49097, 323.5041, 322.04364, 321.9713, 320.1643, 320.48407, 319.3588, 318.73853, 317.73853]

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


def calkc():
    data = [306, 292]
    reg = linear_model.LinearRegression()
    reg.fit(np.array(range(2)).reshape(-1, 1), np.array(data).reshape(-1, 1))
    return reg.coef_[0][0], reg.intercept_[0]


def main(file_NDVI, file_LST, doy):
    # 读取文件
    _, ndvi = open_tiff(file_NDVI)
    _, LST = open_tiff(file_LST)
    print(ndvi.shape)
    print(LST.shape)    # 差1，进行裁剪

    # 预处理（统一文件大小）
    # 921
    # ndvi = ndvi[:-1, :-1] * 0.0001
    # 925
    # ndvi = ndvi[:, :-1] * 0.0001
    if ndvi.shape != LST.shape:
        # shape[0]不同
        if ndvi.shape[0] != LST.shape[0]:
            # ndvi范围更小
            if ndvi.shape[0] < LST.shape[0]:
                LST = LST[:ndvi.shape[0]-LST.shape[0], :]
            else:
                ndvi = ndvi[:LST.shape[0]-ndvi.shape[0], :]
        if ndvi.shape[1] != LST.shape[1]:
            # ndvi范围更小
            if ndvi.shape[1] < LST.shape[1]:
                LST = LST[:, :ndvi.shape[1] - LST.shape[1]]
            else:
                ndvi = ndvi[:, :LST.shape[1] - ndvi.shape[1]]

    ndvi = ndvi * 0.0001
    ndvi[np.isnan(ndvi)] = 0
    LST[np.isnan(LST)] = 0
    LST[LST < 260] = 0

    # 建立空间，获取顶点
    # k1, c1, k2, c2 = getEdges_manu(LST, ndvi)
    k1, c1 = calkc()
    print(k1, c1)
    k2 = 0
    c2 = 270
    scatter_BTs_fvc(LST, ndvi, k1, c1, k2, c2, band=doy)

    # 对每个点，计算TVDI值
    shape = ndvi.shape
    TVDI = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # 判断是否为有效点
            if LST[i, j] >= 260:
                # 当前点横坐标与干边的交点
                ymax = ndvi[i, j] * k1 + c1
                value = (LST[i, j]-c2)/(ymax-c2)
                if value >= 1:
                    TVDI[i, j] = 1
                elif value <= 0:
                    TVDI[i, j] = 0
                else:
                    TVDI[i, j] = value
            else:
                continue
    TVDI[LST < 260] = np.nan
    write_tiff(TVDI, "TVDI_" + str(doy))
    export("pics/TVDI_" + str(doy) + ".tif", file_LST)

    # 导出有效点txt
    # ndvi = ndvi[LST>=260]
    # LST = LST[LST>=260]
    # txt_LST = open("pics/LST.txt", 'w')
    # txt_NDVI = open("pics/NDVI.txt", 'w')
    # print(len(LST))
    # for i in range(len(LST)):
    #     txt_LST.write(str(LST[i]) + "\n")
    #     txt_NDVI.write(str(ndvi[i]) + "\n")
    # txt_NDVI.close()
    # txt_LST.close()


def export(source, target):
    _, data = open_tiff(source)
    # 用于参考的数据
    ds_LAI, LAI = open_tiff(target)
    proj = ds_LAI.GetProjection()
    # 写新的文件
    driver = gdal.GetDriverByName("GTiff")
    ds_new = driver.Create(source, data.shape[1], data.shape[0], 1, gdal.GDT_Float32)
    # 添加坐标信息
    geoTrans = ds_LAI.GetGeoTransform()
    # 赋值
    ds_new.SetProjection(proj)
    ds_new.SetGeoTransform(geoTrans)
    ds_new.GetRasterBand(1).WriteArray(data)
    del ds_new


def processAll():
    lstfiles = os.listdir("temp/LST/")
    ndvifiles = os.listdir("temp/NDVI/")
    for index in range(len(lstfiles)):
        doy = ndvifiles[index].split(".")[0][-3:]
        print(doy)
        print(ndvifiles[index])
        print(lstfiles[index])
        main("temp/NDVI/" + ndvifiles[index], "temp/LST/" + lstfiles[index], doy)


def processIndex(index):
    lstfiles = os.listdir("temp/LST/")
    ndvifiles = os.listdir("temp/NDVI/")
    doy = ndvifiles[index].split(".")[0][-3:]
    main("temp/NDVI/" + ndvifiles[index], "temp/LST/" + lstfiles[index], doy)


if __name__ == '__main__':
    # main("temp/NDVI/P_2021097.tif", "temp/LST/20210418_0555_day_LST.tif", 097)
    processIndex(7)
    # processAll()