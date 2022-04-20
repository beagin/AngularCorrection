import os
import sys
sys.path.append("..")
# from osgeo import gdal, gdalconst
from ASTER.simu import *

# 文件路径
# 河套平原
# string_folder = "data/Hetao_2019/A2019187/"
# 张掖
string_folder = "data/Zhangye_2019/A2019222/"

File_LST = string_folder + "LST.tif"
File_VZA = string_folder + "VZA.tif"
File_Emis29 = string_folder + "Emis29.tif"
File_Emis31 = string_folder + "Emis31.tif"
File_Emis32 = string_folder + "Emis32.tif"
File_LAI = string_folder + "Lai_500m.tif"
File_CI = string_folder + "CI.tif"

# constants
h = 6.62606896 * 1e-34      # Planck constant
c = 299792458               # speed of light
k = 1.3806 * 1e-23          # Boltzmann constant


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
    # print(result)
    return result


def create_LUT():
    """
    建立8-13.5微米的LST-Radiance查找表
    :return:
    """
    LUT_file = open("data/LUT.txt", 'w')
    LUT_file.write("Ts\tB(Ts)\n")
    for LST in range(270000, 340000):
        sum = 0
        for i in range(800, 1350):
            wv = i * 1e-8
            sum += cal_Planck(LST / 1000, wv)
        sum = sum / 550
        LUT_file.write(str(LST/1000) + "," + str(sum) + "\n")
    LUT_file.close()


def hdf_reproj():
    """
    将下载的原始.hdf文件都转换为WGS84投影，0.01度分辨率的.tif文件
    :return:
    """
    # 获取目录下的所有hdf文件
    folderPath = "..\\Application\\data\\Zhangye_2019\\MOD21_2\\"
    filelist = os.listdir(folderPath)
    print(filelist)
    for file in filelist:
        file_lst = folderPath + file
        hdf = gdal.Open(file_lst)
        datasets = hdf.GetSubDatasets()
        # for x in datasets:
        #     print(x)
        datalist = []
        for i in range(15, 28):
            datalist.append(datasets[i][0])
        # # 查看dataset的信息
        # info = gdal.Info(datasets[0][0])
        # print(info)
        # 文件名
        name = file.split('.')[1]
        # 输出温度、发射率、VZA
        # gdal.Warp("data/Zhangye_2019/" + name + "LST.tif", datasets[15][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)
        gdal.Warp("data/Zhangye_2019/" + name + "/LST.tif", datasets[15][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)
        gdal.Warp("data/Zhangye_2019/" + name + "/Emis29.tif", datasets[17][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)
        gdal.Warp("data/Zhangye_2019/" + name + "/Emis31.tif", datasets[18][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)
        gdal.Warp("data/Zhangye_2019/" + name + "/Emis32.tif", datasets[19][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)
        gdal.Warp("data/Zhangye_2019/" + name + "/VZA.tif", datasets[27][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)
        # gdal.Warp("../Application/data/Hetao_2019/" + name + ".tif", datalist, dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)


def emis_fit(emis_29, emis_31, emis_32):
    """
    根据MODIS三个波段发射率与宽波段发射率的关系
    :param emis_29:
    :param emis_31:
    :param emis_32:
    :return:
    """
    # （Tang, 2011）
    # return 0.0127 + 0.7852 * emis_29 - 0.0151 * emis_31 + 0.2139 * emis_32
    # （Zeng，2021）
    return 0.095 + 0.329 * emis_29 + 0.572 * emis_31


def cal_Radiance(emis, LST):
    """
    计算辐亮度：热红外波段，LST对应黑体辐射乘以发射率
    :param emis:
    :param LST:
    :return:
    """
    shape = LST.shape
    rad = np.zeros(shape, dtype=np.float64)
    for x in range(shape[0]):
        for y in range(shape[1]):
            # 温度异常的像元 辐亮度直接设为零
            if LST[x, y] < 250:
                rad[x, y] = 0
                continue
            # 8-13.5 um
            sum = 0
            for i in range(800, 1350):
            # for i in range(800, 1350):
                wv = (float)(i * 1e-8)
                sum += cal_Planck(LST[x, y], wv)
            sum = sum / 550
            rad[x, y] = sum * emis[x, y]
    return rad


def up_resolution(data_ori, title, merge_pixel=2):
    """
    对数据进行上采样操作
    :param data_ori: 原始数据
    :param merge_pixel: 默认新分辨率为两倍
    :return:
    """
    shape = data_ori.shape
    shape_new = (int(shape[0] / 2), int(shape[1] / 2))
    data_new = np.zeros(shape_new)
    for i in range(shape_new[0]):
        for j in range(shape_new[1]):
            data_new[i, j] = np.mean(data_ori[i*2:i*2+2, j*2:j*2+2])
    write_tiff(data_new, title)


def CI_fill(ori: np.ndarray):
    """
    给定CI数据，对其缺失值进行填充
    :param ori:
    :return:
    """
    shape = ori.shape
    new = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            # 边缘点或者有有效值，都直接赋值
            if i == 0 or i == shape[0]-1 or j == 0 or j == shape[1]-1 or ori[i, j] != 0:
                new[i, j] = ori[i, j]
            # 否则，是中间点且为0，判断其邻域内的有效点的个数，超过5个则进行平均
            else:
                neighbor = ori[i-1:i+2, j-1:j+2]
                valid = neighbor[neighbor != 0]
                if len(valid) >= 5:
                    new[i, j] = np.mean(valid)
    return new


def get_validity():
    """
    计算出中间结果后，获取整个影像所有像元的有效性
    :return:
    """
    # 打开相关文件
    _, LST = open_tiff("pics/LST_subset.tif")
    _, fvc = open_tiff("pics/FVC.tif")
    _, fvc_0 = open_tiff("pics/FVC_0.tif")

    shape = LST.shape
    valid = np.ones(shape, dtype=np.bool)
    for i in range(shape[0]):
        for j in range(shape[1]):
            if LST[i, j] < 250 or LST[i, j] > 350:
                valid[i, j] = 0
                continue
            if fvc[i, j] < 1e-2 or fvc[i, j] > 0.99:
                valid[i, j] = 0
                continue
            if fvc_0[i, j] < 1e-2 or fvc_0[i, j] > 0.99:
                valid[i, j] = 0
                continue
    write_tiff(valid, "is_valid")


def getLUT(fileName):
    """
    Func: to get the lookup table and save in a list
    :param fileName: file name of the LUT text file
    :return: the list of LUT
    """
    with open(fileName, "r") as file:
        lines = file.readlines()
        LUTlist = [float(lines[i].split(',')[1]) for i in range(1, len(lines))]
        # print(LUTlist)
    return LUTlist


def BT2LST(BT: np.ndarray):
    """
    将辐亮度数组转换为地表温度
    :param BT:
    :return:
    """
    LUT = getLUT("data/LUT.txt")   # 获取查找表
    _, SE = open_tiff("pics/emis_all.tif")
    BTs = BT / SE
    shape_ori = BT.shape
    BTs = BTs.reshape(-1)
    shape = BTs.shape
    LST = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        if BTs[i] <= 0:
            LST[i] = 0
            continue
        index = np.searchsorted(np.array(LUT), BTs[i])
        LST[i] = 270 + 0.001 * index
    LST = LST.reshape(shape_ori)
    return LST


def export():
    """
    将主要结果导出至txt
    :return:
    """
    # 打开相关文件
    _, LST = open_tiff("pics/LST_subset.tif")
    _, fvc = open_tiff("pics/FVC.tif")
    _, valid = open_tiff("pics/is_valid.tif")
    _, LST_0 = open_tiff("pics/LST_0.tif")
    _, VZA = open_tiff("pics/VZA_subset.tif")

    # 写txt
    file_LST0 = open("pics/LST_0.txt", 'w')
    file_LST = open("pics/LST.txt", 'w')
    file_fvc = open("pics/FVC.txt", 'w')
    file_VZA = open("pics/VZA.txt", 'w')

    shape = LST.shape
    for i in range(shape[0]):
        for j in range(shape[1]):
            # 是有效像元
            if valid[i, j]:
                file_fvc.write(str(fvc[i, j]) + "\n")
                file_LST0.write(str(LST_0[i, j]) + "\n")
                file_LST.write(str(LST[i, j]) + "\n")
                file_VZA.write(str(VZA[i, j]) + "\n")

    file_LST0.close()
    file_LST.close()
    file_fvc.close()
    file_VZA.close()


def process_space():
    # 从文件获取相关数据
    _, LST = open_tiff("pics/LST_subset.tif")
    _, fvc = open_tiff("pics/FVC.tif")
    _, fvc_0 = open_tiff("pics/FVC_0.tif")
    _, rad_ori = open_tiff("pics/Radiance_ori.tif")
    _, valid = open_tiff("pics/is_valid.tif")

    # 获取有效数据
    LST_valid = LST[valid > 0]
    fvc_valid = fvc[valid > 0]
    fvc_0_valid = fvc_0[valid > 0]
    rad_valid = rad_ori[valid > 0]

    # 构建特征空间
    k1, c1, k2, c2 = getEdges_fvc(rad_valid, fvc_valid)
    print(k1, c1, k2, c2)
    scatter_BTs_fvc(rad_valid, fvc_valid, k1, c1, k2, c2, 0, True, 60)
    point_fvc, point_rad = cal_vertex(k1, c1, k2, c2)
    print(point_fvc, point_rad)

    # 计算角度纠正结果
    shape = LST.shape
    radiance_0 = np.zeros(shape, dtype=np.float64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            k, c = cal_params(point_rad, point_fvc, rad_ori[i, j], fvc[i, j])
            radiance_0[i, j] = k * fvc_0[i, j] + c
    write_tiff(radiance_0, "Radiance_0")

    # 辐亮度到地表温度的转换
    LST_0 = BT2LST(radiance_0)
    write_tiff(LST_0, "LST_0")

    # 有效性
    rad_0_valid = radiance_0[valid > 0]
    LST_0_valid = LST_0[valid > 0]

    # 结果定量分析
    RMSE_Rad = np.sqrt(metrics.mean_squared_error(rad_0_valid, rad_valid))
    print(RMSE_Rad)
    display_hist(rad_valid - rad_0_valid, "Radiance_diff")
    RMSE_LST = np.sqrt(metrics.mean_squared_error(LST_0_valid, LST_valid))
    print(RMSE_LST)
    display_hist(LST_0_valid - LST_valid, "LST_diff")


def exportGeo(index):
    """
    给两个LST文件添加地理信息
    :param index: 0为河套，1为张掖
    :return:
    """
    _, LST = open_tiff("pics/LST_subset.tif")
    _, LST0 = open_tiff("pics/LST_0.tif")
    # 用于参考的LAI数据
    ds_LAI, LAI = open_tiff(File_LAI)
    proj = ds_LAI.GetProjection()
    # 写新的文件
    driver = gdal.GetDriverByName("GTiff")
    ds_LST = driver.Create("pics/LST_geo.tif", LST.shape[1], LST.shape[0], 1, gdal.GDT_Float32)
    ds_LST0 = driver.Create("pics/LST_0_geo.tif", LST0.shape[1], LST0.shape[0], 1, gdal.GDT_Float32)
    # 添加坐标信息
    # Zhangye 2019 222
    if index == 1:
        geoTrans = (99.23072662353515, 0.01, 0.0, 39.31099319458008, 0.0, -0.01)
    # Hetao 2019 187
    elif index == 0:
        geoTrans = (105.74903767581467, 0.01, 0.0, 41.79, 0.0, -0.01)
    # 赋值
    ds_LST.SetProjection(proj)
    ds_LST.SetGeoTransform(geoTrans)
    ds_LST.GetRasterBand(1).WriteArray(LST)
    del ds_LST
    ds_LST0.SetProjection(proj)
    ds_LST0.SetGeoTransform(geoTrans)
    ds_LST0.GetRasterBand(1).WriteArray(LST0)
    del ds_LST0


def process_all(region=0):
    """
    应用至真实MODIS数据的整个流程
    :return:
    :param region: 地区代号，河套平原为0，张掖为1
    """
    # 打开所需的数据文件
    ds_LST, LST = open_tiff(File_LST)
    _, VZA = open_tiff(File_VZA)
    _, emis29 = open_tiff(File_Emis29)
    _, emis31 = open_tiff(File_Emis31)
    _, emis32 = open_tiff(File_Emis32)
    ds_LAI, LAI = open_tiff(File_LAI)
    ds_CI, CI_ori = open_tiff(File_CI)

    # <editor-fold> 裁剪，scale，异常值筛选
    # MOD21数据：LST，VZA，emis
    # Hetao 187
    if region == 0:
        index_ymin = 1180
        index_ymax = 1720
        index_xmin = 640
        index_xmax = 1310
    # Zhangye 222
    else:
        index_ymin = 1720
        index_ymax = 1840
        index_xmin = 730
        index_xmax = 940

    LST = LST[index_ymin:index_ymax, index_xmin:index_xmax] * 0.02
    LST[LST < 250] = 0
    print(LST.shape)
    VZA = VZA[index_ymin:index_ymax, index_xmin:index_xmax] * 0.5
    emis29 = emis29[index_ymin:index_ymax, index_xmin:index_xmax] * 0.002 + 0.49
    emis31 = emis31[index_ymin:index_ymax, index_xmin:index_xmax] * 0.002 + 0.49
    emis32 = emis32[index_ymin:index_ymax, index_xmin:index_xmax] * 0.002 + 0.49
    emis_all = emis_fit(emis29, emis31, emis32)
    write_tiff(LST, "LST_subset")
    write_tiff(VZA, "VZA_subset")
    write_tiff(emis29, "emis_29")
    write_tiff(emis31, "emis_31")
    write_tiff(emis32, "emis_32")
    write_tiff(emis_all, "emis_all")

    # 地理坐标范围
    geotrans_LST = ds_LST.GetGeoTransform()
    geotrans_CI = ds_CI.GetGeoTransform()
    geotrans_LAI = ds_LAI.GetGeoTransform()
    print(geotrans_LST)
    # print(geotrans_CI)
    print(geotrans_LAI)
    minLat = geotrans_LST[3] + geotrans_LST[5] * index_ymax
    maxLat = geotrans_LST[3] + geotrans_LST[5] * index_ymin
    minLon = geotrans_LST[0] + geotrans_LST[1] * index_xmin
    maxLon = geotrans_LST[0] + geotrans_LST[1] * index_xmax
    # print(minLat, maxLat, minLon, maxLon)

    # CI
    base_lon = geotrans_CI[0]
    base_lat = geotrans_CI[3]
    inter_lon = geotrans_CI[1]
    inter_lat = geotrans_CI[5]
    min_x = cal_index(base_lon, inter_lon, minLon)
    max_x = cal_index(base_lon, inter_lon, maxLon)
    min_y = cal_index(base_lat, inter_lat, maxLat)  # 这里由于索引大对应纬度小，进行调换
    max_y = cal_index(base_lat, inter_lat, minLat)
    # Hetao 187
    # CI_ori = CI_ori[min_y:max_y + 1, min_x:max_x + 1] * 0.001
    # Zhangye 222
    CI_ori = CI_ori[min_y:max_y, min_x:max_x] * 0.001

    print(CI_ori.shape)
    CI_new = CI_fill(CI_ori)
    # display(CI_ori, "CI_subset")
    write_tiff(CI_ori, "CI_subset")
    write_tiff(CI_new, "CI_new")

    # LAI
    base_lon = geotrans_LAI[0]
    base_lat = geotrans_LAI[3]
    inter_lon = geotrans_LAI[1]
    inter_lat = geotrans_LAI[5]
    min_x = cal_index(base_lon, inter_lon, minLon)
    max_x = cal_index(base_lon, inter_lon, maxLon)
    min_y = cal_index(base_lat, inter_lat, maxLat)  # 这里由于索引大对应纬度小，进行调换
    max_y = cal_index(base_lat, inter_lat, minLat)
    LAI = LAI[min_y:max_y, min_x:max_x] * 0.1
    print(min_x, max_x, min_y, max_y)
    print(geotrans_LAI)
    print(geotrans_LAI[0])
    print(LAI.shape)
    # LAI上采样
    # up_resolution(LAI_ori, "LAI")
    # _, LAI = open_tiff("pics/LAI.tif")
    write_tiff(LAI, "LAI")

    # </editor-fold>

    # 计算FVC
    fvc = cal_fvc_gap(LAI, CI_ori, VZA)
    fvc_0 = cal_fvc_gap(LAI, CI_ori, 0)
    write_tiff(fvc, "FVC")
    write_tiff(fvc_0, "FVC_0")
    # 计算Radiance
    radiance_ori = cal_Radiance(emis_all, LST)
    write_tiff(radiance_ori, "Radiance_ori")

    # 获取像元有效性
    get_validity()

    # 构建特征空间，进行角度纠正
    process_space()

    # 输出结果以及带地理信息的tif文件
    export()
    exportGeo(region)


def test_landsat():
    ds, landsat = open_tiff("data/LC08_L1TP_123032_20211228_20220105_02_T1_B9.TIF")
    print(landsat.shape)
    # display(landsat, "L9")
    write_tiff(landsat, "L9")
    trans = ds.GetGeoTransform()
    proj = ds.GetProjection()
    print(trans)
    print(proj)
    gdal.Warp("pics/L9_test.tif", ds)


def test():
    # 打开所需的数据文件
    ds_LST, LST = open_tiff(File_LST)
    _, VZA = open_tiff(File_VZA)
    _, emis29 = open_tiff(File_Emis29)
    _, emis31 = open_tiff(File_Emis31)
    _, emis32 = open_tiff(File_Emis32)
    ds_LAI, LAI = open_tiff(File_LAI)
    ds_CI, CI_ori = open_tiff(File_CI)

    # <editor-fold> 裁剪，scale，异常值筛选
    # MOD21数据：LST，VZA，emis
    index_ymin = 1720
    index_ymax = 1840
    index_xmin = 730
    index_xmax = 940

    LST = LST[index_ymin:index_ymax, index_xmin:index_xmax] * 0.02
    LST[LST < 250] = 0
    print(LST.shape)
    VZA = VZA[index_ymin:index_ymax, index_xmin:index_xmax] * 0.5
    emis29 = emis29[index_ymin:index_ymax, index_xmin:index_xmax] * 0.002 + 0.49
    emis31 = emis31[index_ymin:index_ymax, index_xmin:index_xmax] * 0.002 + 0.49
    emis32 = emis32[index_ymin:index_ymax, index_xmin:index_xmax] * 0.002 + 0.49
    emis_all = emis_fit(emis29, emis31, emis32)
    write_tiff(LST, "LST_subset")
    write_tiff(VZA, "VZA_subset")
    write_tiff(emis29, "emis_29")
    write_tiff(emis31, "emis_31")
    write_tiff(emis32, "emis_32")
    write_tiff(emis_all, "emis_all")

    # 地理坐标范围
    geotrans_LST = ds_LST.GetGeoTransform()
    geotrans_CI = ds_CI.GetGeoTransform()
    geotrans_LAI = ds_LAI.GetGeoTransform()
    print(geotrans_LST)
    # print(geotrans_CI)
    print(geotrans_LAI)


if __name__ == '__main__':
    # hdf_reproj()
    # process_all(1)
    # create_LUT()
    # process_space()
    # export()
    exportGeo(1)
    # test()