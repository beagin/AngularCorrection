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


def CI_fill(ori:np.ndarray):
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
    VZA = VZA[index_ymin:index_ymax, index_xmin:index_xmax] * 0.5
    LST[LST < 250] = 0
    write_tiff(LST, "LST_subset")
    write_tiff(VZA, "VZA_subset")

    # 地理坐标范围
    geotrans_LST = ds_LST.GetGeoTransform()
    geotrans_CI = ds_CI.GetGeoTransform()
    geotrans_LAI = ds_LAI.GetGeoTransform()
    # print(geotrans_LST)
    # print(geotrans_CI)
    # print(geotrans_LAI)
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
    CI_ori = CI_ori[min_y:max_y + 1, min_x:max_x + 1] * 0.001
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

    # radiance_ori =

    # 构建特征空间


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


if __name__ == '__main__':
    # hdf_reproj()
    process_all(1)
