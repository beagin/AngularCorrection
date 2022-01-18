import os
from osgeo import gdal, gdalconst
# from ..ASTER.simu import *

# 文件路径
file_LST = ""


def hdf_reproj():
    """
    将下载的原始.hdf文件都转换为WGS84投影，0.01度分辨率的.tif文件
    :return:
    """
    # 获取目录下的所有hdf文件
    folderPath = "..\\Application\\data\\Hetao_2019\\original\\"
    filelist = os.listdir(folderPath)
    print(filelist)
    for file in filelist:
        file_lst = folderPath + file
        hdf = gdal.Open(file_lst)
        datasets = hdf.GetSubDatasets()
        for x in datasets:
            print(x)
        datalist = []
        for i in range(15, 28):
            datalist.append(datasets[i][0])
        # # 查看dataset的信息
        # info = gdal.Info(datasets[0][0])
        # print(info)
        # 文件名
        name = file.split('.')[1]
        gdal.Warp("../Application/data/Hetao_2019/" + name + "LST.tif", datasets[15][0], dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)
        gdal.Warp("../Application/data/Hetao_2019/" + name + ".tif", datalist, dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)


def process_all():
    """
    应用至真实MODIS数据的整个流程
    :return:
    """
    # 打开所需的数据文件，进行投影转换、subset等


    # 构建特征空间


if __name__ == '__main__':
    hdf_reproj()

