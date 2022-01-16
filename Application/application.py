import os
from osgeo import gdal, gdalconst
# from ..ASTER.simu import *


def hdf_reproj():
    """
    将下载的原始.hdf文件都转换为WGS84投影，0.01度分辨率的.tif文件
    :return:
    """
    # 获取目录下的所有hdf文件
    folderPath = "..\\Application\\data\\Hetao\\origin\\"
    filelist = os.listdir(folderPath)
    print(filelist)
    for file in filelist:
        file_lst = folderPath + file
        hdf = gdal.Open(file_lst)
        datasets = hdf.GetSubDatasets()
        for x in datasets:
            print(x)
        datalist = []
        for i in range(9, 16):
            datalist.append(datasets[i][0])
        # # 查看dataset的信息
        # info = gdal.Info(datasets[0][0])
        # print(info)
        # 文件名
        name = file.split('.')[1]
        gdal.Warp("../Application/data/Hetao/" + name + ".tif", datalist, dstSRS='EPSG:4326', xRes=0.01, yRes=0.01)


if __name__ == '__main__':
    hdf_reproj()

