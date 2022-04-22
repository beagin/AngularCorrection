"""
get TVDI for Huabei plain

"""
from ASTER.simu import *
import os

# 文件路径
# file_NDVI = "temp/MOD13_mask.tif"
# file_LST1 = "temp/MERSI_SW_LST.tif"
# file_LST2 = "temp/190921add1.tif"


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
    k1, c1, k2, c2 = getEdges(LST, ndvi)
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
    print(TVDI.shape)
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


if __name__ == '__main__':
    lstfiles = os.listdir("temp/LST/")
    ndvifiles = os.listdir("temp/NDVI/")
    for index in range(len(lstfiles)):
        doy = ndvifiles[index].split(".")[0][-3:]
        print(doy)
        print(ndvifiles[index])
        print(lstfiles[index])
        main("temp/NDVI/" + ndvifiles[index], "temp/LST/" + lstfiles[index], doy)