from osgeo import gdal


def readLatLon(fileName):
    file = open(fileName, 'r')

    lines = file.readlines()
    for i in range(len(lines)):
        line = lines[i].split()

    file.close()


def readtiff(filePath):
    dataset = gdal.Open(filePath)
    if dataset == None:
        print(filePath + "文件无法打开")
        return
    im_width = dataset.RasterXSize  # 栅格矩阵的列数
    im_height = dataset.RasterYSize  # 栅格矩阵的行数
    im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 获取数据
    print(im_data)


if __name__ == '__main__':
    readtiff("data/ASTER/12275_Lat.tif")