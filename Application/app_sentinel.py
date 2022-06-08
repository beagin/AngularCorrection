from ASTER.simu import open_tiff, write_tiff
import numpy as np

file_S8_in = "sentinel/S3B_SL_1_RBT____20210830T020848_20210830T021148_20210831T124634_0179_056_217_2160_LN2_O_NT_004.SEN3/S8_BT_in.tif"
file_S8_io = "sentinel/S3B_SL_1_RBT____20210830T020848_20210830T021148_20210831T124634_0179_056_217_2160_LN2_O_NT_004.SEN3/S8_BT_io.tif"
file_S9_in = "sentinel/S3B_SL_1_RBT____20210830T020848_20210830T021148_20210831T124634_0179_056_217_2160_LN2_O_NT_004.SEN3/S9_BT_in.tif"
file_S9_io = "sentinel/S3B_SL_1_RBT____20210830T020848_20210830T021148_20210831T124634_0179_056_217_2160_LN2_O_NT_004.SEN3/S9_BT_io.tif"
file_geo_in = "sentinel/S3B_SL_1_RBT____20210830T020848_20210830T021148_20210831T124634_0179_056_217_2160_LN2_O_NT_004.SEN3/geodetic_in.tif"
file_geo_io = "sentinel/S3B_SL_1_RBT____20210830T020848_20210830T021148_20210831T124634_0179_056_217_2160_LN2_O_NT_004.SEN3/geodetic_io.tif"


def main():
    # 打开相关文件
    _, B8_in = open_tiff(file_S8_in)
    _, B8_io = open_tiff(file_S8_io)
    _, geo_in = open_tiff(file_geo_in)
    _, geo_io = open_tiff(file_geo_io)
    lon_in = geo_in[2] * 1e-6
    lon_io = geo_io[2] * 1e-6
    B8_io_sub = B8_io[0] * 0.01 + 283.73
    print(B8_io_sub.shape)

    # 坐标匹配，根据io数据的坐标范围裁剪in数据
    # io数据的范围，只有经度因为纬度范围相同
    lon_topleft = lon_io[0, 0]
    lon_bottomright = lon_io[-1, -1]
    # 对应in数据index
    lon_min = np.argmax(lon_in[0] > lon_topleft)        # 第一行
    lon_max = np.argmax(lon_in[-1] > lon_bottomright)   # 最后一行
    # print(lon_max, lon_min)
    # 进行裁剪
    B8_in_sub = B8_in[0][:, lon_min:lon_max+1] * 0.01 + 283.73
    print(B8_in_sub.shape)

    diff = B8_in_sub - B8_io_sub
    diff[diff > 100] = 0
    write_tiff(diff, "B8_diff")



if __name__ == '__main__':
    main()