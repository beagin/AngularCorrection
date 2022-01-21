"""
Func: to calculate the radiance for every temperature and create lookup table
"""

from .hdf_gdal import cal_Planck
import os
import matplotlib.pyplot as plt


def combine_channels():
    """
    Func: To combine all the SRF files
    :return: None
    """
    file_all = open("allSRF.txt", 'w')      # file to store all SRF data
    folderPath = "SRFs/"
    # read all the SRF files and transmit the data into file_all
    for fileName in os.listdir(r'SRFs'):
        with open(folderPath + fileName, 'r') as file:
            print(fileName + " started")
            file_all.write("1\n")
            lines = file.readlines()
            for i in range(4, len(lines)-1):
                pair = lines[i].split()
                lamda = 1e7 / float(pair[0])    # unit: nm
                file_all.write(str(lamda) + "\t")
                file_all.write(pair[1] + "\n")
            print(fileName + " finished")
            file_all.write("0\n")
    file_all.write("2\n")


def cal_BTs(Ts):
    """
    Func: To calculate the B(Ts) for a certain temperature
    :param Ts:
    :return:
    """
    sum_up = 0
    sum_down = 0
    # 文件中波长单位：nm
    with open("allSRF.txt", 'r') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1):
            if len(lines[i].split()) == 1:
                continue
            else:
                pair = lines[i].split()
                # 用当前行与下一行的波长差为delta lamda，如是最后一行则用当前行与上一行
                if len(lines[i + 1].split()) == 2:
                    delta = (float(lines[i + 1].split()[0]) - float(pair[0])) * 1e-9
                else:
                    delta = (float(pair[0]) - float(lines[i - 1].split()[0])) * 1e-9
                L = cal_Planck(Ts, float(pair[0]) * 1e-9)
                # print(L)
                sum_up += float(pair[1]) * L * delta
                sum_down += float(pair[1]) * delta

    return sum_up / sum_down


def create_LUT():
    """
    func: To calculate the Ts-B(Ts) Lookup table and save the LUT in a file.
    :return:
    """
    with open("LUT.txt", "w") as file:
        BTs_all = []
        Ts_all = []
        file.write("Ts\tB(Ts)\n")
        for i in range(1000):
            Ts = 240 + i * 0.1
            BTs = cal_BTs(Ts)
            BTs_all.append(BTs)
            Ts_all.append(Ts)
            file.write(str(Ts) + "\t" + str(BTs) + "\n")
    print(BTs_all)
    print(Ts_all)

    plt.plot(Ts_all, BTs_all)
    plt.xlabel("Ts")
    plt.ylabel("B(Ts)")
    plt.savefig("LUT.png")
    plt.show()


def plot_L_wv():
    """
    Func: To draw the black body radiance - wavelength figure
    :return: None
    """
    L_all = []
    wv_all = range(400, 50000)
    for i in wv_all:
        L = cal_Planck(300, i * 1e-9)
        L_all.append(L)

    plt.plot(wv_all, L_all)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("radiance")
    plt.savefig("BlackBody.png")
    plt.show()


if __name__ == "__main__":
    # combine_channels()
    # 6.594058038074137
    # BTs = cal_BTs(310)
    # print(BTs)
    create_LUT()
    # plot_L_wv()