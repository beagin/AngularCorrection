"""
Func: to calculate the radiance for every temperature and create lookup table
"""

import os
import matplotlib.pyplot as plt
import numpy as np

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
    return result


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


def convert1channel():
    """
    将一个波段的SRF文件转为可用于批处理的txt文件
    :return:
    """



def cal_BTs(Ts, band):
    """
    Func: To calculate the B(Ts) for a certain temperature
    :param Ts:
    :return:
    """
    sum_up = 0
    sum_down = 0
    fileName = "data/SRF_ASTER/rtcoef_eos_1_aster_srf_ch" + str(band+1) + ".txt"
    # 文件中波长单位：nm
    with open(fileName, 'r') as file:
        lines = file.readlines()
        for i in range(len(lines) - 1):
            if len(lines[i].split()) != 2:
                continue
            else:
                pair = lines[i].split()
                wv = 1e7 / float(pair[0])
                # 用当前行与下一行的波长差为delta lamda，如是最后一行则用当前行与上一行
                if len(lines[i + 1].split()) == 2:
                    delta = (1e7 / float(lines[i + 1].split()[0]) - wv) * 1e-9
                else:
                    delta = (wv - 1e7 / float(lines[i - 1].split()[0])) * 1e-9
                L = cal_Planck(Ts, wv * 1e-9)
                # print(L)
                sum_up += float(pair[1]) * L * delta
                sum_down += float(pair[1]) * delta

    return sum_up / sum_down


def create_LUT(band):
    """
    func: To calculate the Ts-B(Ts) Lookup table and save the LUT in a file.
    :return:
    """
    with open("SRF/LUT" + str(band) + ".txt", "w") as file:
        BTs_all = []
        Ts_all = []
        file.write("Ts\tB(Ts)\n")
        for i in range(100000):
            Ts = 240 + i * 0.001
            BTs = cal_BTs(Ts, band)
            BTs_all.append(BTs)
            Ts_all.append(Ts)
            file.write(str(Ts) + "\t" + str(BTs) + "\n")
    # print(BTs_all)
    # print(Ts_all)

    plt.plot(Ts_all, BTs_all)
    plt.xlabel("Ts")
    plt.ylabel("B(Ts)")
    plt.savefig("SRF/LUT" + str(band) + ".png")
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
    for i in range(10, 15):
        create_LUT(i)
    # plot_L_wv()