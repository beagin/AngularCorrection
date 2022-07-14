import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np


def scatters_3D(folder=""):
    """
    3D scatter plot, DFVC as x axis, DCLST as y axis, and LST residual as z axis
    :return:
    """
    # 三个维度的数据，读取为numpy数组
    xfile = open("pics/" + folder + "diff_FVC.txt", 'r')
    yfile = open("pics/" + folder + "diff_CLST.txt", 'r')
    zfile = open("pics/" + folder + "diff_corr_simu_14.txt")
    xdata = np.array([float(x) for x in xfile.readlines()])
    ydata = np.array([float(x) for x in yfile.readlines()])
    zdata = np.array([float(x) for x in zfile.readlines()])
    print(xdata)
    # 获取密度值
    xyz = np.vstack([xdata, ydata, zdata])
    dense = gaussian_kde(xyz)(xyz)
    idx = dense.argsort()
    x, y, z, dense = xdata[idx], ydata[idx], zdata[idx], dense[idx]
    # 绘制3D密度图
    fig = plt.figure(figsize=(12,10), dpi=300)
    ax = plt.axes(projection='3d')
    scatter = ax.scatter3D(x, y, z, c=dense, cmap="Spectral_r", marker='o', s=15)
    divider = make_axes_locatable(ax)
    # cax = divider.append_axes("top", size="5%", pad=0.1)
    # cbar = fig.colorbar(scatter, cax=cax, label='frequency')
    x_major_locator = plt.MultipleLocator(0.05)
    y_major_locator = plt.MultipleLocator(2.5)
    z_major_locator = plt.MultipleLocator(0.5)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.zaxis.set_major_locator(z_major_locator)
    ax.zaxis.set_rotate_label(False)
    ax.set_xlabel(r'$\Delta$' + "FVC", rotation=30, fontdict={'weight':'bold', 'size':13})
    ax.set_ylabel(r'$\Delta$' + "CLST (K)", fontdict={'weight':'bold', 'size':13})
    ax.set_zlabel("LST residual (K)", rotation=90, fontdict={'weight':'bold', 'size':13})
    # ax.view_init(elev=33, azim=-154)
    # plt.savefig("pics/scatter_3D_33-154.jpg", dpi=400)
    # ax.view_init(elev=31, azim=-176)
    # plt.savefig("pics/scatter_3D_31-176.jpg", dpi=400)
    ax.view_init(elev=10, azim=-128)
    plt.savefig("pics/scatter_3D_10-128.jpg", dpi=400)
    # plt.show()


if __name__ == '__main__':
    scatters_3D("v3.12_2327/55/")