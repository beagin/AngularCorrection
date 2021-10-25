"""
kernals.py: calculation of kernals
"""

import math


def Ross_thick(sZenith: float, vZenith: float, rAzimuth: float):
    """
    Ross 厚层，Kvol
    :param sZenith:
    :param vZenith:
    :param rAzimuth:
    :return:
    """
    cosxi = math.cos(sZenith) * math.cos(vZenith) + math.sin(sZenith) * math.sin(vZenith) * math.cos(rAzimuth)
    xi = math.acos(cosxi)
    k1 = (math.pi/2 - xi) * cosxi + math.sin(xi)
    k = k1 / (math.cos(sZenith) + math.cos(vZenith)) - math.pi / 4
    return k


def LI_SparseR(sZenith: float, vZenith: float, rAzimuth: float):
    """
    Li sparse kernel, Kgeo
    :param sZenith:
    :param vZenith:
    :param rAzimuth:
    :return:
    """
    brratio = 1.0
    hbratio = 2.0
    t1 = brratio * math.tan(sZenith)
    theta_ip = math.atan(t1)
    t2 = brratio * math.tan(vZenith)
    theta_vp = math.atan(t2)
    temp1 = math.cos(theta_ip)
    temp2 = math.cos(theta_vp)
    cosxip = temp1 * temp2 + math.sin(theta_ip) * math.sin(theta_vp) * math.cos(rAzimuth)
    D1 = t1 * t1 + t2 * t2 - 2 * t1 * t2 * math.cos(rAzimuth)
    # D = math.sqrt(D1)
    cost1 = t1 * t2 * math.sin(rAzimuth)
    cost2 = D1 + cost1 * cost1
    temp3 = 1 / temp1 + 1 / temp2
    cost = hbratio * math.sqrt(cost2) / temp3
    if cost > 1:
        cost = 1
    t = math.acos(cost)
    O = (t - math.sin(t) * cost) * temp3 / math.pi
    k = O - temp3 + (1 + cosxip) / (2 * temp1 * temp2)
    return k
