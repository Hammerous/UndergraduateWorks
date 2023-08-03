# -*- coding: utf-8 -*-
"""
Created on Tue Dec  1 17:06:01 2020

@author: uuu
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math

def blh2xyz(blh):#大地坐标转空间直角坐标
    a = 6378137.0
    f=1.0/298.257223563
    e=math.sqrt(2*f-f*f);
    e2 = 0.00669437999013

    lat = blh[0]
    lon = blh[1]
    height = blh[2]

    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)

    t2lat = (np.tan(lat))*(np.tan(lat))
    tmp = 1 - e*e
    tmpden = np.sqrt(1 + tmp * t2lat)
    tmp2 = np.sqrt(1 - e*e*slat*slat)
    N=a/tmp2

    x = (N+height)*clat*clon
    y = (N+height)*clat*slon
    z = (a*tmp*slat) / tmp2 + height * slat
    return [x,y,z]

def xyz2enu( xyz,  orgblh):#空间直角坐标转站心坐标系

    lat = orgblh[0]
    lon = orgblh[1]
    height = orgblh[2]

    slat = np.sin(lat)
    clat = np.cos(lat)
    slon = np.sin(lon)
    clon = np.cos(lon)

    tmpxyz=[0,0,0]
    orgxyz=[0,0,0]
    tmporg=[0,0,0]
    difxyz= [0,0,0]
    enu=[0,0,0]

    orgxyz=blh2xyz(orgblh)


    for i in range(3):
        tmpxyz[i] = xyz[i]
        tmporg[i] = orgxyz[i]
        difxyz[i] = tmpxyz[i] - tmporg[i]

    R_list = [[-slon,clon,0] , [-slat * clon,-slat * slon,clat ], [clat*clon,clat*slon,slat ] ]

    for i in range(3):
        enu[0] = enu[0] + R_list[0][i] * difxyz[i]
        enu[1] = enu[1] + R_list[1][i] * difxyz[i]
        enu[2] = enu[2] + R_list[2][i] * difxyz[i]
    return enu
    
def xyz2blh(xyz):  # 空间直角坐标转换为大地坐标
    blh=[0,0,0]
    # 长半轴
    a = 6378137.0
    # 扁率
    f = 1.0/298.257223563
    e2=f*(2-f)
    r2=xyz[0]*xyz[0]+xyz[1]*xyz[1]
    z=xyz[2]
    zk=0.0
    
    while(abs(z-zk)>=0.0001):
        zk=z;
        sinp=z/math.sqrt(r2+z*z);
        v=a/math.sqrt(1.0-e2*sinp*sinp);
        z=xyz[2]+v*e2*sinp;
        
    if(r2>1E-12):
        blh[0]=math.atan(z/math.sqrt(r2))
        blh[1]=math.atan2(xyz[1],xyz[0])
    else:
        if(r2>0):
            blh[0]=math.pi/2.0
        else:
            blh[0]=-math.pi/2.0
        blh[1]=0.0
    
    blh[2]=math.sqrt(r2+z*z)-v
    return blh

# main
if __name__ == "__main__":
    snxPos=[-2831733.583,4675665.958,3275369.410]
    blh0=xyz2blh(snxPos)
    print(blh0)
    xyz2=blh2xyz(blh0)
    print(xyz2)
