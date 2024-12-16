import pysofa1.sofa as sofa
import numpy as np


import math

radius = {'Sun': 696342,'Earth':6378.140,'Moon':1737.100} #KM

#   an         x(")        x_er(")     y(")        y_er(")    UT1-TAI(s)  UT1_er(s)     dX(")       dX_er(")    dY(")       dY_er(")
#   2023.40    0.106036    0.000100    0.490185    0.000100 -37.0160727   0.0000300    0.000169    0.000100   -0.000209    0.000100

def time2TT(time,delta_ut1utc,delta_ut1tai):
    utc0,utc = sofa.dtf2d('UTC',time[0],time[1],time[2],time[3],time[4],time[5])
    ut1_0,ut1 = sofa.utcut1(utc0,utc,delta_ut1utc)
    tai0,tai = sofa.ut1tai(ut1_0,ut1,delta_ut1tai)
    tt0,tt = sofa.taitt(tai0,tai)
    return tt0,tt

def polar_mat(tt0,tt,Xdrift,Ydrift):
    # Step 1: 计算TIO定位子s'
    s  = sofa.sp00(tt0,tt)   # TT日期

    # Step 2: 查表得到x和y 
    # Step 3: 构建P, X, C矩阵
    P = sofa.rz(s,np.identity(3))
    '''
    P = np.array([[math.cos(s), math.sin(s), 0], 
                [-math.sin(s), math.cos(s), 0], 
                [0, 0, 1]])
    '''
    Xdrift_rad = Xdrift/3600
    Ydrift_rad = Ydrift/3600
    #计算极移阵
    W = sofa.rx(Xdrift_rad,sofa.ry(Ydrift_rad,P))

    #岁差章动项
    M = sofa.pmat06(tt0,tt)  # 岁差矩阵  
    N = sofa.num06a(tt0,tt)   # 章动矩阵
    Polar_Mat = (W @ N) @ M
    return Polar_Mat

def WGS2GCRS(City_XYZ,tt0,tt,xp,yp):
    #Polar_Mat = polar_mat(tt0,tt,xp,yp)
    Polar_Mat = sofa.pom00(xp/3600/180*sofa.DPI,yp/3600/180*sofa.DPI,sofa.sp00(tt0,tt))
    # Step 4: 进行坐标转换  
    E_rotate = sofa.era00(tt0,tt)
    cel2itm=sofa.c2i06a(tt0,tt)
    RotateMat = sofa.c2tcio(cel2itm,E_rotate,Polar_Mat)
    xyzGCRS = np.linalg.inv(RotateMat) @ City_XYZ
    return xyzGCRS

time = [2023,6,4,00,00,00]
tt0,tt = time2TT(time,-0.0455331,-37.0160727)
time1 = [2023,6,4,23,56,4]
TT0,TT = time2TT(time1,-0.0455331,-37.0160727)
[xp,yp]  = [0.106036, 0.490185]

#Vallado et al. 2006, AIAA                     [meters]  
XYZ_itrs =[-1033.4793830,7901.2952754,6380.3565958]  
# dx,dy = 0                                       [meters] 
XYZ_gcrs =[5102.5089592,6123.0114033,6378.1369247]  

orientation = [0,0,0]
shanghai = [121.501458,31.282923,7.5]
#shanghai = [113.6,38.8,100]
print(shanghai[0]/180*np.pi,shanghai[1]/180*np.pi)
City_XYZ = sofa.gd2gc(1,shanghai[0]/180*np.pi,shanghai[1]/180*np.pi,shanghai[2])
O_XYZ = sofa.gd2gc(1,orientation[0],orientation[1],orientation[2])

s  = sofa.sp00(tt0,tt)   # TT日期
Polar_Mat = sofa.pom00(xp,yp,s)
Polar_Mat1 = Polar_Mat @ sofa.pnm06a(tt0,tt)

s  = sofa.sp00(TT0,TT)   # TT日期
Polar_Mat = sofa.pom00(xp,yp,s)
Polar_Mat2 = Polar_Mat @ sofa.pnm06a(TT0,TT)

print(Polar_Mat2-Polar_Mat1)
# Step 4: 进行坐标转换  
xyz1 = Polar_Mat1 @ City_XYZ   
xyz2 = Polar_Mat2 @ City_XYZ  

oxyz1 = Polar_Mat1 @ O_XYZ 
oxyz2 = Polar_Mat2 @ O_XYZ 

print(f'TIRS: {City_XYZ[0]},{City_XYZ[1]},{City_XYZ[2]}') 
print(f'ITRS1: {xyz1[0]},{xyz1[1]},{xyz1[2]}')
print(f'ITRS2: {xyz2[0]},{xyz2[1]},{xyz2[2]}')
'''
E_rotate1 = sofa.era00(tt0,tt)
E_rotate2 = sofa.era00(TT0,TT)
#E_rotate1 = sofa.ecm06(tt0,tt)
#E_rotate2 = sofa.ecm06(TT0,TT)

#s  = sofa.sp00(tt0,tt)   # TT日期
astrom1 = sofa.apio(sofa.sp00(tt0,tt),E_rotate1,shanghai[0],shanghai[1],shanghai[2],xp,yp,0,0)
astrom2 = sofa.apio(sofa.sp00(TT0,TT),E_rotate2,shanghai[0],shanghai[1],shanghai[2],xp,yp,0,0)

print((E_rotate2-E_rotate1)/np.pi*180)
print((astrom2.eral-astrom1.eral)/np.pi*180)

RotateMat1 = sofa.rz(E_rotate1,Polar_Mat1)
RotateMat2 = sofa.rz(E_rotate2,Polar_Mat2)
'''
xyzGCRS1 = WGS2GCRS(City_XYZ,tt0,tt,xp,yp)
xyzGCRS2 = WGS2GCRS(City_XYZ,TT0,TT,xp,yp)

oxyzGCRS1 = WGS2GCRS(O_XYZ,tt0,tt,xp,yp)
oxyzGCRS2 = WGS2GCRS(O_XYZ,TT0,TT,xp,yp)

print(f'GCRS1: {xyzGCRS1[0]},{xyzGCRS1[1]},{xyzGCRS1[2]}')
print(f'GCRS2: {xyzGCRS2[0]},{xyzGCRS2[1]},{xyzGCRS2[2]}')

print(f'OGCRS1: {oxyzGCRS1[0]},{oxyzGCRS1[1]},{oxyzGCRS1[2]}')
print(f'OGCRS2: {oxyzGCRS2[0]},{oxyzGCRS2[1]},{oxyzGCRS2[2]}')



