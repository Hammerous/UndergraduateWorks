#计算金星凌日发生的时间(1900-2050)
from jplephem.spk import SPK
from astropy.time import Time,TimeDelta
import numpy as np
import warnings,os
import pysofa1.sofa as sofa
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
warnings.filterwarnings('error', module='astropy._erfa')

radius = {'Sun': 696342,'Earth':6378.140,'Moon':1737.100} #KM
#Vallado et al. 2006, AIAA                     [meters]  
XYZ_itrs =[-1033.4793830,7901.2952754,6380.3565958]  
# dx,dy = 0                                       [meters] 
XYZ_gcrs =[5102.5089592,6123.0114033,6378.1369247]  

#  ECEF坐标系转换至ENU坐标系，输入ECEF坐标x,y,z和参考点xyz，输出ENU坐标
# xyz转换至以测站精确坐标为基准的enu坐标
def xyz2enu(xyz,orgxyz):
    phi = orgxyz[0]/180*sofa.DPI
    lam = orgxyz[1]/180*sofa.DPI
    sinphi = np.sin(phi)
    cosphi = np.cos(phi)
    sinlam = np.sin(lam)
    coslam = np.cos(lam)

    difxyz=[xyz[0]-orgxyz[0],xyz[1]-orgxyz[1],xyz[2]-orgxyz[2]]
    e = -sinlam*difxyz[0]+coslam*difxyz[1]
    n = -sinphi*coslam*difxyz[0]-sinphi*sinlam*difxyz[1]+cosphi*difxyz[2]
    u =  cosphi*coslam*difxyz[0]+cosphi*sinlam*difxyz[1]+sinphi*difxyz[2]

    elevation = np.arctan2(u, np.sqrt(e * e + n * n))
    azimuth = np.arctan2(e, n)
    if (azimuth < 0):
        azimuth += sofa.D2PI;
    if (azimuth > sofa.D2PI):
        azimuth -= sofa.D2PI;
    return np.array([azimuth/sofa.DPI*180,90-elevation/sofa.DPI*180])

def DrawFig(path,SolarClip,LunarClip,SolRadius,LunRadius,time):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    frameNUM = len(SolarClip)
    delta_time = TimeDelta(8/24,format='jd')

    time = (time+delta_time).fits

    relaPosi = LunarClip-SolarClip
    max_x = np.max(relaPosi[0, :])
    min_x = np.min(relaPosi[0, :])
    max_y = np.max(relaPosi[1, :])
    min_y = np.min(relaPosi[1, :])
    # 计算图像的宽度和高度
    width = max(abs(max_x),abs(min_x),abs(max_y),abs(min_y))

    # 设置图像的坐标轴范围和纵横比
    ax.set_xlim(SolarClip[0][0]-width, SolarClip[0][0]+width)
    ax.set_ylim(SolarClip[0][1]-width, SolarClip[0][1]+width)

    ax.spines['top'].set_visible = False
    ax.spines['right'].set_visible = False
    ax.spines['bottom'].set_position(('center'))
    ax.spines['bottom'].set_zorder(-1)
    ax.spines['bottom'].set_label('Direction')
    ax.spines['left'].set_position(('center'))
    ax.spines['left'].set_zorder(-1)
    ax.spines['left'].set_label('Elevation')
    ax.set_aspect('equal')
    ax.set_title(time[0])

    Sun = plt.Circle(SolarClip[0], SolRadius[0], color="yellow")
    Moon = plt.Circle(LunarClip[0], LunRadius[0], color="black")

    # input the radius of the yellow circle
    # set the center of the yellow circle to (0, 0)
    # create a yellow circle

    # 清空当前帧
    def init():
        Sun.set_radius(np.nan)
        Moon.set_radius(np.nan)       
        return Sun, Moon,

    # 更新新一帧的数据
    def update(frame):
        Sun.center = SolarClip[frame]
        Sun.set_radius(SolRadius[frame])
        Moon.center = LunarClip[frame]
        Moon.set_radius(LunRadius[frame])

        ax.set_title(time[frame])
        ax.set_xlim(SolarClip[frame][0]-width, SolarClip[frame][0]+width)
        ax.set_ylim(SolarClip[frame][1]-width, SolarClip[frame][1]+width)

        # add the circle to the axis
        ax.add_patch(Sun)
        ax.add_patch(Moon)
        print("\r",'Frame Processing: {0}/{1}'.format(frame,frameNUM),end="",flush=True)
        return Sun, Moon,

    # 调用 FuncAnimation
    ani = FuncAnimation(fig
                    ,update
                    ,init_func=init
                    ,frames=len(SolarClip)
                    ,interval=2
                    ,blit=True
                    )

    ani.save(path + "animation.gif", fps=30, writer="imagemagick")
    print("\r",'GIF saved!',end="\n",flush=True)

def time2TT(time,delta_ut1utc,delta_ut1tai):
    utc0,utc = sofa.dtf2d('UTC',time[0],time[1],time[2],time[3],time[4],time[5])
    ut1_0,ut1 = sofa.utcut1(utc0,utc,delta_ut1utc)
    tai0,tai = sofa.ut1tai(ut1_0,ut1,delta_ut1tai)
    tt0,tt = sofa.taitt(tai0,tai)
    return tt0,tt

def WGS2GCRS(City_XYZ,tt0,tt,xp,yp):
    #Polar_Mat = polar_mat(tt0,tt,xp,yp)
    Polar_Mat = sofa.pom00(xp/3600/180*sofa.DPI,yp/3600/180*sofa.DPI,sofa.sp00(tt0,tt))
    # Step 4: 进行坐标转换  
    E_rotate = sofa.era00(tt0,tt)
    cel2itm=sofa.c2i06a(tt0,tt)
    RotateMat = sofa.c2tcio(cel2itm,E_rotate,Polar_Mat)
    xyzGCRS = np.linalg.inv(RotateMat) @ City_XYZ
    return xyzGCRS,RotateMat

def Calulation(start_time,end_time,City_XYZ,time_interval,bsp_path,out_putPath,City,loose):
    st = Time(start_time, format='isot')
    et = Time(end_time, format='isot')

    st_jd=st.jd
    et_jd=et.jd                #将起止时间转换为儒略历

    bins=(et_jd-st_jd)*time_interval      #每隔一个小时计算是否发生金星凌日

    if int(bins+1)-bins<0.5:
        bins+=1

    times=np.linspace(st_jd,et_jd,int(bins)+1)

    kernel = SPK.open(bsp_path)       #读入JPL星表
    #v_position = kernel[0,2].compute(times)#金星相对太阳系质心的位置
    Solar_position = kernel[0,10].compute(times)
    EM_position = kernel[0,3].compute(times)#地月系相对太阳系质心的位置
    e_position = kernel[3,399].compute(times)#地球相对地月系质心的位置
    m_position = kernel[3,301].compute(times)#月球相对地月系质心的位置

    Shanghai_position = []
    BCRS2ECEFmat = []
    i = 0
    target = len(times)
    for each in times:
        xp , yp = sofa.xy06(each,0)
        InstantPosi,Rmat = WGS2GCRS(City_XYZ,each,0,xp,yp)
        Shanghai_position.append(InstantPosi)
        BCRS2ECEFmat.append(Rmat)
        i+=1
        if(i%10000==0):
            print("\r",'{0}/{1}'.format(i,target),end="",flush=True)
    print("\r",'City_correction DONE',end="\n",flush=True)
    Shanghai_position = np.array(Shanghai_position).T/1000 #KM

    SH_SolarPosi = Solar_position - EM_position - e_position - Shanghai_position
    E_SolarPosi = Solar_position - EM_position - e_position

    SH_mPosi = m_position - e_position - Shanghai_position
    E_mPosi = m_position - e_position

    eclipse_deg=np.zeros([len(times),3])
    LunRePosi = []
    SolRePosi = []
    #SolarsightRadius=np.zeros(len(times))
    #LunarsightRadius=np.zeros(len(times))
    for i in range(len(times)):
        daytime_deg = np.arccos(Shanghai_position[:,i].dot(E_SolarPosi[:,i])/np.linalg.norm(Shanghai_position[:,i])/np.linalg.norm(E_SolarPosi[:,i]))*180/sofa.DPI
        #计算地点与地心连线和地心太阳连线的夹角
        if(abs(daytime_deg)<=90):
            eclipse_deg[i][0] = np.arccos(SH_mPosi[:,i].dot(SH_SolarPosi[:,i])/np.linalg.norm(SH_mPosi[:,i])/np.linalg.norm(SH_SolarPosi[:,i]))*180/sofa.DPI
            eclipse_deg[i][1] = np.arctan(radius['Sun'] / np.linalg.norm(SH_SolarPosi[:,i])) * 180 / sofa.DPI
            eclipse_deg[i][2] = np.arctan(radius['Moon'] / np.linalg.norm(SH_mPosi[:,i])) * 180 / sofa.DPI
            if(loose==False):
                #x,y=np.array(ecef2enu_with_station(BCRS2ECEFmat[i] @ e_mPosi[:,i] * 1000,City_XYZ,city_lat))
                #print(BCRS2ECEFmat[i] @ Shanghai_position[:,i]*1000)
                #print(City_XYZ)
                LunRePosi.append(xyz2enu(BCRS2ECEFmat[i] @ E_mPosi[:,i] * 1000,City_XYZ))
                SolRePosi.append(xyz2enu(BCRS2ECEFmat[i] @ E_SolarPosi[:,i] * 1000,City_XYZ))
        else:
            eclipse_deg[i][0] = 1
            eclipse_deg[i][1] = 0
            eclipse_deg[i][2] = 0
            if(loose==False):
                LunRePosi.append([0,0])
                SolRePosi.append([0,0])
        #计算地月连线和地球太阳连线的夹角
    LunRePosi = np.array(LunRePosi)
    SolRePosi = np.array(SolRePosi)
    #transit_deg=[x for x in eclipse_deg if x<=(0.25+0.25)]
    if(loose):
        scale = 2
        transit_deg = [x[0] for x in eclipse_deg if x[0]<=(x[1]+x[2])*scale]
    else:
        transit_deg = [x[0] for x in eclipse_deg if x[0]<=(x[1]+x[2])]
    ind_dict = dict((k,i) for i,k in enumerate(eclipse_deg[:,0]))
    inter = set(ind_dict).intersection(transit_deg)
    indices = [ ind_dict[x] for x in inter ]
    #找到日食发生时的索引

    indices.sort()#顺序排列
    duration = Time(times[indices], format='jd')  
    if len(duration)!=0:    
        all_transit=times[indices]
        if(loose):
            threashod = 100
        else:
            threashod = 1e-5
        if(len(all_transit)>1):
            transit_bin = [all_transit[i+1]-all_transit[i] for i in range(len(all_transit)-1)]
        else:
            transit_bin = all_transit
        #print(max(transit_bin))
        knot=[i for (i, v) in enumerate(transit_bin) if v > threashod]

        #输出日食发生的时间
        f1 = open(out_putPath+'preview_log.txt','w')
        utc_duration=duration.fits  #将儒略历时间转换为普通时间
        if(loose):
            print('from ',utc_duration[0],'to ',utc_duration[knot[0]])
        f1.write('from '+ str(utc_duration[0]) + 'to '+ str(utc_duration[knot[0]]) +'\n')

        for i in range(len(knot)-1):
            if(loose):
                print('from ',utc_duration[knot[i]+1],'to ',utc_duration[knot[i+1]])
            f1.write('from '+ str(utc_duration[knot[i]+1])+'to '+str(utc_duration[knot[i+1]])+'\n')
        if(len(utc_duration)>1):
            if(loose):
                print('from ',utc_duration[knot[-1]+1],'to ',utc_duration[-1])
            f1.write('from '+ str(utc_duration[knot[-1]+1])+'to '+str(utc_duration[-1])+'\n')
        f1.close()

        if(loose):
            f2 = open(out_putPath+'timelog.txt','w')
            #f2.write('Max Magnitude: {0}\n'.format((0.5-min(transit_deg))/0.5*100))
            f2.write('###\n')
            for each in utc_duration[0:knot[0]+1]:
                f2.write(str(each) +'\n')
            f2.write('###\n')
            for i in range(len(knot)-1):
                for each in utc_duration[knot[i]+1:knot[i+1]+1]:
                    f2.write(str(each) +'\n')
                f2.write('###\n')
            for each in utc_duration[knot[-1]+1:]:
                f2.write(str(each) +'\n')
            f2.close()

        if(loose==False):
            print('Max Magnitude: ',(0.5-min(transit_deg))/0.5*100)
            LunRePosi = np.array(LunRePosi[indices])
            SolRePosi = np.array(SolRePosi[indices])
            relative = LunRePosi - SolRePosi
            print(min(transit_deg))
            print(min([np.linalg.norm(x) for x in relative]))
            SolarRadius = eclipse_deg[:,1][indices]
            LunarRadius = eclipse_deg[:,2][indices]
            print('Drawing GIF...')
            DrawFig(out_putPath,SolRePosi,LunRePosi,SolarRadius,LunarRadius,duration)
        return True
    else:
        return False

def main():
    shanghai = [121.501458,31.282923,7.5]
    City_XYZ = sofa.gd2gc(1,shanghai[0]/180*sofa.DPI,shanghai[1]/180*sofa.DPI,shanghai[2])

    if(simplize):
        start_time = '2009-01-01T00:00:00.0'
        end_time = '2009-12-31T00:00:00.0'
        Calulation(start_time,end_time,City_XYZ,24,'de421.bsp','',shanghai,True)
    else:
        start_time = '2000-01-01T00:00:00.0'
        end_time = '2074-01-01T00:00:00.0'#起止时间
        Calulation(start_time,end_time,City_XYZ,24,'de440.bsp','',shanghai,True)

if __name__ == '__main__':
    os.chdir(os.path.dirname(__file__))
    simplize = False
    main()