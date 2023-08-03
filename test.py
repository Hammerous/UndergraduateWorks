#计算金星凌日发生的时间(1900-2050)
from jplephem.spk import SPK
from astropy.time import Time
import numpy as np

start_time = '1996-01-03T00:00:00.0'
end_time = '2020-06-10T00:00:00.0'#起止时间

st = Time(start_time, format='isot')
et = Time(end_time, format='isot')

st_jd=st.jd
et_jd=et.jd                #将起止时间转换为儒略历

bins=(et_jd-st_jd)*24      #每隔一个小时计算是否发生金星凌日

if int(bins+1)-bins<0.5:
    bins+=1

times=np.linspace(st_jd,et_jd,int(bins)+1)

kernel = SPK.open('de421.bsp')       #读入JPL星表

v_position = kernel[0,2].compute(times)#金星相对太阳系质心的位置
e_position = kernel[0,3].compute(times)#地球相对太阳系质心的位置

ev_deg=np.zeros(len(times))

for i in range(len(times)):
    ev_deg[i]=np.arccos(e_position[:,i].dot(v_position[:,i])/np.linalg.norm(e_position[:,i])/np.linalg.norm(v_position[:,i]))*180/np.pi
    #计算金星太阳连线和地球太阳连线的夹角

transit_deg=[x for x in ev_deg if x<=0.25]#夹角大于太阳角直径的一半即可判断为凌日

ind_dict = dict((k,i) for i,k in enumerate(ev_deg))
inter = set(ind_dict).intersection(transit_deg)
indices = [ ind_dict[x] for x in inter ]
#找到金星凌日发生时的索引

indices.sort()#顺序排列

duration = Time(times[indices], format='jd')  
duration = Time(times[indices], format='jd')  
if len(duration)!=0:
    utc_duration=duration.fits  #将儒略历时间转换为普通时间
    
    #输出金星凌日发生的时间
    all_transit=times[indices]
    transit_bin = [all_transit[i+1]-all_transit[i] for i in range(len(all_transit)-1)]
    knot=[i for (i, v) in enumerate(transit_bin) if v >100]

    print('from ',utc_duration[0],'to ',utc_duration[knot[0]])

    for i in range(len(knot)-1):
        print('from ',utc_duration[knot[i]+1],'to ',utc_duration[knot[i+1]])
    print('from ',utc_duration[knot[-1]+1],'to ',utc_duration[-1])
else:
    utc_duration=[]
