from jplephem.spk import SPK
from astropy.time import Time,TimeDelta
import numpy as np
import warnings
import os,shutil
os.chdir(os.path.dirname(__file__))
import pysofa1.sofa as sofa
import calculate as cal
#import pysofa3.pysofa3.sofa as sofa
warnings.filterwarnings('error', module='astropy._erfa')

shanghai = [121.501458,31.282923,7.5]
City_XYZ = sofa.gd2gc(1,shanghai[0]/180*np.pi,shanghai[1]/180*np.pi,shanghai[2])

time_blocks = []
#with open('Suspected_time.txt', 'r') as f:
with open('timelog.txt', 'r') as f:
    tmp_block = []
    for line in f:
        if(line=="###\n"):
            if(len(tmp_block)!=0):
                time_blocks.append(tmp_block)
                tmp_block=[]
        else:
            tmp_block.append(line.rstrip())
time_blocks.append(tmp_block)
for each in time_blocks:
    print(each)
    timeinterval = 24*60*2
    bsp_path = 'de440.bsp'
    if(len(each)>0):
        delta_time = TimeDelta(1/24,format='jd')

        starttime = Time(each[0])
        endtime = Time(each[-1])
        starttime = starttime-delta_time
        endtime = endtime+delta_time

        Stamp = str(starttime.datetime64)[:10]
        saving_path_root = 'testfolder/'
        saving_path = saving_path_root+Stamp+'/'.format(Stamp)
        try:
            os.mkdir(saving_path_root)
        except FileExistsError:
            print('Root Folder already exists!')
        try:
            os.mkdir(saving_path)
        except FileExistsError:
            print('Saving Folder already exists!')
        feedback = cal.Calulation(str(starttime.fits),str(endtime.fits),City_XYZ,timeinterval,bsp_path,saving_path,shanghai,False)
        print("Eclipse Varification: ",feedback)
        if(feedback==False):          
            try:
                shutil.rmtree(saving_path)
            except FileExistsError:
                print('Saving Folder removing ERROR!')

