from dbfread import DBF
from threading import Thread #引入库
import csv,os,time,numpy,argparse,stat,sys
os.chdir(os.path.dirname(__file__))

thread_num = 0
startposition = 30
KM_threshod = 1000   #KM

def table_tetMatching(all_area,maunal = False):
    area_dict = {}#dictationary with stable .clip
    for tet_area in all_area:
        if(maunal):
            selected_area = [[tet_area[0][0],tet_area[0][1]],[tet_area[1][0],tet_area[1][1]]]
        else:
            selected_area = [[tet_area[0],tet_area[1]],[tet_area[0]+s.clipStep,tet_area[1]+s.clipStep]]
        xy_floored = numpy.floor([float(selected_area[0][0])/s.clipStep,float(selected_area[0][1])/s.clipStep])*s.clipStep
        area_dict.update({'{0}/{1}'.format(int(xy_floored[0]),int(xy_floored[1])):[]})
    return area_dict

def tet_extraction(rootFolderPath,stantard_posi = 30):
    tet_area = []
    dirs = os.listdir(rootFolderPath)# 读取所有的文件
    for file in dirs:
        if os.path.splitext(file)[1] == '':   # 只读取固定后缀的文件
            foldername = file[stantard_posi:].split('_')# 截取文件名字符
            if(len(foldername)>1):
                lon_lat = [float(foldername[0][1:]),float(foldername[1][1:])]
                tet_area.append(lon_lat)
    return tet_area

def write_to_csv(path,data):
    global thread_num
    thread_num+=1    
    with open(path + 'data.csv', 'w',newline='') as f:
        writer = csv.writer(f, delimiter=',')  
        writer.writerow(['CraterID','Lon', 'Lat', 'Diameter','is_cut'])  
        for row in data:
            writer.writerow([row['CraterID'], row['LON_E'], row['LAT'], row['DiamKM'],0])
            #print([row['LON_E'], row['LAT'], row['DiamKM'],row['CraterID']])
    os.chmod(path + 'data.csv', stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
    thread_num-=1    

def Crater_extraction(path,rootfolder,savefolder):
    all_area = tet_extraction(rootfolder)
    #all_area = [[[-24.855,18.029],[-24.053,18.737]]]
    maunal = False
    table = DBF(path,load=False)
    print("Building area dictionary...")
    sys.stdout.flush() 
    area_dict = table_tetMatching(all_area,maunal)
    try:
        os.mkdir(savefolder)
        os.chmod(savefolder, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
    except FileExistsError:
        print('Root Folder already exists!')
        sys.stdout.flush() 

    if(s.clipStep):
        print("Scanning Craters...")
        sys.stdout.flush() 
        for record in table:
            diameter = record['DiamKM']
            if(diameter < KM_threshod):
                lon = float(record['LON_E'])
                lat  = float(record['LAT'])
                if(maunal):
                    if(lon<float(all_area[0][0][0]) or lon>float(all_area[0][1][0]) or 
                       lat<float(all_area[0][0][1]) or lat>float(all_area[0][1][1])):   
                        continue
                LonLat_floored = numpy.floor([lon/s.clipStep,lat/s.clipStep])*s.clipStep
                LonLat_key = '{0}/{1}'.format(int(LonLat_floored[0]),int(LonLat_floored[1]))
                if(LonLat_key in area_dict.keys()):
                    area_dict[LonLat_key].append(record)
        print("Creating Image-corrsponding Folder and Writing Crater info...")
        sys.stdout.flush() 
        with open(s.saveFolder + 'ImageStatus.csv', 'w',newline='') as f:
            writer = csv.writer(f, delimiter=',')  
            writer.writerow(['ImageArea','is_cut','is_validated'])  
            for area_key in area_dict:
                area_info = area_key.split('/')
                folderName = 'E{0:04d}_N{1:03d}'.format(int(area_info[0]),int(area_info[1]))
                currentPath = s.saveFolder + '/{0}/'.format(folderName)
                try:
                    os.mkdir(currentPath)
                    os.chmod(currentPath, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
                except FileExistsError:
                    print('Sub Folder already exists!')
                    sys.stdout.flush() 
                global thread_num
                while(True):
                    if(thread_num>4):
                        time.sleep(0.2)
                        continue
                    else:
                        t = Thread(target=write_to_csv,args=(currentPath,area_dict[area_key]))
                        writer.writerow([folderName, 0, 0])
                        print('{0}N {1}E: {2} craters'.format(area_info[0],area_info[1],len(area_dict[area_key])))
                        sys.stdout.flush() 
                        t.start()
                        break    
            #write_to_csv(currentPath,area_dict[area_key].value())    # 将数据写入CSV 
        t.join() 

def parse_opt():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--dbfPath", type=str, 
                        default=r"RobbinsCraterDatabase_20120821\RobbinsCraterDatabase_20120821\RobbinsCraterDatabase_20120821_LatLonDiam.dbf")
    parser.add_argument("--rootFolder", type=str, 
                        default=r"RealDatainBelt/")
    parser.add_argument("--clipStep", type=float, default=4)
    parser.add_argument("--saveFolder", type=str, 
                        default="Processed/")
    s = parser.parse_args()
    return s

if __name__ == '__main__':
    s = parse_opt()
    Crater_extraction(s.dbfPath,s.rootFolder,s.saveFolder)
    #main(s)
    time.sleep(1)
    print('done')
    sys.stdout.flush() 