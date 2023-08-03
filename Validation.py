# -*- encoding: utf-8 -*-
from osgeo import gdal
from osgeo import osr
import cupy as np
#import numpy
import os,sys,time,argparse, atexit,subprocess,threading,csv
import pandas as pd
from threading import Thread #引入库
os.chdir(os.path.dirname(__file__))

def getSRSPair(dataset):
    '''
    获得给定数据的投影参考系和地理参考系
    :param dataset: GDAL地理数据
    :return: 投影参考系和地理参考系
    '''
    prosrs = osr.SpatialReference()
    prosrs.ImportFromWkt(dataset.GetProjection())
    geosrs = prosrs.CloneGeogCS()
    return prosrs, geosrs

def imagexy2geo(dataset, row, col):
    '''
    根据GDAL的六参数模型将影像图上坐标（行列号）转为投影坐标或地理坐标（根据具体数据的坐标系统转换）
    :param dataset: GDAL地理数据
    :param row: 像素的行号
    :param col: 像素的列号
    :return: 行列号(row, col)对应的投影坐标或地理坐标(x, y)
    '''
    trans = dataset.GetGeoTransform()
    px = trans[0] + col * trans[1] + row * trans[2]
    py = trans[3] + col * trans[4] + row * trans[5]
    return px, py

def geo2lonlat(dataset, x, y):
    '''
    将投影坐标转为经纬度坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param x: 投影坐标x
    :param y: 投影坐标y
    :return: 投影坐标(x, y)对应的经纬度坐标(lon, lat)
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(prosrs, geosrs)
    coords = ct.TransformPoint(x, y)
    return coords[:2]

def box_coordinate(dataset_path,box):
    sub_dataset = gdal.Open(dataset_path) 
    width = sub_dataset.RasterXSize  # 获取数据宽度
    height = sub_dataset.RasterYSize  # 获取数据高度

    #x_center y_center width height
    x = box[0]*width
    y = box[1]*height
    box_width = box[2]*width
    box_height = box[3]*width

    coords_UL = imagexy2geo(sub_dataset,x-box_width/2,y-box_height/2)
    coords_DR = imagexy2geo(sub_dataset,x+box_width/2,y+box_height/2)
    coords_center = [float(x) for x in imagexy2geo(sub_dataset,x,y)]
    coor_info = [coords_UL[0],coords_UL[1],
                 coords_DR[0],coords_DR[1]]
    return coor_info,coords_center

def CraterCSV_Extraction(path):
    with open(path, newline='') as f:
        Crater_Info = csv.DictReader(f)
        return [*Crater_Info]
    
def CraterCSV_Saving(CraterCSV_Dict,csv_path):
    df = pd.DataFrame(CraterCSV_Dict)
    df.to_csv(csv_path, index=False, header=True)

def UpperCSV_Extraction(path):
    Dict = {}
    with open(path, newline='') as f:
        _Info = csv.DictReader(f)
        _Info = [*_Info]
    main_key = list(_Info[0].keys())[0]
    for each in _Info:
        itemID = each[main_key]
        each.pop(main_key)
        Dict.update({itemID:each})
    return Dict,main_key

def del_path(folderpath):
    del_list = os.listdir(folderpath)
    for f in del_list:
        file_path = os.path.join(folderpath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
    os.rmdir(folderpath)

def box_convert(record):
    return [record[1]-record[3]/2,
                          record[0]-record[2]/2,
                          record[1]+record[3]/2,
                          record[0]+record[2]/2]

def iou(box1, box2):
    '''
    两个框（二维）的 iou 计算
    
    注意：边框以左上为原点
    
    box:[top, left, bottom, right]
    '''
    in_h = min(box1[2], box2[2]) - max(box1[0], box2[0])
    in_w = min(box1[3], box2[3]) - max(box1[1], box2[1])
    inter = 0 if in_h<0 or in_w<0 else in_h*in_w
    union = (box1[2] - box1[0]) * (box1[3] - box1[1]) + \
            (box2[2] - box2[0]) * (box2[3] - box2[1]) - inter
    iou = inter / union
    return float(iou)

def collection_write(Path,collection):
    if(len(collection)>0):
        for each in collection:
            filename = each[0]
            with open(Path + filename, 'w') as f:
                #print([x for x in each[1]])
                #print(*[0,1,2,3,4,5], file=f, end='\n')
                print(*each[1], file=f, end='\n')
                print(*each[3], file=f, end='\n')
                f.write("IoU: {0}".format(each[2]))

def filterd_write(dataCSV_path,label_path,imgClip_path):
    CraterCSV = CraterCSV_Extraction(dataCSV_path)
    ProcessedCSV = []
    for record in CraterCSV:
        if(record['is_cut'] == '1'):
            filename = record['CraterID']+'.txt'
            #if(filename == '12-000321.txt'):
                #print('!!!')
            path=os.path.join(label_path,filename)
            if(os.path.exists(path)):
                f=open(path,'r')
                YOLO_label = f.readlines()
                IoU = 0
                conf = 0
                default_box = [float(x) for x in record['box info'].split()]
                best_box = []
                for line in YOLO_label:
                    label_record = [float(x) for x in line.split()]
                    if(len(label_record)>5):
                        LBLbox = box_convert(label_record[1:-1])
                        conf = label_record[-1]
                        REFbox = box_convert(default_box)
                        tmp = iou(REFbox,LBLbox)
                        if(tmp>IoU and conf>=args.Conf):
                            IoU = tmp
                            best_box = np.append(label_record[1:-1],conf)
                if(IoU>=args.IoU):
                    #YOLO validation successed
                    best_box = default_box
                    record['is_AUTOvalidated'] = 1
                    conf = 1
                elif(IoU>=0.33 and conf>=args.Conf):
                    #YOLO position adjusted
                    box_info = best_box
                    record['box info'] = " ".join(str(i) for i in box_info)
                    record['is_AUTOvalidated'] = 2
                else:
                    #YOLO validation failed
                    best_box = default_box
                    record['is_AUTOvalidated'] = 3
                    conf = 0
                record.update({'confidence' : conf})
                img_path = imgClip_path + record['CraterID']+'.jpg'
                coor_box,coor_xy = box_coordinate(img_path,best_box)
                sub_dataset = gdal.Open(img_path) 
                coor_lonlat = geo2lonlat(sub_dataset,coor_xy[0],coor_xy[1])
                record['Lon'] =  '{0:.6f}'.format(coor_lonlat[0])
                record['Lat'] = '{0:.6f}'.format(coor_lonlat[1])
                record['coordinary info'] = " ".join(str('{0:.6f}'.format(i)) for i in coor_box)
                #record['Diameter'] = str((np.abs(coor_box[2]-coor_box[0])+np.abs(coor_box[3]-coor_box[1]))/2/1000)
        ProcessedCSV.append(record)
    CraterCSV_Saving(ProcessedCSV,imgClip_path + 'data_verify.csv')
    #print('Label Filted!')

#def getSubInfo(text):
#    #print("子进程测试代码实时输出内容=>" + text)
#    print(text)

class CMDProcess(threading.Thread):
    '''
        执行CMD命令行的 进程
    '''
    def __init__(self, args,callback):
        threading.Thread.__init__(self)
        self.args = args
        self.callback=callback
        
    def run(self):
        self.proc = subprocess.Popen(
            self.args,
            bufsize=0,
            shell = False,
            stdout=subprocess.PIPE,
        )
        
        while self.proc.poll() is None:
            line = self.proc.stdout.readline()
            line = line.decode("utf8") 
            if(self.callback):
                self.callback(line)
        pass

# 定义一个函数，接受一个文件夹路径作为参数
def has_jpg(folder):
  # 遍历文件夹中的所有文件和子文件夹
  for file in os.listdir(folder):
    # 拼接完整的文件路径
    file_path = os.path.join(folder, file)
    # 如果是文件，判断文件扩展名是否是jpg
    if os.path.isfile(file_path) and file_path.endswith(".jpg"):
      # 如果是，返回True
      return True
    '''
    # 如果是子文件夹，递归调用函数
    elif os.path.isdir(file_path):
      # 如果子文件夹中有jpg文件，返回True
      if has_jpg(file_path):
        return True
    '''
  # 如果遍历完没有找到jpg文件，返回False
  return False

def Folder_Validation(args,file):
    global thread_num
    thread_num+=1
    folder_path = args.rootFolder + '{0}/'.format(file)
    if(has_jpg(folder_path)):
        command = ['python', args.YOLOPath,
                    '--weights', args.ptPath, 
                    '--source', folder_path, 
                    '--project', folder_path,
                    #'--img-size',["384"],
                    '--name','',
                    '--exist-ok',
                    '--save-txt',
                    '--nosave',
                    '--save-conf',
                    #'--save-crop',
                    ]
        #command = ['python', 'yolov5/detect.py','--weights','best.pt', '--source', folder_path, '--project', folder_path,'--name','','--exist-ok','--save-txt','--save-crop','--save-conf']
        try:   
            # 父进程等待子进程结束之后再继续运行
            p = subprocess.Popen(command, shell=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            label_path = folder_path + args.labPath
            while p.poll() is None:
                line = p.stdout.readline()
                line = line.decode('utf-8')
                if line:
                    print(line,end='')
            if p.returncode == 0:
                dataCSV_path = folder_path + "data.csv"
                imgClip_path = folder_path
                filterd_write(dataCSV_path,label_path,imgClip_path)
                #print('Subprogram success')
            else:
                #print('Subprogram failed')
                pass
        except subprocess.CalledProcessError as e:
            print ( "Error:\nreturn code: ", e.returncode, "\nOutput: ", e.stderr.decode("utf-8") )
            sys.stdout.flush() 
            del_path(label_path)
            sys.exit()
        #if(args.deleteJPG==1):
            #for item in os.listdir(folder_path):
            #    if item.endswith ( '.jpg' ):
            #        os.remove (os.path.join (folder_path,item))
        del_path(label_path)
    thread_num-=1

thread_num = 0
ImageDict = {}
datacsv_path = ''
mainkey = ''
ImageStatus_path = ''

@atexit.register
def MainCSV():
    global ImageDict,mainkey,ImageStatus_path,datacsv_path
    Final_dict = []
    for each in ImageDict:
        mainkey_dict = {mainkey:each}
        Final_dict.append(dict(mainkey_dict,**ImageDict[each]))
    df = pd.DataFrame(Final_dict)
    df.to_csv(ImageStatus_path, index=False, header=True)
    print('done')
    sys.stdout.flush() 

def main(args):
    dirs = os.listdir(args.rootFolder)# 读取所有的文件
    global ImageDict,mainkey,ImageStatus_path,datacsv_path
    ImageStatus_path = args.rootFolder + "ImageStatus.csv"
    ImageDict,mainkey = UpperCSV_Extraction(ImageStatus_path)
    for idx in range(len(dirs)):
        print('{2} {0} {1}\n'.format(idx+1,len(dirs),'start'),end = '')
        sys.stdout.flush() 
        file = dirs[idx]
        if os.path.splitext(file)[1] == '' and (file in ImageDict):   # 只读取固定后缀的文件
            if(args.doubleThread==0):
                if(args.reValidate == 1 or ImageDict[file]['is_validated']=='0'):
                    Folder_Validation(args,file)
                    ImageDict[file]['is_validated']='1'
            else:     
                while(True):
                    if(thread_num>1):
                        time.sleep(5)
                        continue
                    else:
                        t = Thread(target=Folder_Validation,args=(args,file))
                        t.start()
                        break
        print('{2} {0} {1}\n'.format(idx+1,len(dirs),'end'),end = '')
        sys.stdout.flush() 

def parse_opt():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--rootFolder", type=str, default="Processed/")
    parser.add_argument("--ptPath", type=str, default="best.pt")
    parser.add_argument("--YOLOPath", type=str, default="yolov5/detect.py")
    parser.add_argument("--labPath", type=str, default="labels/")
    #parser.add_argument("--filtedPath", type=str, default="label_filted/")
    parser.add_argument("--IoU",type=float, default=0.667)
    parser.add_argument("--Conf",type=float, default=0.5)
    #parser.add_argument("--deleteJPG",type=int, default=0)
    parser.add_argument("--doubleThread",type=int, default=0)
    parser.add_argument("--reValidate",type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_opt()
    main(args)
    time.sleep(1)