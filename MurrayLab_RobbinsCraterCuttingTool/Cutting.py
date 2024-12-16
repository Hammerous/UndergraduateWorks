from osgeo import gdal
from osgeo import osr
from threading import Thread #引入库
import pandas as pd
import csv,os,time,sys,tempfile
import atexit
os.chdir(os.path.dirname(__file__))
try:
    import cupy as np
except ModuleNotFoundError:
    import numpy as np
import numpy
import argparse

os.environ['PROJ_LIB'] = os.path.dirname(os.path.dirname(gdal.__file__)+'\\data\\proj\\')
os.environ['GDAL_DATA'] = os.path.dirname(os.path.dirname(gdal.__file__))

thread_num = 0
startposition = 30
KM_threshod = 200   #KM
threshod = 10000 #pixel    
diameter_threshod = 2000 #m

'''
Projection: Equidistant Cylindrical
Center Longitude: 0°
Latitude Type: Planetocentric
'''

datum_parameter = {
                'Equatorial Radius': 3396190.0, #m
                '1/e': 169.894447223612
                }

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
 
def lonlat2geo(dataset, lon, lat):
    '''
    将经纬度坐标转为投影坐标（具体的投影坐标系由给定数据确定）
    :param dataset: GDAL地理数据
    :param lon: 地理坐标lon经度
    :param lat: 地理坐标lat纬度
    :return: 经纬度坐标(lon, lat)对应的投影坐标
    '''
    prosrs, geosrs = getSRSPair(dataset)
    ct = osr.CoordinateTransformation(geosrs, prosrs)
    coords = ct.TransformPoint(lon, lat)
    return coords[:2]
 
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

def geo2imagexy(dataset, x, y):
    '''
    根据GDAL的六 参数模型将给定的投影或地理坐标转为影像图上坐标（行列号）
    :param dataset: GDAL地理数据
    :param x: 投影或地理坐标x
    :param y: 投影或地理坐标y
    :return: 影坐标或地理坐标(x, y)对应的影像图上行列号(row, col)
    '''
    trans = dataset.GetGeoTransform()
    a = np.array([[trans[1], trans[2]], [trans[4], trans[5]]])
    b = np.array([x - trans[0], y - trans[3]])
    return numpy.linalg.solve(a, b)  # 使用numpy的linalg.solve进行二元一次方程的求解

def box_coordinate(box,dataset_path):
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
    coor_info = label4processed_crater(coords_UL,coords_DR)
    return " ".join(str(i) for i in coor_info)

def intersectType(offset_x,offset_y,width,height,block_xsize,block_ysize):
    intersection_status = []
    if(offset_x < 0):
        #矩形框左侧超限
        intersection_status.append('1')
    if(offset_y < 0):
        #矩形框上侧超限
        intersection_status.append('2')
    if(offset_x + block_xsize > width):
        #矩形框右侧超限
        intersection_status.append('3')
    if(offset_y + block_ysize > height):
        #矩形框下侧超限
        intersection_status.append('4')
    return intersection_status

def ImageCut(in_ds,saving_path_root,ID,coords_UL,coords_DR,central_length):
    #global projection_threshod
    is_LargeIMG = 0
    # 将一幅遥感影像拆建为多张
    # 读取要切的原图
    width = in_ds.RasterXSize  # 获取数据宽度
    height = in_ds.RasterYSize  # 获取数据高度
    outbandsize = in_ds.RasterCount  # 获取数据波段数
    #im_geotrans = in_ds.GetGeoTransform()  # 获取仿射矩阵信息
    #im_proj = in_ds.GetProjection()  # 获取投影信息
    #im_data = in_ds.ReadAsArray()  # 获取数据

    # 获取原图的原点坐标信息，# 获取仿射矩阵信息
    ori_transform = in_ds.GetGeoTransform()
    #if ori_transform:
        #print(ori_transform)
        #print("Origin = ({}, {})".format(ori_transform[0], ori_transform[3]))
        #print("Pixel Size = ({}, {})".format(ori_transform[1], ori_transform[5]))

    # 定义切图的起始点坐标
    offset_x = coords_UL[0]
    offset_y = coords_UL[1]

    # 读取原图仿射变换参数值
    top_left_x = ori_transform[0]  # 左上角x坐标
    w_e_pixel_resolution = ori_transform[1]  # 东西方向像素分辨率
    top_left_y = ori_transform[3]  # 左上角y坐标
    n_s_pixel_resolution = ori_transform[5]  # 南北方向像素分辨率

    # 读取原图中的每个波段
    in_band = in_ds.GetRasterBand(1)
    datatype = in_band.DataType

    # 定义切图的大小（矩形框）
    block_xsize = coords_DR[0] - coords_UL[0]
    #block_xsize = min(abs(x_delta_tmp),abs(projection_threshod+x_delta_tmp))
    #block_xsize = (coords_DR[0] - coords_UL[0]) # 行
    block_ysize = coords_DR[1] - coords_UL[1] # 列

    col = int(np.round(block_xsize))
    row = int(np.round(block_ysize))
    img_colrow = int(np.round(central_length/abs(w_e_pixel_resolution)))

    intersection_status = intersectType(offset_x,offset_y,width,height,block_xsize,block_ysize)
    if(len(intersection_status)> 0):
        return False,intersection_status,is_LargeIMG
 
    ## 从每个波段中切需要的矩形框内的数据(注意读取的矩形框不能超过原图大小)
    out_band = in_band.ReadAsArray(int(offset_x), int(offset_y), col, row,
                                   buf_xsize = img_colrow, buf_ysize = img_colrow)
    # 获取Tif的驱动，为创建切出来的图文件做准备
    gtif_driver = gdal.GetDriverByName("GTiff")

    # 创建切出来的要存的文件（3代表3个不都按，最后一个参数为数据类型，跟原文件一致）
    options=["TILED=YES", "COMPRESS=LZW","NUM_THREADS=ALL_CPUS"]
    #options=["TILED=YES","NUM_THREADS=ALL_CPUS"]
    ending = '.jpg'
    if(img_colrow>threshod):
        saving_path_root += '/LargeIMG/'
        options.append("BIGTIFF=IF_SAFER")
        is_LargeIMG = 1
        if(img_colrow*5>150000):
            ending = '.tiff'
    out_ds_JPG = gtif_driver.Create(saving_path_root + ID + ending, img_colrow, img_colrow, outbandsize, datatype, options=options)
    #, options=["TILED=YES", "COMPRESS=LZW"])
    #out_ds_TIF = gtif_driver.Create(saving_path_root + 'tiff' + '/' + ID + '.tiff', col, col, outbandsize, datatype, options=["TILED=YES", "COMPRESS=LZW","NUM_THREADS=ALL_CPUS"])

    # 根据反射变换参数计算新图的原点坐标
    top_left_x = top_left_x + offset_x * w_e_pixel_resolution
    top_left_y = top_left_y + offset_y * n_s_pixel_resolution
    height_resize_factor = col/row
    width_resize_factor = col/block_xsize
    scale_resize_factor = col/img_colrow
    # 将计算后的值组装为一个元组，以方便设置
    dst_transform = (top_left_x, ori_transform[1]/width_resize_factor*scale_resize_factor, ori_transform[2], top_left_y, ori_transform[4], ori_transform[5]/height_resize_factor*scale_resize_factor)

    # 设置裁剪出来图的原点坐标
    #if(block_xsize<threshod):
    out_ds_JPG.SetGeoTransform(dst_transform)
    out_ds_JPG.SetProjection(in_ds.GetProjection())
    out_ds_JPG.GetRasterBand(1).WriteArray(out_band)
    out_ds_JPG.FlushCache()
    del out_ds_JPG

    # 设置SRS属性（投影信息）
    # 写入目标文件
    #使用更大的缓冲读取影像，与重采样后影像行列对应
    # 将缓存写入磁盘
    #out_ds_TIF.SetGeoTransform(dst_transform)
    #out_ds_TIF.SetProjection(in_ds.GetProjection())
    #out_ds_TIF.GetRasterBand(1).WriteArray(out_band)
    #out_ds_TIF.FlushCache()
    #del out_ds_TIF
    # 计算统计值
    # for i in range(1, 3):
    #     out_ds.GetRasterBand(i).ComputeStatistics(False)
    # print("ComputeStatistics succeed")
    
    return True,intersection_status,is_LargeIMG

def tet_extraction(rootFolderPath,stantard_posi = 30):
    global ImageDict
    area_dict = {}#dictationary with stable .clip
    dirs = os.listdir(rootFolderPath)# 读取所有的文件
    for file in dirs:
        sourcePath,CorrespondFolder = correspondFileMatch(file)
        if(sourcePath):
            area_dict.update({CorrespondFolder:sourcePath})
    return area_dict

def GetExtent(infile):
    ds = gdal.Open(infile)
    geotrans = ds.GetGeoTransform()
    xsize = ds.RasterXSize
    ysize = ds.RasterYSize
    min_x, max_y = geotrans[0], geotrans[3]
    max_x, min_y = geotrans[0] + xsize * geotrans[1], geotrans[3] + ysize * geotrans[5]
    ds = None
    return min_x, max_y, max_x, min_y

def rect_overlap(rect1,rect2):
    """
        rect1：矩形1，四元组，左上角坐标和右下角坐标
        rect2：矩形2，四元组，左上角坐标和右下角坐标
        本函数返回矩形相交区域的矩形
    """
    [x11,y11,x12,y12] = rect1     # 矩形1左上角(x11,y11)和右下角(x12,y12) 
    [x21,y21,x22,y22] = rect2     # 矩形2左上角(x21,y21)和右下角(x22,y22)
    
    # 下面求最小的外包矩形
    #startx = min(x11,x21)         # 外包矩形在x轴上左边界
    #endx = max(x12,x22)           # 外包矩形在x轴上右边界
    #starty = min(y12,y22)         # 外包矩形在y轴上上边界
    #endy = max(y11,y21)           # 外包矩形在y轴上下边界
    # 想像一下两个矩形隔得比较远，那么外包矩形的宽度是不是肯定大于两个矩形的宽度和呢？
    # 所以，两个矩形相交，它们的宽度和必然大于最小外包矩形的宽度，它们的高度和也是必然大于外包矩形的高度
    #width = (x12-x11) + (x22-x21) - (endx-startx)      # (endx-startx)表示外包矩形的宽度
    #height = - (y11-y12) - (y21-y22) + (endy-starty)     # (endy-starty)表示外包矩形的高度
    #height = -height
    #print(width,height)
    #if(width<0 or height < 0):
        #return False                             
    # 相交
    X1 = max(x11,x21)        # 有相交则相交区域位置
    Y1 = max(y11,y21)
    X2 = min(x12,x22)
    Y2 = min(y12,y22)
    
    #area = width * height        # 相交区域面积
    return X1,Y1,X2,Y2

def RdinIMG(in_ds,geotrans,columns,rows,out_ds):
    in_gt = in_ds.GetGeoTransform()

    #x_deviation = geotrans[0]-in_gt[0]
    x_delta_tmp = geotrans[0]-in_gt[0]
    map_threashod = projection_threshod * in_gt[1]
    x_deviation = x_delta_tmp if x_delta_tmp == -(map_threashod / 2) \
        else (abs(x_delta_tmp)//(map_threashod/2)) * np.sign(-x_delta_tmp) * map_threashod + x_delta_tmp
    # x_deviation = min(abs(x_delta_tmp),abs(projection_threshod * in_gt[1]-x_delta_tmp)) * np.sign(x_delta_tmp)
    y_deviation = geotrans[3]-in_gt[3]

    offset_x = x_deviation.get()/in_gt[1]
    offset_y = y_deviation/in_gt[5]

    rectImg = list([0,0,in_ds.RasterXSize,in_ds.RasterYSize])
    rectClip = list([offset_x,offset_y,offset_x + columns,offset_y + rows])

    x_UL,y_UL,x_DR,y_DR = rect_overlap(rectImg,rectClip)

    block_xsize = int(np.round(x_DR - x_UL))
    block_ysize = int(np.round(y_DR - y_UL))

    if(block_xsize<0 or block_ysize<0):
        return

    x_Clip = int(np.round(x_UL - offset_x))
    y_Clip = int(np.round(y_UL - offset_y))

    #for i in range(3):  #每个波段都要考虑 
    data = in_ds.GetRasterBand(1).ReadAsArray(int(x_UL), int(y_UL), block_xsize, block_ysize)
    out_ds.GetRasterBand(1).WriteArray(data,x_Clip,y_Clip)  # x，y是开始写入时左上角像元行列号


def RasterMosaic(file_list, saving_path_root, crater_record):
    #global projection_threshod
    #min_x, max_y, max_x, min_y = GetExtent(file_list[0])
    #for infile in file_list:
    #    minx, maxy, maxx, miny = GetExtent(infile)
    #    min_x, min_y = min(min_x, minx), min(min_y, miny)
    #    max_x, max_y = max(max_x, maxx), max(max_y, maxy)
    #in_ds = gdal.Open(file_list[0])
    tif_path = saving_path_root + crater_record['craterID'] + 'tmp.tiff'
    in_ds = file_list[0]
    in_band = in_ds.GetRasterBand(1)
    outbandsize = in_ds.RasterCount  # 获取数据波段数

    geotrans = list(in_ds.GetGeoTransform())
    w_e, n_s = geotrans[1], geotrans[5]
    columns = int(crater_record['coords_DR'][0] - crater_record['coords_UL'][0])
    #columns = min(abs(col_tmp),abs(projection_threshod + col_tmp))  # 列数
    rows = int(crater_record['coords_DR'][1] - crater_record['coords_UL'][1]) # 行数

    driver = gdal.GetDriverByName('GTiff')
    out_ds = driver.Create(tif_path, columns, rows, outbandsize, in_band.DataType, options=["NUM_THREADS=ALL_CPUS","BIGTIFF=YES"])
    
    out_ds.SetProjection(in_ds.GetProjection())
    geotrans[0] += float(crater_record['coords_UL'][0]) * w_e  # 更正左上角坐标
    geotrans[3] += float(crater_record['coords_UL'][1]) * n_s

    out_ds.SetGeoTransform(geotrans)

    thread = []
    for in_ds in file_list:
        t = Thread(target=RdinIMG,args=(in_ds,geotrans,columns,rows,out_ds))
        t.start()
        thread.append(t)
    for t in thread:
        t.join()
    #inv_geotrans = gdal.InvGeoTransform(geotrans)
    '''
    for in_ds in file_list:
        #in_ds = gdal.Open(in_fn)
        in_gt = in_ds.GetGeoTransform()

        #x_deviation = geotrans[0]-in_gt[0]
        x_delta_tmp = geotrans[0]-in_gt[0]
        map_threashod = projection_threshod * in_gt[1]
        x_deviation = x_delta_tmp if x_delta_tmp == -(map_threashod / 2) \
            else (abs(x_delta_tmp)//(map_threashod/2)) * np.sign(-x_delta_tmp) * map_threashod + x_delta_tmp
        # x_deviation = min(abs(x_delta_tmp),abs(projection_threshod * in_gt[1]-x_delta_tmp)) * np.sign(x_delta_tmp)
        y_deviation = geotrans[3]-in_gt[3]

        offset_x = x_deviation.get()/in_gt[1]
        offset_y = y_deviation/in_gt[5]

        rectImg = list([0,0,in_ds.RasterXSize,in_ds.RasterYSize])
        rectClip = list([offset_x,offset_y,offset_x + columns,offset_y + rows])

        x_UL,y_UL,x_DR,y_DR = rect_overlap(rectImg,rectClip)

        block_xsize = int(np.round(x_DR - x_UL))
        block_ysize = int(np.round(y_DR - y_UL))

        if(block_xsize<0 or block_ysize<0):
            continue

        x_Clip = int(np.round(x_UL - offset_x))
        y_Clip = int(np.round(y_UL - offset_y))

        #for i in range(3):  #每个波段都要考虑 
        data = in_ds.GetRasterBand(1).ReadAsArray(int(x_UL), int(y_UL), block_xsize, block_ysize)
        out_ds.GetRasterBand(1).WriteArray(data,x_Clip,y_Clip)  # x，y是开始写入时左上角像元行列号
    '''
    # 将缓存写入磁盘
    out_ds.FlushCache()
    del out_ds
    return tif_path

def RasterResize(tif_path,saving_path_root,crater_record):
    is_LargeIMG = 0
    in_ds=gdal.Open(tif_path)
    in_band=in_ds.GetRasterBand(1)
    ID = crater_record['craterID']

    xsize=in_band.XSize
    ysize=in_band.YSize

    height_resize_factor = xsize/ysize
    # 将计算后的值组装为一个元组，以方便设置
    geotrans=list(in_ds.GetGeoTransform())
    geotrans[5] /= height_resize_factor

    img_colrow = int(np.round(crater_record['fig length']/abs(geotrans[1])))
    
    scale_resize_factor = xsize/img_colrow
    geotrans[1] *= scale_resize_factor
    geotrans[5] *= scale_resize_factor
    #重采样后的影像
    out_band = in_band.ReadAsArray(buf_xsize = img_colrow,buf_ysize = img_colrow)
    # 获取Tif的驱动，为创建切出来的图文件做准备
    gtif_driver = gdal.GetDriverByName("GTiff")

    # 创建切出来的要存的文件（3代表3个不都按，最后一个参数为数据类型，跟原文件一致）
    options=["TILED=YES", "COMPRESS=LZW","NUM_THREADS=ALL_CPUS"]
    #options=["TILED=YES","NUM_THREADS=ALL_CPUS"]
    ending = '.jpg'
    if(img_colrow>threshod):
        saving_path_root += '/LargeIMG/'
        options.append("BIGTIFF=IF_SAFER")
        is_LargeIMG = 1
        if(crater_record['fig length']>KM_threshod*1000*2):
            ending = '.tiff'
    out_ds_JPG = gtif_driver.Create(saving_path_root + ID + ending, img_colrow, img_colrow, 1, in_band.DataType, options=options)
    #, options=["TILED=YES", "COMPRESS=LZW"])
    #out_ds_TIF = gtif_driver.Create(saving_path_root + 'tiff' + '/' + ID + '.tiff', xsize, xsize, 1, in_band.DataType, options=["TILED=YES", "COMPRESS=LZW","NUM_THREADS=ALL_CPUS"])

    #if(xsize<threshod):
    # 设置裁剪出来图的原点坐标
    out_ds_JPG.SetGeoTransform(geotrans)
    out_ds_JPG.SetProjection(in_ds.GetProjection())
    out_ds_JPG.GetRasterBand(1).WriteArray(out_band)
    out_ds_JPG.FlushCache()
    del out_ds_JPG
    return is_LargeIMG

    #out_ds_TIF.SetGeoTransform(geotrans)
    # 设置SRS属性（投影信息
    #out_ds_TIF.SetProjection(in_ds.GetProjection())
    # 写入目标文件
    #使用更大的缓冲读取影像，与重采样后影像行列对应
    #out_ds_TIF.GetRasterBand(1).WriteArray(out_band)
    # 将缓存写入磁盘
    #out_ds_TIF.FlushCache()
    #del out_ds_TIF
    # 计算统计值
    # for i in range(1, 3):
    #     out_ds.GetRasterBand(i).ComputeStatistics(False)
    # print("ComputeStatistics succeed")
        
def Cut_and_Mosaic(file_list,save_path_root,crater_record):
    global Crater_Dict
    tmp_path = tempfile.gettempdir()
    tif_path = RasterMosaic(file_list,tmp_path,crater_record)
    is_LargeIMG = RasterResize(tif_path,save_path_root,crater_record)
    coor_info = label4processed_crater(crater_record['XY_UL'],crater_record['XY_DR'])
    scale = crater_record['scale']
    box_info = [0.5,0.5,1/scale,1/scale]
    Crater_Dict[crater_record['craterID']].update({'is_cut' : '1',
                                                'box info':" ".join(str(i) for i in box_info),
                                                'coordinary info':" ".join(str(i) for i in coor_info),
                                                'is_AUTOvalidated' : '{0}'.format(is_LargeIMG),
                                                'is_MANUALvalidated' : '{0}'.format(is_LargeIMG)})
    #print(crater_record['craterID'])
    #del_idx.append(idx)
    try:
        os.remove(tif_path)
    except:
        pass
    return Unprocessed_Craters,Crater_Dict

def imgList4Mosaicing(CorrespondFolder,crater_record,activatedIMG,IMGwidth,IMGheight):
    file_list = [activatedIMG[CorrespondFolder]]
    folder_lonlat = [float(x[1:]) for x in CorrespondFolder.split("_")]
    coords_UL = crater_record['coords_UL']
    coords_DR = crater_record['coords_DR']

    x_min = int(np.floor((coords_UL[0])/IMGwidth))
    y_min = int(np.floor((coords_UL[1])/IMGheight))
    x_max = int(np.floor((coords_DR[0])/IMGwidth))
    y_max = int(np.floor((coords_DR[1])/IMGwidth))

    for idx in range(x_min,x_max+1):
        for idy in range(y_min,y_max+1):
            if(idx == 0 and idy == 0):
                continue
            lon_tmp = (folder_lonlat[0] + idx * args.clipStep)
            lon = lon_tmp if lon_tmp == -180 else (abs(lon_tmp)//180) * np.sign(-lon_tmp) * 360 + lon_tmp #这样可以让lon_tmp = -180
            lat = folder_lonlat[1] - idy * args.clipStep
            img_serial = 'E{0:04d}_N{1:03d}'.format(int(lon),int(lat))
            if(img_serial in activatedIMG):
                file_list.append(activatedIMG[img_serial])
            else:
                return False
    #for each in file_list:
        #print(each.GetGeoTransform())
    return file_list

def imgDict4Mosaicing(area_dict,Unprocessed_Craters,CorrespondFolder,IMGwidth,IMGheight):
    folder_lonlat = [float(x[1:]) for x in CorrespondFolder.split("_")]
    img = {}
    for crater_record in Unprocessed_Craters:
        #img_intersect_type = crater_record['intersection type']
        coords_UL = crater_record['coords_UL']
        coords_DR = crater_record['coords_DR']

        x_min = int(np.floor((coords_UL[0])/IMGwidth))
        y_min = int(np.floor((coords_UL[1])/IMGheight))
        x_max = int(np.floor((coords_DR[0])/IMGwidth))
        y_max = int(np.floor((coords_DR[1])/IMGwidth))

        for idx in range(x_min,x_max+1):
            for idy in range(y_min,y_max+1):
                lon_tmp = (folder_lonlat[0] + idx * args.clipStep)
                lon = lon_tmp if lon_tmp == -180 else (abs(lon_tmp)//180) * np.sign(-lon_tmp) * 360 + lon_tmp #这样可以让lon_tmp = -180
                lat = folder_lonlat[1] - idy * args.clipStep
                img_serial = 'E{0:04d}_N{1:03d}'.format(int(lon),int(lat))
                if(img_serial in area_dict):
                    if(img_serial in img):
                        pass
                    else:
                        img.update({img_serial:area_dict[img_serial]})
    if(len(img)>1):
        for each in img:
            img_open  = gdal.Open(img[each]) 
            img.update({each:img_open})
        #for each in img:
            #print(img[each].GetGeoTransform())
        return img
    else:
        return False

def Mosaicing_Unprocessed_Craters(area_dict,save_path_root,IMGwidth,IMGheight):
    global Unprocessed_Craters,Crater_Dict,CorrespondFolder
    activatedIMG = imgDict4Mosaicing(area_dict,Unprocessed_Craters,CorrespondFolder,IMGwidth,IMGheight)
    if(activatedIMG):
        idx = 0
        thread = []
        while idx < len(Unprocessed_Craters):
            crater_record = Unprocessed_Craters[idx]
            file_list = imgList4Mosaicing(CorrespondFolder,crater_record,activatedIMG,IMGwidth,IMGheight)
            #print(image_dict)
            if(file_list):
                t = Thread(target=Cut_and_Mosaic,args=(file_list,save_path_root,crater_record))
                t.start()
                thread.append(t)
                #Cut_and_Mosaic(file_list,save_path_root,crater_record)
                Unprocessed_Craters.remove(crater_record)
                #time.sleep(0.2)
                idx-=1
            idx+=1
        for t in thread:
            t.join()
        for each in activatedIMG:
            #activatedIMG_single = activatedIMG[each]
            del activatedIMG[each]
            activatedIMG[each] = 0
    #return Unprocessed_Craters,Crater_Dict

def Unprocessed_Craters_csv(saving_path_root,Unprocessed_Craters):
    # 使用os.path.splitext()函数获取不带扩展名的文件名
    with open(saving_path_root+'Unprocessed_Craters.csv', 'w',newline='') as f:
        writer = csv.writer(f, delimiter=',')  
        writer.writerow(['CraterID','Intersection_type','Upperleft_X','Upperleft_Y','Downright_X','Downright_Y','Upperleft_i','Upperleft_j','Downright_i','Downright_j'] )  

        for crater in Unprocessed_Craters:
            Intersection_type = " ".join(str(i) for i in crater['intersection type'])
            XY_UL = crater['XY_UL']
            XY_DR = crater['XY_DR']
            UL = crater['coords_UL']
            DR = crater['coords_DR']
            writer.writerow([crater['craterID'],Intersection_type,XY_UL[0],XY_UL[1],XY_DR[0],XY_DR[1],UL[0],UL[1],DR[0],DR[1]])
            #print([row['LON_E'], row['LAT'], row['DiamKM'],row['CraterID']])
        Unprocessed_Craters = []

def target_extraction(rootFolderPath,stantard_posi = 30):
    target_area = []
    dirs = os.listdir(rootFolderPath)# 读取所有的文件
    for file in dirs:
        if os.path.splitext(file)[1] == '':   # 只读取固定后缀的文件
            foldername = file[stantard_posi:].split('_')# 截取文件名字符
            if(len(foldername)>1):
                lon_lat = [float(foldername[0][1:]),float(foldername[1][1:])]
                target_area.append(lon_lat)
    return target_area   

def table_targetMatching(all_area,table,maunal = False):
    area_dict = {}#dictationary with stable arg.clip
    for target_area in all_area:
        if(maunal):
            selected_area = [[target_area[0][0],target_area[0][1]],[target_area[1][0],target_area[1][1]]]
        else:
            selected_area = [[target_area[0],target_area[1]],[target_area[0]+args.clipStep,target_area[1]+args.clipStep]]
        xy_floored = numpy.floor([float(selected_area[0][0])/args.clipStep,float(selected_area[0][1])/args.clipStep])*args.clipStep
        area_dict.update({'{0}/{1}'.format(int(xy_floored[0]),int(xy_floored[1])):[]})
    return area_dict

def label4processed_crater(coords_UL,coords_DR):
    #x_center y_center width height
    #框坐标必须为归一化的xywh格式(0 - 1范围)。如果框是以像素为单位,请将`x_center`和`width`除以图像宽度,y_center和`height`除以图像高度
    #info1 = [x,y,width_percent,height_percent]
    coor_info = [coords_UL[0],coords_UL[1],
                 coords_DR[0],coords_DR[1]]
    return coor_info

#大地主题解算
def relative_BLforDistance(lat,lon,distance,Azimuth):
    a = datum_parameter['Equatorial Radius'] # 长半轴，单位m
    f = 1/datum_parameter['1/e'] # 扁率
    e2 = f * (2.0 - f) # 第一偏心率的平方
    # 计算卯酉圈曲率半径，单位m
    N = a / np.sqrt(1.0 - e2 * np.sin(lat)**2)
    M = a * (1 - e2) / (1 - e2 * np.sin(lat)**2)**(3/2)

    #正算，带入大地线S(distance)与方位角A(Azimuth)求出另一点经纬度L，B
    bk=distance*np.cos(Azimuth)/M;
    ak=np.tan(lat)*distance*np.sin(Azimuth)/N;
    lk=distance*np.sin(distance)/(N*np.cos(lat));
    delta=5e-7
    dif=1;
    mm=nm=0
    while dif>delta:
        ak0=ak;
        bk0=bk;
        lk0=lk;
        bm=lat+ 0.5*bk;
        am=Azimuth+0.5*ak;
        wm=np.sqrt(1-e2*np.sin(bm)*np.sin(bm))
        mm=a*(1-e2)/wm**3
        nm=a/wm
        b0=distance*np.cos(am)/mm;
        l0=distance*np.sin(am)/(nm*np.cos(bm));
        a0=l0*np.sin(bm);
        lk=l0*(1+a0*a0/24.0-b0*b0/24.0);
        bk=b0*(1+l0*l0*np.cos(bm)*np.cos(bm)/12.0+a0*a0/8.0);
        ak=a0*(1+b0*b0/12.0+l0*l0*(1+np.cos(bm)*np.cos(bm))/24.0);
        dif1= np.array([np.abs(lk-lk0),np.abs(ak-ak0),np.abs(bk-bk0)]);
        dif=np.linalg.norm(dif1,ord=np.inf);
    return [(lon + lk)*180/np.pi , (lat + bk)*180/np.pi , (am+0.5*ak+np.pi)*180/np.pi]

def single_IMG(dataset,Crater_Dictrecord,ID,i,saving_path_root,dict_sum,Unprocessed_Craters):
    global thread_num,projection_threshod
    thread_num+=1
    lon = float(Crater_Dictrecord['Lon'])/180*np.pi
    lat = float(Crater_Dictrecord['Lat'])/180*np.pi
    diameter = float(Crater_Dictrecord['Diameter']) * 1000

    if(diameter > diameter_threshod):
        scale = 2
    else:
        scale = diameter_threshod/diameter*2

    UL_lla = relative_BLforDistance(lat,lon,diameter*scale/np.sqrt(2),np.pi*(7/4))
    DR_lla = relative_BLforDistance(lat,lon,diameter*scale/np.sqrt(2),np.pi*(3/4))
    central_length = diameter*scale

    xy_UL = lonlat2geo(dataset,float(UL_lla[0]),float(UL_lla[1]))
    xy_DR = lonlat2geo(dataset,float(DR_lla[0]),float(DR_lla[1]))

    #coords_ij = geo2imagexy(dataset, x, y)
    coords_UL = geo2imagexy(dataset, xy_UL[0], xy_UL[1])
    coords_DR = geo2imagexy(dataset, xy_DR[0], xy_DR[1])

    if(projection_threshod == 0):
        projection_threshod = int(dataset.RasterXSize * 360 / args.clipStep)

    coords_UL[0] = coords_UL[0] if coords_UL[0] == -(projection_threshod / 2) \
        else ((abs(coords_UL[0])//(projection_threshod/2)) * np.sign(-coords_UL[0]) * projection_threshod + coords_UL[0]).get()
    coords_DR[0] = coords_DR[0] if coords_DR[0] == -(projection_threshod / 2) \
        else ((abs(coords_DR[0])//(projection_threshod/2)) * np.sign(-coords_DR[0]) * projection_threshod + coords_DR[0]).get()

    CutAvailability,intersection_type,is_LargeIMG = ImageCut(dataset,saving_path_root,ID,coords_UL,coords_DR,central_length)
    if(CutAvailability==False):
        crater_record = {}
        crater_record.update({'craterID':ID,
                             #'Lon':lon,
                             #'Lat':lat,
                             #'diameter':diameter,
                             'intersection type':intersection_type,
                             #'coords_XY':coords_XY,
                             'XY_UL':xy_UL,
                             'XY_DR':xy_DR,
                             'coords_UL':coords_UL,
                             'coords_DR':coords_DR,
                             'fig length':central_length,
                             'scale':scale
                             })
        Unprocessed_Craters.append(crater_record)
    else:
        #width = dataset.RasterXSize  # 获取数据宽度
        #height = dataset.RasterYSize  # 获取数据高度
        coords_UL = xy_UL
        coords_DR = xy_DR
        coor_info = label4processed_crater(coords_UL,coords_DR)
        box_info = [0.5,0.5,1/scale,1/scale]
        Crater_Dictrecord.update({'is_cut' : '1',
                     'box info':" ".join(str(i) for i in box_info),
                     'coordinary info':" ".join(str(i) for i in coor_info),
                     'is_AUTOvalidated' : '{0}'.format(is_LargeIMG),
                     'is_MANUALvalidated' : '{0}'.format(is_LargeIMG)})
    print('{0} {1} {2}\n'.format(i,dict_sum,ID),end = '')
    sys.stdout.flush() 
    thread_num-=1

def CraterCSV_Extraction(path):
    with open(path, newline='') as f:
        Crater_Info = csv.DictReader(f)
        return [*Crater_Info]
    
def CraterCSV_Saving(CraterCSV_Dict,csv_path):
    df = pd.DataFrame(CraterCSV_Dict)
    df.to_csv(csv_path, index=False, header=True)
    return CraterCSV_Dict

def UpperCSV_Extraction(path):
    Dict = {}
    with open(path, newline='') as f:
        _Info = csv.DictReader(f)
        _Info = [*_Info]
    if(len(_Info)>0):
        main_key = list(_Info[0].keys())[0]
        for each in _Info:
            itemID = each[main_key]
            each.pop(main_key)
            Dict.update({itemID:each})
        return Dict,main_key
    else:
        return False,False

def UpperCSV_Saving(_Dict,mainkey,csv_path):
    Final_dict = []
    for each in _Dict:
        mainkey_dict = {mainkey:each}
        Final_dict.append(dict(mainkey_dict,**_Dict[each]))
    return CraterCSV_Saving(Final_dict,csv_path)

@atexit.register
def MainCSV():
    global ImageDict,mainkey,ImageStatus_path,image_cutted,datacsv_path
    Final_dict = []
    if(image_cutted>0):
        for each in ImageDict:
            mainkey_dict = {mainkey:each}
            Final_dict.append(dict(mainkey_dict,**ImageDict[each]))
        df = pd.DataFrame(Final_dict)
        df.to_csv(ImageStatus_path, index=False, header=True)
    #print("{0} Images Cutted".format(image_cutted))
    print('done')
    sys.stdout.flush() 

def Image_Cutting(sourcePath,subFolder,area_dict):
    global thread_num,Unprocessed_Craters,Crater_Dict
    status = False
    datacsv_path = subFolder + 'data.csv'
    Crater_Dict,CraterMainkey = UpperCSV_Extraction(datacsv_path)
    if(Crater_Dict != False and len(Crater_Dict)>0):
        saving_path_root = subFolder
        i = 0
        dataset = gdal.Open(sourcePath) 
        try:
            os.mkdir(saving_path_root)        
        except FileExistsError:
            pass
            #print('Root Folder already exists!')
        try:
            os.mkdir(saving_path_root+'/LargeIMG')
        except FileExistsError:
            pass
            #print('LargeIMG Folder already exists!')
        threads = []
        for ID in Crater_Dict:
            i+=1
            if(float(Crater_Dict[ID]['Diameter'])<KM_threshod):
                while(True):
                    if(thread_num>31):
                        time.sleep(0.2)
                        continue
                    else:
                        if(args.reCutCrater == 1 or Crater_Dict[ID]['is_cut']=='0'):
                            t = Thread(target=single_IMG,args=(dataset,Crater_Dict[ID],ID,i,saving_path_root,len(Crater_Dict),Unprocessed_Craters))
                            t.start()
                            threads.append(t)
                        break   
        if(len(threads)>0):
            for t in threads:
                t.join()
            status = True
            Mosaicing_Unprocessed_Craters(area_dict,saving_path_root,dataset.RasterXSize,dataset.RasterYSize)
            UpperCSV_Saving(Crater_Dict,CraterMainkey,datacsv_path)
    return Unprocessed_Craters,status

projection_threshod = 0
Unprocessed_Craters = []
Crater_Dict = {}
ImageDict = {}
datacsv_path = ''
mainkey = ''
ImageStatus_path = ''
image_cutted = 0
CorrespondFolder = ''

def correspondFileMatch(file):
    global ImageDict
    if os.path.splitext(file)[1] == '':   # 只读取固定后缀的文件
        foldername = file[startposition:].split('_')# 截取文件名字符
        if(len(foldername)>1):
            lon_lat = [float(foldername[0][1:]),float(foldername[1][1:])]
            CorrespondFolder = 'E{0:04d}_N{1:03d}'.format(int(lon_lat[0]),int(lon_lat[1]))
            path = args.rootFolder+"{0}/".format(file)
            f_list = os.listdir(path)
            if(len(f_list)==1):
                path = args.rootFolder+"{0}/{0}/".format(file)
                f_list = os.listdir(path)
            # print f_list
            for i in f_list:
                if os.path.splitext(i)[1] == '.tif':
                    sourcePath = path + i
                    return sourcePath,CorrespondFolder
    return False,False

def main(args):
    if(args.Cut==1):
        gdal.AllRegister()
        dirs = os.listdir(args.rootFolder)# 读取所有的文件
        global ImageDict,mainkey,ImageStatus_path,image_cutted,CorrespondFolder,Unprocessed_Craters
        ImageStatus_path = args.saveFolder + 'ImageStatus.csv'
        ImageDict,mainkey = UpperCSV_Extraction(ImageStatus_path)
        area_dict = tet_extraction(args.rootFolder)
        for idx in range(len(dirs)):
            file = dirs[idx]
            Unprocessed_Craters =  []
            print('{0} {1} {2}\n'.format(idx+1,len(dirs),'start'),end = '')
            sys.stdout.flush() 
            sourcePath,CorrespondFolder = correspondFileMatch(file)
            if(CorrespondFolder):
                if(args.reCutImage == 1 or ImageDict[CorrespondFolder]['is_cut'] == '0'):
                    if(sourcePath):
                        Status = Image_Cutting(sourcePath,args.saveFolder+CorrespondFolder+"/",area_dict)
                        ImageDict[CorrespondFolder]['is_cut'] = 1
                        if(len(Unprocessed_Craters)>0):
                            Unprocessed_Craters_csv(args.saveFolder+CorrespondFolder+"/",Unprocessed_Craters)
                        else:
                            if(Status):
                                try:
                                    os.remove(args.saveFolder+CorrespondFolder+"/Unprocessed_Craters.csv")
                                except:
                                    pass
                    else:
                        #print('TIFF not Exist\n',end = '')
                        #sys.stdout.flush() 
                        pass
            #print("{0} has been Cut".format(CorrespondFolder))
            image_cutted+=1
            print('{0} {1} {2}\n'.format(idx+1,len(dirs),'end'),end = '')
            sys.stdout.flush() 

def parse_opt():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument("--rootFolder", type=str, 
                        default=r"RealDatainBelt/")
    parser.add_argument("--clipStep", type=float, default=4)
    parser.add_argument("--saveFolder", type=str, 
                        default="Processed/")
    parser.add_argument("--Cut", type=int, default=1)
    parser.add_argument("--reCutImage", type=int, default=0)
    parser.add_argument("--reCutCrater", type=int, default=0)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_opt()
    try:
        main(args)
        time.sleep(1)
    except (KeyboardInterrupt, SystemExit):
        print('Program Interrupted')
    exit()