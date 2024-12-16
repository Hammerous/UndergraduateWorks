import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pymysql #数据库模块
import math# 数学模块
import numpy as np
import cupy as cp
from decimal import Decimal
from mpl_toolkits.basemap import Basemap  #地图底图库
import time

#核心测站列表
core_list=["AREQ",	"SCRZ",	"UNSA",		
        "BADG",	"ULAB",	"IRKJ",	"IRKT",	
        "BAKE",	"CHUR",	"YELL",	"HOLM",	
        "BHR4",	"BHR1",	"BAHR",	"YIBL",	"NAMA",
        "CHPI",	"UFPR",	"BRAZ",	"SAVO",	
        "CHTI",	"CHAT",			
        "CKIS",	"NIUM",	"ASPA",	"FALE",	
        "CPVG",	"DAKR",	"DAKA",		
        "CRO1",	"ABMF",	"LMMF",	"RDSD",	"SCUB",
        "CZTG",				
        "DAEJ",	"SUWN",	"TAEJ",		
        "DARW",	"KAT1",	"TOW2",	"ALIC",	
        "DAV1",	"MAW1",			
        "DGAR",	"SEYG",	"SEY1",		
        "FAIR",	"WHIT",	"INVK",		
        "GAMB",				
        "GLPS",	"GALA",			
        "GODN",	"GODE",	"ALGO",	"NRC1",	
        "GOUG",				
        "GUAM",	"CNMR",			
        "GUAT",				
        "HRAO",	"SUTM",	"SUTH",	"SBOK",	"HNUS",
        "IISC",	"HYDE",	"SGOC",		
        "ISPA",	"EISL",			
        "KIRI",	"NAUR",	"MAJU",	"KWJ1",	"POHN",
        "KIRU",	"TRO1",	"TROM",	"NYAL",	
        "KOKB",	"HNLC",			
        "KRGG",	"KERG",			
        "KZN2",	"ARTU",	"MDVJ",	"MDVO",	
        "MAC1",	"OUS2",	"MQZG",		
        "MAL2",	"MALI",	"RCMN",	"NURK",	"MBAR",
        "MAS1",	"LPAL",	"FUNC",		
        "MATE",	"NOT1",	"NOTO",	"M0SE",	"CAGL",
        "MCIL",	"CCJ2",	"CCJM",		
        "MKEA",	"MAUI",			
        "MOBS",	"SYDN",	"TIDB",	"TID1",	"HOB2",
        "NANO",	"ALBH",	"BREW",	"DRAO",	"WILL",
        "NKLG",	"BJCO",			
        "NNOR",	"YAR2",	"YAR1",	"PERT",	"MRO1",
        "NRMD",	"NOUM",	"KOUC",	"LAUT",	"TUVA",
        "OHI3",	"OHI2",	"OHIG",	"PALM",	
        "POL2",	"CHUM",	"KIT3",	"GUAO",	"URUM",
        "REUN",	"VACS",	"ABPO",	"VOIM",	
        "RIO2",	"RIOG",	"PARC",	"FALK",	
        "SALU",	"BRFT",	"FORT",	"KOUG",	"KOUR",
        "SANT",	"ANTC",	"CONZ",	"CFAG",	"LPGS",
        "SCTB",	"MCM4",	"DUM1",	"CAS1",	
        "STHL",	"ASCG",	"ASC1",		
        "STJ3",	"STPM",	"STJO",	"HLFX",	"NAIN",
        "THU2",	"THU1",	"KELY",	"QAQ1",	"SCOR",
        "THTG",	"THTI",	"FAA1",		
        "TNML",	"TCMS",	"CKSV",	"KMNM",	"SHAO",
        "VESL",	"SYOG",			
        "VNDP",	"MONP",	"GOLD",	"PIE1",	"QUIN",
        "XMIS",	"COCO",	"BAKO",	"SIN1",	"NTUS"
 ]

#简化后的核心测站列表  每一行只取第一个
"""
core_list_omit=["AREQ",
        "BADG",
        "BAKE",
        "BHR4",
        "CHPI",
        "CHTI",
        "CKIS",
        "CPVG",
        "CRO1",
        "CZTG",
        "DAEJ",
        "DARW",
        "DAV1",
        "DGAR",
        "FAIR",
        "GAMB",
        "GLPS",
        "GODN",
        "GOUG",
        "GUAM",
        "GUAT",
        "HRAO",
        "IISC",
        "ISPA",
        "KIRI",
        "KIRU",
        "KOKB",
        "KRGG",
        "KZN2",
        "MAC1",
        "MAL2",
        "MAS1",
        "MATE",
        "MCIL",
        "MKEA",
        "MOBS",
        "NANO",
        "NKLG",
        "NNOR",
        "NRMD",
        "OHI3",
        "POL2",
        "REUN",
        "RIO2",
        "SALU",
        "SANT",
        "SCTB",
        "STHL",
        "STJ3",
        "THU2",
        "THTG",
        "TNML",
        "VESL",
        "VNDP",
        "XMIS",
]
"""

def database():

    """ 
    读取数据库中的测站信息数据
    :return: data_dict 包含全部测站数据的字典
    """
    #连接数据库
    conn=pymysql.connect(host='localhost',user='root',password='Pathfinder',database='db3',charset='utf8')
    #使用with语句创建游标对象
    with conn.cursor() as cur: 
        # 执行查询语句，获取表中的所有数据 
        cur.execute("SELECT * FROM 测站信息")
        data = cur.fetchall()
        # 使用列表推导式将数据转换为字典列表
        data_dict = [
            {
                'Site-Name': row[0],
                'Country/Region': row[1],
                'Receiver': row[2],
                'Antenna': row[3],
                'Radome': row[4],
                'Satellite-System': row[5],
                'Latitude': row[6],
                'Longitude': row[7],
                'Height(m)': row[8],
                'X(m)': row[9],
                'Y(m)': row[10],
                'Z(m)': row[11],
                'Calibration': row[12],
                'Networks': row[13],
                'Data-Center': row[14],
                'IERS-DOMES': row[15],
                'Clock': row[16],
                'Agencies': row[17],
                'Last-Data': row[18]
            }
            for row in data
        ]

    # 关闭数据库连接
    conn.close()
    return data_dict

def is_in_list(value, lst):
    """
    判断核心测站
    """
    # 遍历列表中的每个元素
    for item in lst:
        # 如果找到了相等的元素，返回 True
        if value == item:
            return True
    # 如果遍历完了没有找到相等的元素，返回 False
    return False

def split_dict(d):
    """
    将导入的原始数据分割成dict1(核心测站) dict2(非核心测站)两个字典
    """
    dict1 = []
    dict2 = []
    for v in d:

        if is_in_list(str(v['Site-Name'])[:4] , core_list):
            dict1.append(v) 
        else:
            dict2.append(v) 
    return dict1, dict2

# 定义一个函数，接受一个字典和一个列表作为参数
def filter_dict_by_list(dict_list, list):
    # 遍历字典中的每个键
    i = 0
    idx_max = len(dict_list)
    while(i<idx_max):
        # 如果键不在列表集合中，将其添加到需要删除的键列表中
        if dict_list[i]['Site-Name'][:4] not in list:
            #print(dict_list[i]['Site-Name'][:4])
            del dict_list[i]
            idx_max = len(dict_list)
            # 从字典中删除该键及其对应的值
        else:
            i+=1
    # 遍历需要删除的键列表中的每个键
    # 返回筛选后的字典
    return dict_list

def split_dict_omit(d):
    """
    从核心测站表中筛选出简略版的核心测站表
    return：简略版的核心测站表
    """
    dict1 = []
    for v in d:

        if is_in_list(str(v['Site-Name'])[:4] , core_list):
            dict1.append(v) 

    return dict1

def gdop(data_dict):
    """
    计算多个点的三维坐标gdop值

    :param data_dict: 包含点的三维坐标数据的测站数据列表，
    :param n: 要计算的点的索引
    :return: gdop值
    """
    n = len(data_dict) # 获取字典的长度
    a = np.zeros((n, 4)) # 创建一个n行4列的零矩阵
    #print(a[0])
    for i in range(n): # 遍历字典中的每个点
        x = float(data_dict[i]['X(m)']) # 获取点的x坐标
        y = float(data_dict[i]['Y(m)']) # 获取点的y坐标
        z = float(data_dict[i]['Z(m)']) # 获取点的z坐标
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        #print(x / dist)
        a[i] = np.array([x / dist, y / dist,
                z / dist, 1])
        #print(array)
        #a[i] = cp.array(array) # 计算点的单位向量和1，并赋值给矩阵a的第i行
    q = np.dot(a.T, a) # 计算矩阵a的转置和a的乘积，得到q矩阵
    q_inv = np.linalg.inv(q) # 计算q矩阵的逆矩阵，得到q_inv矩阵
    #gdop = cp.sqrt(q_inv[0][0] + q_inv[1][1] + q_inv[2][2] + q_inv[3][3])
    gdop = np.sqrt(q_inv.trace()) # 计算q_inv矩阵对角线元素之和的平方根，得到gdop值

    return gdop # 返回gdop值

# 定义一个函数，接受一个二维数组作为参数
def distance_matrix(points):
    # 获取数组的形状
    n, m = points.shape
    # 沿着第一个轴复制n次
    points_row = cp.repeat(points, n, axis=0)
    # 沿着第二个轴复制n次
    points_col = cp.tile(points, (n, 1))

    # 相减得到差值数组
    diff = points_row - points_col
    # 沿着第三个轴求平方和
    square_sum = cp.sum(diff**2, axis=1)
    # 开根号得到距离矩阵
    dist_mat = cp.sqrt(square_sum).reshape(n, n)
    # 返回距离矩阵
    return dist_mat

# 定义一个函数，接受一个距离矩阵作为参数
def min_distance(dist_mat):
    # 获取矩阵的形状
    n, n = dist_mat.shape
    # 创建一个无穷大值的对角矩阵
    inf_diag = cp.diag([np.inf] * n)
    # 将距离矩阵与对角矩阵相加，忽略自身与自身的距离
    dist_mat += inf_diag
    # 沿着第二个轴求最小值
    min_dist = cp.min(dist_mat, axis=1)
    idx = list.index(min_dist)
    # 返回最小距离列表
    return min_dist,idx


def Wgdop(data_dict,weight_list):
    """
    计算多个点的三维坐标gdop值

    :param data_dict: 测站数据字典，包含三维坐标数据
    :param weight_list：测站权重列表
    :param n: 要计算的点的索引
    :return: gdop值
    """
    n = len(data_dict) # 获取字典的长度
    a = cp.zeros((n, 4)) # 创建一个n行4列的零矩阵
    for i in range(n): # 遍历字典中的每个点
        x = data_dict[i]['X(m)'] # 获取点的x坐标
        y = data_dict[i]['Y(m)'] # 获取点的y坐标
        z = data_dict[i]['Z(m)'] # 获取点的z坐标
        dist = cp.sqrt(x ** 2 + y ** 2 + z ** 2)
        a[i] = [x / dist, y / dist,
                z / dist, 1] # 计算点的单位向量和1，并赋值给矩阵a的第i行
    #计算权矩阵
    q = cp.dot(a.T,a) # 计算矩阵a的转置和a的乘积，得到q矩阵
    q_inv = cp.linalg.inv(q) # 计算q矩阵的逆矩阵，得到q_inv矩阵
    gdop = cp.sqrt(q_inv[0][0] + q_inv[1][1] + q_inv[2][2] + q_inv[3][3]) # 计算q_inv矩阵对角线元素之和的平方根，得到gdop值

    return gdop # 返回Wgdop值

#根据gdop值的局部最优化选取
def Iterative_operations(dict1,dict2,n):
    """
    迭代循环,将普通测站依据gdop值一步步添加到核心测站中
    :param dict1: 核心测站字典
    :param dict2: 普通测站字典
    :param n:选取普通测站的数目
    :return: 优化后的测站列表
    """

    h1_dict = dict1.copy() # 复制一份核心测站字典，避免修改原字典
    h2_dict = dict2.copy() # 复制一份普通测站字典，避免修改原字典
    count = 0 # 定义一个计数器变量
    while count < n: # 当计数器小于n时，继续循环
        list=[]
        for i in range(len(h2_dict)):
            h_dict=h1_dict.copy()
            h_dict.append(h2_dict[i].copy())
            list.append(gdop(h_dict)) # 用列表推导式生成gdop值的列表
        min_index = list.index(min(list)) # 用min()函数和index()方法找到最小值的索引
        h1_dict += [h2_dict.pop(min_index)] # 用pop()方法删除并返回普通测站字典中的最小值元素，并用+=运算符合并到核心测站字典中
        globals()["Gdop_gdop"].append(gdop(h1_dict))
        
        count=count+1
    return h1_dict

#根据最邻近指数的局部最优化选取
def nearest_neighbor_index_operations(dict1,dict2,n):

    """
    迭代循环,将普通测站依据最邻近指数一步步添加到核心测站中
    :param dict1: 核心测站字典
    :param dict2: 普通测站字典
    :param n:选取普通测站的数目
    :return: 优化后的测站列表
    """
    h1_dict = dict1.copy() # 复制一份核心测站字典，避免修改原字典
    h2_dict = dict2.copy() # 复制一份普通测站字典，避免修改原字典
    count = 0 # 定义一个计数器变量
    while count < n: # 当计数器小于n时，继续循环
        list=[]
        for i in range(len(h2_dict)):
            h_dict=h1_dict.copy()
            h_dict.append(h2_dict[i].copy())
            list.append(nearest_neighbor_index_and_point_density(h_dict)[0]) # 用列表推导式生成最邻近指数的列表
        min_index = list.index(max(list)) # 用max()函数和index()方法找到最小值的索引
        h1_dict += [h2_dict.pop(min_index)] # 用pop()方法删除并返回普通测站字典中的最大值元素，并用+=运算符合并到核心测站字典中

        globals()["Gdop_nearest"].append(gdop(h1_dict))
        count=count+1  
    return h1_dict

def Monte_Carlo_randomized_trial(dictname,n):
    """

    蒙特卡洛随机试验传统格网

    定义问题和目标
    问题：从每个格网里选一个测站，使gdop值最小
    目标：找到最优的测站组合和对应的gdop值

    确定输入变量和分布
    输入变量：二维字典dictname，随机试验次数n
    分布：假设n=10000，每次从列表中随机选择一个数
    """
    # 初始化最小gdop值和最优测站组合
    min_gdop=cp.inf
    opt_comb = None
    # 循环n次
    for i in range(n):
        # 从每个格网中随机选择一个测站，组成一个组合
        comb = []
        for key1 in dictname:
            for key2 in dictname[key1]:
                comb.append(cp.random.choice(dictname[key1][key2],1))
        # 计算组合的gdop值
        Gdop = gdop(comb)
        # 如果小于最小gdop值，就更新最小gdop值和最优测站组合
        if Gdop < min_gdop:
            min_gdop = Gdop
            opt_comb = comb
    # 返回最小gdop值和最优测站组合
    globals()["Gdop_grid"].append(min_gdop)
    return min_gdop, opt_comb

def Monte_Carlo_randomized_trial_plus(station_list_ceil,station_list_normal,weight_list_normal,n,m):
    """
    蒙特卡洛随机试验加权格网

    m:选取的测站数
    n:随机次数
    
    定义问题和目标
    问题：利用格网对测站加权，再取出一定数量的测站，使gdop值最小
    目标：找到最优的测站组合和对应的gdop值

    确定输入变量和分布
    输入变量：二维字典dictname，随机试验次数n
    分布：假设n=10000，每次从列表中随机选择一个数
    """

    # 初始化最小gdop值和最优测站组合
    # 初始化最小gdop值和最优测站组合
    min_gdop=cp.inf
    opt_comb = None
    # 循环n次
    for i in range(n):
        if(i%100==0):
            print('\rSimulation Loop: {0}           '.format(i),end="")
        # 从测站列表中依照权值随机选择m个测站，组成一个组合
        comb = []
        comb= np.random.choice(station_list_normal,m,weight_list_normal)
        # 计算组合的gdop值
        Gdop = gdop(station_list_ceil+comb.tolist())
        # 如果小于最小gdop值，就更新最小gdop值和最优测站组合
        if Gdop < min_gdop:
            min_gdop = Gdop
            opt_comb = comb
    print('\n')
    globals()["Gdop_grid"].append(min_gdop) 
    # 返回最小gdop值和最优测站组合
    return min_gdop, opt_comb

#格网法选取
def Grid_operations(dict1,dict2,n1,m,n):
    """
    通过格网法选取测站
    :param dict1: 核心测站字典
    :param dict2: 普通测站字典
    :param n1:选取普通测站的数目
    :param m：经度分割的块数
    :param n：纬度分割的块数
    :return: 优化后的测站列表
    """

    #格网数
    #num_grid=math.ceil((360/m)(180/n))

    #每个网格的经度尺寸
    grid_size_x=360/m

    #每个网格的纬度尺寸
    grid_size_y=180/m

    # 创建一个空的列表，用来在格网中存储全部测站
    dict_2d = {k: {i: None for i in range(1, n+1)} for k in range(1, m+1)} # 列表推导式创建二维字典

    # 遍历每个点 
    for station in dict2:

        # 计算点所在的格网单元位置
        grid_x_num = cp.ceil((station['Longitude']+180 )/ grid_size_x)
        grid_y_num = cp.ceil((station['Latitude']+90) / grid_size_y) 

        #存入测站列表中
        dict_2d[grid_x_num][grid_y_num].append(station)

    #删除所有空的格网
    
    # 创建一个新的空字典new_dictname，用来存储非空的值
    new_dictname = {}
    # 遍历二维字典的每一个键和值
    for key, value in dict_2d.items():
        # 判断值是否为空，如果不为空，就添加到新的字典中
        if value != None:
            new_dictname[key] = value
    
    #进行蒙特卡洛随机试验,返回最小gdop值和最优测站组合
    print('Monte Carlo Started')
    min_gdop,opt_comb = Monte_Carlo_randomized_trial(new_dictname,10000)

def station_dict(this_dict,grid_size_x,grid_size_y,n):
    # 创建一个空的列表，用来在格网中存储全部测站    
    dict_d = [[[] for i in range(n)] for i in range(2*n)]

     # 遍历每个点 
    for station in this_dict:
        if float(station['Longitude'])>180:
            # 计算点所在的格网单元位置
            grid_x_num = cp.ceil((float(station['Longitude']) )/ grid_size_x)
            grid_y_num = cp.ceil((float(station['Latitude'])+90) / grid_size_y) 
        else:
            grid_x_num = cp.ceil((float(station['Longitude'])+180 )/ grid_size_x)
            grid_y_num = cp.ceil((float(station['Latitude'])+90) / grid_size_y) 
        #存入测站列表中
        dict_d[int(grid_x_num)-1][int(grid_y_num)-1].append(station)
    return dict_d

#加权格网法选取
def Grid_operations_weight(normal_grid_dict,dict1,n1,n):
    """
    通过格网法加权选取测站
    :param normal_grid_dict: 非核心测站网格列表
    :param dict1: 核心测站字典列表
    :param n1:选取普通测站的数目
    :param n：纬度方向划分的格网数
    :return: 优化后的测站列表
    """

    #格网数
    #num_grid=n*2*n

    #每个网格的尺寸
    #grid_size_y = 180 / n
    #grid_size_x = grid_size_y

    # 删除所有空的格网
    # 创建一个新的空列表new_dictname，用来存储非空的值
    new_dictname =  [[sub_sub_lst for sub_sub_lst in sub_lst if sub_sub_lst] for sub_lst in normal_grid_dict if any(sub_sub_lst for sub_sub_lst in sub_lst)]

    station_list_normal=[]
    weight_list_normal=[]

    for sub_lst in new_dictname:
        for sub_sub_lst in sub_lst:
            # 计算每个值所在列表的长度
            length = len(sub_sub_lst)
            # 计算每个值的权重，即列表长度的倒数
            weight = 1 / length
            for each in sub_sub_lst:
                if is_in_list(str(each['Site-Name'])[:4] , core_list_omit)!=True:
                    for element in sub_sub_lst:               
                        # 把元素和权重合起来形成一个列表
                        station_list_normal.append(element)
                        # 把这个权列表添加到二维列表中
                        weight_list_normal.append(weight)
    print('Monte Carlo Started')
    return Monte_Carlo_randomized_trial_plus(dict1,station_list_normal,weight_list_normal,10000,n1)

def euclidean_distance(x1,y1,z1,x2,y2,z2):
    """
    定义一个计算两个三维点之间欧氏距离的函数
    p1和p2都是包含x, y, z坐标的元组
    使用勾股定理计算距离
    """
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def nearest_neighbor_index_and_point_density(data_dict):
    """
    定义一个计算三维点集的最邻近指数和点密度的函数
    points是一个列表，每个元素是一个包含x, y, z坐标的元组
    """

    n = len(data_dict) # 点的数量
    observed_mean_distance = 0 # 观测平均距离
    expected_mean_distance = 0 # 预期平均距离
    nearest_neighbor_index = 0 # 最邻近指数
    point_density = 0 # 点密度
    area = 0 # 研究区域面积

    # 如果点的数量小于2，无法计算，返回None
    if n < 2:
        return None

    # 计算观测平均距离
    # 对于每个点，找到其最近邻要素，并累加距离
    for i in range(n):
        min_distance = float("inf") # 最近邻距离，初始化为无穷大
        for j in range(0,n):
            if i != j: # 排除自身
                distance = euclidean_distance(data_dict[i]['X(m)'], data_dict[i]['Y(m)'],data_dict[i]['Z(m)'],
                                              data_dict[j]['X(m)'], data_dict[j]['Y(m)'],data_dict[j]['Z(m)']) # 计算两点之间的距离
                if distance < min_distance: # 如果距离小于当前最小值，更新最小值
                    min_distance = distance
        observed_mean_distance += min_distance # 累加最近邻距离

    # 计算观测平均距离的平均值
    observed_mean_distance /= n
    """
    # 计算研究区域面积
    # 使用点的最大和最小坐标构造一个长方体，并计算其体积作为面积的近似值
    points = [(d['X(m)'], d['Y(m)'], d['Z(m)']) for d in data_dict]


    min_x = min(points, key=lambda p: p[0])[0] # 最小x坐标
    max_x = max(points, key=lambda p: p[0])[0] # 最大x坐标
    min_y = min(points, key=lambda p: p[1])[1] # 最小y坐标
    max_y = max(points, key=lambda p: p[1])[1] # 最大y坐标
    min_z = min(points, key=lambda p: p[2])[2] # 最小z坐标
    max_z = max(points, key=lambda p: p[2])[2] # 最大z坐标

    area = (max_x - min_x) * (max_y - min_y) * (max_z - min_z) # 长方体体积
    

    # 如果面积为零，无法计算，返回None
    if area == 0:
        return None
    """
    #地球表面积
    area=510062670*1000000

    # 计算预期平均距离
    expected_mean_distance = 0.5 / math.sqrt(n / area)

    # 计算最邻近指数
    nearest_neighbor_index = observed_mean_distance / expected_mean_distance

    # 计算点密度
    point_density = n / area

    # 返回结果
    return nearest_neighbor_index, point_density

def point_draw_GDOP(n1,i):
    """
    绘制选取完后的测站gdop图

    data_dict ：未添加测站的测站数据
    data_dict0：添加完测站后的测站数据
    n1：gdop评测列表结果

    i：添加的测站数
    
    """
     # 设置字体为宋体，防止中文标签不显示
    plt.rcParams['font.sans-serif'] = ['STzhongsong']
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 创建第一个图形对象
    fig1=plt.figure()

    
    # 二维：定义横坐标列表
    x = range(5, 5*i+5,5)

    # 二维：调用plot函数绘制两条折线图，并设置不同的颜色和标签
    plt.plot(x, n1, color='red', label='基于gdop值的加权格网法选取')
 

    # 二维：设置x轴和y轴的标签
    plt.xlabel('引入的非核心测站数')
    plt.ylabel('gdop值')

    # 二维：设置图表的标题
    plt.title('gdop值图')

    # 非阻塞地显示第一个图像
    plt.savefig('grid.png',dpi=800)
    #plt.show(block=False)
    
def point_draw(data_dict,data_dict1,data_dict2,n1,n2,i):
    """
    绘制二维gdop值对比图和三维选取完后的测站散点图

    data_dict ：未添加测站的测站数据
    data_dict1：法1添加完测站后的测站数据
    data_dict2：法2添加完测站后的测站数据
    n1：基于gdop值的局部最优算法的gdop评测列表结果

    n2：基于最邻近指数的局部最优算法的gdop评测列表结果
    i：添加的测站数
    
    """
    # 设置字体为宋体，防止中文标签不显示
    plt.rcParams['font.sans-serif'] = ['STzhongsong']
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    # 创建第一个图形对象
    fig1=plt.figure()

    
    # 二维：定义横坐标列表
    x = range(1, i+1)

    # 二维：调用plot函数绘制两条折线图，并设置不同的颜色和标签
    plt.plot(x, n1, color='red', label='基于gdop值的局部最优化选取')
    plt.plot(x, n2, color='blue', label='基于最邻近指数的局部最优化选取')

    # 二维：设置x轴和y轴的标签
    plt.xlabel('引入的非核心测站数')
    plt.ylabel('gdop值')

    # 二维：设置图表的标题
    plt.title('gdop值对比图')

    # 非阻塞地显示第一个图像
    #plt.show(block=False)
    plt.savefig('nearest_gdop_compare.png',dpi=800)

    """
    导出测站经纬度数据
    lons_0，lats_0:简略版核心测站经纬度列表
    lons_1，lats_1：法1添加的测站经纬度列表
    lons_2，lats_2：法2添加的测站经纬度列表
    """
    lons_0 = [float(d['Longitude']) for d in data_dict]
    lats_0 = [float(d['Latitude']) for d in data_dict]

    lons_1 = [float(d['Longitude']) for d in data_dict1][len(data_dict):]
    lats_1 = [float(d['Latitude']) for d in data_dict1][len(data_dict):]

    lons_2 = [float(d['Longitude']) for d in data_dict2][len(data_dict):]
    lats_2 = [float(d['Latitude']) for d in data_dict2][len(data_dict):]

    # 创建第二个图形对象
    fig2 = plt.figure(figsize=(16,6))

    # 生成两个子图
    ax2 = fig2.add_subplot(121)
    ax3 = fig2.add_subplot(122)
    
    # 创建一个二维世界地图对象
    m1 = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax2)

    x01, y01 = m1(lons_0, lats_0) # 将经纬度转换为地图坐标
    x1, y1 = m1(lons_1, lats_1)

    # 绘制地图的边界，海岸线，国家
    m1.drawmapboundary(linewidth=0.25)
    m1.drawcoastlines(linewidth=0.25)
    m1.drawcountries(linewidth=0.25)

    # 在地图上添加散点
    m1.scatter(x01, y01, s=20, c='red', marker='o', zorder=3) # 绘制散点
    m1.scatter(x1, y1, s=20, c='green', marker='v', zorder=3)

    # 设置图表的标题
    ax2.set_title('法一测站分布图')
    

    # 创建一个二维世界地图对象
    m2 = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax3)

    x02, y02 = m2(lons_0, lats_0) # 将经纬度转换为地图坐标
    x2, y2 = m2(lons_2, lats_2)

    # 绘制地图的边界，海岸线，国家
    m2.drawmapboundary(linewidth=0.25)
    m2.drawcoastlines(linewidth=0.25)
    m2.drawcountries(linewidth=0.25)

    # 在地图上添加散点
    m2.scatter(x02, y02, s=20, c='red', marker='o', zorder=3) # 绘制散点
    m2.scatter(x2, y2, s=20, c='green', marker='v', zorder=3)

    # 设置图表的标题
    ax3.set_title('法二测站分布图')

    # 非阻塞地显示第二个图像
    #plt.show(block=False)
    plt.savefig('nearest_ditribute.png',dpi=800)

    # 保持所有图像打开
    #plt.show()
    
def list_readin(filename):
    # 创建一个空列表，用于存储转换后的大写字符
    uppercase_list = []
    # 打开文件，使用with语句可以自动关闭文件
    with open(filename, "r") as f:
        # 逐行读取文件内容
        for line in f:
            # 去掉每行末尾的换行符
            line = line.strip()
            # 将每行的英文字符转换成大写，并添加到列表中
            uppercase_list.append(line.upper())
    # 返回列表
    return uppercase_list

start = time.time()
core_list_omit=[]
#定义两个全局变量列表，用于存储每添加一个非核心测站后的gdop值
Gdop_gdop=[]
Gdop_nearest=[]
Gdop_grid=[]

# 从数据库中读取全部测站数据
data_dict = database()

#core_dict = [] # 核心测站数据

core_dict_omit=[] # 简略版的核心测站数据

normal_dict = [] # 普通测站数据

core_dict, normal_dict = split_dict(data_dict)

list_station_path = 'D:\\2023秋\\大创\\list\\4be7e800e5b54629b582840ec7ba7e54_c6a2afaf9d238128dc15b9dda107ba3f_8'
list = list_readin(list_station_path)
# 将列表转换成集合，以提高查找效率
core_dict = filter_dict_by_list(core_dict, list)
normal_dict = filter_dict_by_list(normal_dict, list)
#core_dict_omit=split_dict_omit(core_dict)

"""
其后代码供调试，参数可能会有修改
"""
#选取的非核心测站数
i=20+53
#格网纬度方向分成的块数
i2=10
max_num = int((len(data_dict)-i)/10)*10
grid_size_y = 180 / i2
grid_size_x = grid_size_y
core_grid_dict = station_dict(core_dict,grid_size_x,grid_size_y,i2)
normal_grid_dict = station_dict(normal_dict,grid_size_x,grid_size_y,i2)

for i1 in range(10,110,10):
    print('Calculating {0} with step interval {1}'.format(i1,i2))
    Grid_operations_weight(normal_grid_dict,core_dict,i1,i2)
#print(Gdop_grid)
point_draw_GDOP(Gdop_grid,10)
Gdop_grid.clear()

end = time.time()
# 计算并打印运算使用的时间，单位是秒
execution_time = end - start
print(f"运算使用的时间是：{execution_time}秒")
exit

# core_dict: 核心测站字典, normal_dict: 普通测站字典, 10: 从普通测站字典中选取的测站数

new_dict_1 = Iterative_operations(core_dict, normal_dict, i) 
new_dict_2 = nearest_neighbor_index_operations(core_dict, normal_dict, i) 


"""
# core_dict_omit: 简略版的核心测站字典, normal_dict: 普通测站字典, 10: 从普通测站字典中选取的测站数
new_dict_1 = Iterative_operations(core_dict_omit, normal_dict, i) 
new_dict_2 = nearest_neighbor_index_operations(core_dict_omit, normal_dict, i) 
"""

point_draw(core_dict_omit,new_dict_1,new_dict_2,Gdop_gdop,Gdop_nearest,i)
