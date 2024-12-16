import pymysql #数据库模块
import numpy as np

#核心测站列表

core_list_full=["AREQ",	"SCRZ",	"UNSA",		
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

core_list=["AREQ",
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
    return data_dict

def is_in_list(value, lst):
    """
    判断是否为核心测站
    """
    # 遍历列表中的每个元素
    for item in lst:
        # 如果找到了相等的元素，返回 True
        if value == item:
            return True
    # 如果遍历完了没有找到相等的元素，返回 False
    return False

def split_dict(d,core_list):
    """
    将导入的原始数据分割成dict1(核心测站) dict2(非核心测站)两个字典
    """
    dict1 = []
    dict2 = []
    dict_bool = {}
    for v in d:
        if is_in_list(str(v['Site-Name'])[:4] , core_list):
            dict1.append(v) 
            dict_bool.update({str(v['Site-Name'])[:4]:True})
        else:
            dict2.append(v) 
            dict_bool.update({str(v['Site-Name'])[:4]:False})
    return dict1, dict2,dict_bool

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

def station_grid_dict(this_dict,bool_dict,grid_size_x,grid_size_y,n):
    # 创建一个空的列表，用来在格网中存储全部测站    
    dict_d = [[[] for i in range(n)] for i in range(2*n)]
     # 遍历每个点 
    for i in range(len(this_dict)):
        this_dict[i].update({'is_core_station':bool_dict[this_dict[i]['Site-Name'][:4]]})
        station = this_dict[i]
        if float(station['Longitude'])>180:
            # 计算点所在的格网单元位置
            grid_x_num = np.ceil((float(station['Longitude']) )/ grid_size_x)
            grid_y_num = np.ceil((float(station['Latitude'])+90) / grid_size_y) 
        else:
            grid_x_num = np.ceil((float(station['Longitude'])+180 )/ grid_size_x)
            grid_y_num = np.ceil((float(station['Latitude'])+90) / grid_size_y) 
        #存入测站列表中
        dict_d[int(grid_x_num)-1][int(grid_y_num)-1].append(station)
    return dict_d

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

import algorithums as alg
def dict2aMat(dic):
    n = len(dic) # 获取字典的长度
    #print(a[0])
    a_core = np.empty((n, 4)) # 创建一个n行4列的空矩阵
    for i in range(n):
        each=dic[i]
        x = float(each['X(m)'])
        y = float(each['Y(m)'])
        z = float(each['Z(m)'])
        dist = alg.euclidean_distance(x,y,z,0,0,0)
        a_core[i]= np.array([x / dist, y / dist,z / dist, 1])
        #返回方向余弦阵
    return a_core
