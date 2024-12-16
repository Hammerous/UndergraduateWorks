import time
import numpy as np
import random
import pymysql
import copy

# 简略的核心测站列表
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

def split_dict(d):
    """
    将导入的原始数据分割成dict1(核心测站) dict2(非核心测站)两个字典
    """
    dict1 = []
    dict2 = []
    for v in d:

        if is_in_list(str(v['Site-Name'])[:4] , core_list_omit):
            dict1.append(v) 
        else:
            dict2.append(v) 
    return dict1, dict2

def station_grid_dict(this_dict,grid_size_x,grid_size_y,n):
    # 创建一个空的列表，用来在格网中存储全部测站    
    dict_d = [[[] for i in range(n)] for i in range(2*n)]

     # 遍历每个点 
    for station in this_dict:
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

def database():

    """ 
    读取数据库中的测站信息数据
    :return: data_dict 包含全部测站数据的字典
    """
    #连接数据库
    conn=pymysql.connect(host='localhost',user='root',password='111111',database='db3',charset='utf8')
    
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

# 定义一个函数，接受3个字典和一个列表作为参数
def filter_dict_by_list_1(dict_list1,dict_list2,dict_list3, list):
    # 遍历字典中的每个键
    i = 0
    idx_max = len(dict_list1)
    while(i<idx_max):
        # 如果键不在列表集合中，将其添加到需要删除的键列表中
        if dict_list1[i] in list:
            #print(dict_list[i]['Site-Name'][:4])
            del dict_list1[i]
            del dict_list2[i]
            del dict_list3[i]
            idx_max = len(dict_list1)
            # 从字典中删除该键及其对应的值
        else:
            i+=1
    # 遍历需要删除的键列表中的每个键
    # 返回筛选后的字典
    return dict_list1,dict_list2,dict_list3

def euclidean_distance(x1,y1,z1,x2,y2,z2):
    """
    定义一个计算两个三维点之间欧氏距离的函数
    p1和p2都是包含x, y, z坐标的元组
    使用勾股定理计算距离
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

def gdop(a):
    """
    计算多个点的三维坐标gdop值

    :param data_dict: 包含点的三维坐标数据的测站数据列表，
    :param n: 要计算的点的索引
    :return: gdop值
    """
    q = np.dot(a.T, a) # 计算矩阵a的转置和a的乘积，得到q矩阵
    q_inv = np.linalg.inv(q) # 计算q矩阵的逆矩阵，得到q_inv矩阵
    gdop = np.sqrt(q_inv.trace()) # 计算q_inv矩阵对角线元素之和的平方根，得到gdop值
    return gdop # 返回gdop值

def create_initial_population(length, size, lower, upper):
    """
    length: 一个整数，表示每个个体的编码长度
    size: 一个整数，表示初始种群的规模
    lower: 一个整数或浮点数，表示基因位的最小取值
    upper: 一个整数或浮点数，表示基因位的最大取值
    """
    # 初始化初始种群列表
    population = []
    # 循环生成每个个体
    for i in range(size):
        # 初始化一个空列表，表示一个个体
        individual = []
        # 循环生成每个基因位
        for j in range(length):
            # 随机生成一个在取值范围内的值，添加到个体列表中
            gene = random.uniform(lower, upper)
            individual.append(gene)
        # 将个体添加到初始种群列表中
        population.append(individual)
    # 返回初始种群列表
    return population

def Monte_Carlo_randomized_trial_plus(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,n,m,Initial_population,gdop_list):
    """
    蒙特卡洛随机加权初始种群

    m:选取的测站数
    n:随机次数
    
    定义问题和目标
    问题：利用格网对测站加权，重复取出一定数量的测站，建立初始种群
    目标：找到最优的测站组合和对应的gdop值

    确定输入变量和分布
    输入变量：二维字典dictname，随机试验次数n
    分布：假设n=10000，每次从列表中随机选择一个数
    """
    # 循环n次
    for i in range(n):
        # 从测站列表中依照权值随机选择m个测站，组成一个组合
        idx_random = np.random.choice(len(NormalStation_factors),m, p = NormalStation_weights.ravel(),replace=False)
        opt_st_comb = [NormalStation_names[i] for i in idx_random]

        st_val = np.array([NormalStation_factors[i] for i in idx_random])

        gdop_list.append(gdop(np.vstack((a_core,st_val))))

        Initial_population.append(opt_st_comb)

    # 使用zip函数将两个数组打包成元组列表
    zipped = list(zip(gdop_list, Initial_population))

    # 使用sorted函数对元组列表按照第一个元素升序排序
    sorted_zipped = sorted(zipped, key=lambda x: x[0])

    # 使用zip函数将排序后的元组列表解压成两个数组
    gdop_list, Initial_population = zip(*sorted_zipped)

    return gdop_list,Initial_population

def dict2aMat(dic):
    n = len(dic) # 获取字典的长度
    #print(a[0])
    a_core = np.empty((n, 4)) # 创建一个n行4列的空矩阵
    for i in range(n):
        each=dic[i]
        x = float(each['X(m)'])
        y = float(each['Y(m)'])
        z = float(each['Z(m)'])
        dist = euclidean_distance(x,y,z,0,0,0)
        a_core[i]= np.array([x / dist, y / dist,z / dist, 1])
        #返回方向余弦阵
    return a_core

# Gdop转适应度
def Gdop_fitness_translate(gdop_list):
    # 获取列表的长度
    n = len(gdop_list)
    # 获取列表的最小元素和最大元素
    min_val = gdop_list[0]
    max_val = gdop_list[n-1]
    # 计算等差数列的公差
    d = (0.5 - 1) / (n - 1)
    # 创建一个新的列表，用于存储转换后的元素
    new_lst = []
    # 遍历原列表中的每个元素，按照公式进行转换，并添加到新列表中
    for i in range(n):
        # 转换公式为：new_val = 1 + i * d
        new_val = 1 + i * d
        new_lst.append(new_val)
    # 返回归一化之后的新列表
    return normalize(new_lst)

# 定义一个函数，用于将列表进行归一化，即将每个元素除以列表的和
def normalize(lst):
    # 计算列表的和
    s = sum(lst)
    # 创建一个新的列表，用于存储归一化后的元素
    new_lst = []
    # 遍历原列表中的每个元素，按照公式进行归一化，并添加到新列表中
    for x in lst:
        # 归一化公式为：new_val = x / s
        new_val = x / s
        new_lst.append(new_val)
    # 返回新列表
    return new_lst

# 遗传算法，轮盘赌概率的选择函数
def roulette_selection(population, fitness, num,NormalStation_names,a_core):
    """
    population: 一个二维列表，表示初始种群，每一行代表一个个体
    fitness: 一个一维列表，表示每个个体的适应度值
    num: 一个整数，表示要选择出的个体数量
    """
    
    # 初始化结果列表
    result = []
    new_gdop_list=[]

    # 计算总适应度值
    total_fitness = sum(fitness)
    # 计算每个个体的适应度比例
    fitness_ratio = [f / total_fitness for f in fitness]
    # 计算每个个体的累积适应度比例
    cumulative_ratio = [sum(fitness_ratio[:i+1]) for i in range(len(fitness_ratio))]
    for i in range(len(fitness)//2):
        # 循环2次,生成一对父代和母代
        for j in range(2):
            # 生成一个随机数
            r = random.random()
            # 遍历累积适应度比例
            for k in range(len(cumulative_ratio)):
                # 如果随机数小于等于当前累积适应度比例
                if r <= cumulative_ratio[k]:
                    # 将对应的个体添加到结果列表中
                    result.append(population[k])
                    # 跳出循环
                    break
        # 交叉算法
        if random.random()<0.7:
            # PMX算子+单点交叉算法
            result[2*i],result[2*i+1]=Partial_single_point_cross_matching(result[2*i],result[2*i+1])

    for i in range(len(fitness)):
        #  变异算法
        if random.random()<0.05:
            Random_single_point_variation(result[i],NormalStation_names)
    
    for i in range(len(result)):
        new_gdop_list.append(name_gdop_translate(result[i],a_core,normal_dict))


    # 使用zip函数将两个数组打包成元组列表
    zipped = list(zip(new_gdop_list, result))

    # 使用sorted函数对元组列表按照第一个元素升序排序
    sorted_zipped = sorted(zipped, key=lambda x: x[0])

    # 使用zip函数将排序后的元组列表解压成两个数组
    new_gdop_list, result = zip(*sorted_zipped)

    return result,new_gdop_list

# PMX算子+单点交叉算法
def Partial_single_point_cross_matching(a1,a2):

    # 深拷贝两个列表，避免修改原列表
    a1_1 = copy.deepcopy(a1)
    a2_1 = copy.deepcopy(a2)

    # 随机选择一个交叉位置
    y = random.randint(0,len(a1_1))
    # 记录交叉位置后的片段
    fragment1 = a1[y:]
    fragment2 = a2[y:]

    # 将两个列表的交叉位置后的片段互换
    a1_1[y:], a2_1[y:] = a2_1[y:], a1_1[y:]

    # 创建两个空列表，用于存放修正后的元素
    a1_2 = []
    a2_2 = []
    # 对第一个列表的交叉位置前的元素进行修正
    for i in a1_1[:y]:
        # 如果元素在第二个片段中出现过，就用第一个片段中对应的元素替换它，直到没有重复
        while i in fragment2:
            i = fragment1[fragment2.index(i)]
        # 将修正后的元素添加到新列表中
        a1_2.append(i)
    # 对第二个列表的交叉位置前的元素进行修正
    for i in a2_1[:y]:
        # 如果元素在第一个片段中出现过，就用第二个片段中对应的元素替换它，直到没有重复
        while i in fragment1:
            i = fragment2[fragment1.index(i)]
        # 将修正后的元素添加到新列表中
        a2_2.append(i)

    # 将修正后的列表和交换后的片段拼接起来，得到子代
    child1 = a1_2 + fragment2
    child2 = a2_2 + fragment1

    return child1,child2

def Random_single_point_variation(a1,normal_dict):
    # 随机选择一个变异位置
    y = random.randint(0,len(a1)-1)

    a2 = list(set(normal_dict) - set(a1))

    a = random.choice(a2) 

    a1[y] = a 

def name_gdop_translate(namelist,a_core,normal_dict):
    st_val=[]
    for i in namelist:
        for j in normal_dict:
            if i == j['Site-Name']: 
                x = float(j['X(m)'])
                y = float(j['Y(m)'])
                z = float(j['Z(m)'])
                dist = euclidean_distance(x,y,z,0,0,0)
                st_val.append(np.array([x / dist, y / dist,z / dist, 1]))
    return gdop(np.vstack((a_core,st_val)))

    
# 从数据库中读取全部测站数据
data_dict = database()

#core_dict = [] # 核心测站数据

normal_dict = [] # 普通测站数据

# 将数据库中的测站数据分为核心、非核心
core_dict, normal_dict =split_dict(data_dict)

list_station_path = 'siteAvaluable_2023249'
a_list = list_readin(list_station_path)

data_dict = filter_dict_by_list(data_dict, a_list)
core_dict = filter_dict_by_list(core_dict, a_list)
normal_dict = filter_dict_by_list(normal_dict, a_list)

#格网纬度方向分成的块数
i2=72
grid_size_y = 180 / i2
grid_size_x = grid_size_y

data_grid_dict= station_grid_dict(data_dict,grid_size_x,grid_size_y,i2)
core_grid_dict = station_grid_dict(core_dict,grid_size_x,grid_size_y,i2)
normal_grid_dict = station_grid_dict(normal_dict,grid_size_x,grid_size_y,i2)

data_grid_lst =  [[sub_sub_lst for sub_sub_lst in sub_lst if sub_sub_lst] for sub_lst in data_grid_dict if any(sub_sub_lst for sub_sub_lst in sub_lst)]


a_core = dict2aMat(core_dict)

AllStation_names = []
AllStation_factors = []
AllStation_weights = []
for grid_lst in data_grid_lst:
        for st_lst in grid_lst:
            # 计算每个值所在列表的长度
            length = len(st_lst)
            # 计算每个值的权重，即列表长度的倒数
            weight = 1 / length
            for each in st_lst:
                #if data.is_in_list(str(each['Site-Name'])[:4] , core_list_omit)!=True:
                    #for element in sub_sub_lst:               
                # 把元素和权重合起来形成一个列表
                x = float(each['X(m)'])
                y = float(each['Y(m)'])
                z = float(each['Z(m)'])
                dist = euclidean_distance(x,y,z,0,0,0)
                a_line = np.array([x / dist, y / dist,z / dist, 1])
                # 把这个权列表添加到列表中
                AllStation_names.append(each['Site-Name'])
                AllStation_factors.append(a_line)
                AllStation_weights.append(weight)

core_grid_dict_name = [d["Site-Name"] for d in core_dict]
NormalStation_names,NormalStation_factors,NormalStation_weights=filter_dict_by_list_1(AllStation_names,
                                                                AllStation_factors,AllStation_weights,core_grid_dict_name)

NormalStation_weights /= np.sum(NormalStation_weights)

n=300
m=100

population=[]
gdop_list=[]
fitness=[]
best_gdop=[]

gdop_list,population=Monte_Carlo_randomized_trial_plus(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,n,m,population,gdop_list)

fitness=Gdop_fitness_translate(gdop_list)
for i in range(100):

    population,gdop_list=roulette_selection(population, fitness, 100,NormalStation_names,a_core)

    fitness=Gdop_fitness_translate(gdop_list)

    best_gdop.append(min(gdop_list))
b=min(best_gdop)

print("finish")
