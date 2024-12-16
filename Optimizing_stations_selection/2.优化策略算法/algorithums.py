#import cupy as cp   #Numpy 在数据点低于 1000 万时实际运行更快
import numpy as np
import scipy as sp
#np.show_config()

def euclidean_distance(x1,y1,z1,x2,y2,z2):
    """
    定义一个计算两个三维点之间欧氏距离的函数
    p1和p2都是包含x, y, z坐标的元组
    使用勾股定理计算距离
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)

###在进行gdop运算前，需要主动构造多点坐标对应的a阵
def gdop(a):
    """
    计算三维坐标gdop值
    :param a: 多点三维坐标构成的矩阵
    :return: gdop值
    """
    q = sp.dot(a.T, a) # 计算矩阵a的转置和a的乘积，得到q矩阵
    q_inv = np.linalg.inv(q) # 计算q矩阵的逆矩阵，得到q_inv矩阵
    gdop = np.sqrt(q_inv.trace()) # 计算q_inv矩阵对角线元素之和的平方根，得到gdop值
    return gdop # 返回gdop值

'''
def Wgdop(data_dict,weight_list):
    """
    计算多个点的三维坐标gdop值

    :param data_dict: 测站数据字典，包含三维坐标数据
    :param weight_list：测站权重列表
    :param n: 要计算的点的索引
    :return: gdop值
    """
    n = len(data_dict) # 获取字典的长度
    a = np.zeros((n, 4)) # 创建一个n行4列的零矩阵
    for i in range(n): # 遍历字典中的每个点
        x = data_dict[i]['X(m)'] # 获取点的x坐标
        y = data_dict[i]['Y(m)'] # 获取点的y坐标
        z = data_dict[i]['Z(m)'] # 获取点的z坐标
        dist = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        a[i] = [x / dist, y / dist,
                z / dist, 1] # 计算点的单位向量和1，并赋值给矩阵a的第i行
    #计算权矩阵
    q = np.dot(a.T,a) # 计算矩阵a的转置和a的乘积，得到q矩阵
    q_inv = np.linalg.inv(q) # 计算q矩阵的逆矩阵，得到q_inv矩阵
    gdop = np.sqrt(q_inv[0][0] + q_inv[1][1] + q_inv[2][2] + q_inv[3][3]) # 计算q_inv矩阵对角线元素之和的平方根，得到gdop值

    return gdop # 返回Wgdop值

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
    min_gdop = np.inf
    opt_comb = None
    # 循环n次
    for i in range(n):
        # 从每个格网中随机选择一个测站，组成一个组合
        comb = []
        for key1 in dictname:
            for key2 in dictname[key1]:
                comb.append(np.random.choice(dictname[key1][key2],1))
        # 计算组合的gdop值
        Gdop = gdop(comb)
        # 如果小于最小gdop值，就更新最小gdop值和最优测站组合
        if Gdop < min_gdop:
            min_gdop = Gdop
            opt_comb = comb
    # 返回最小gdop值和最优测站组合
    #globals()["Gdop_grid"].append(min_gdop)
    return min_gdop, opt_comb
'''

###蒙特卡洛随机生成测站列表组合
def Monte_Carlo_Select(factors,weights,n,m,replace = False):
    opt_random = np.empty((n,m),dtype=int)
    # 随机生成n个组合
    for i in range(n):
        # 从测站列表中依照权值随机选择m个测站，组成一个组合
        #comb= np.random.choice(station_list_normal,m,weight_list_normal)
        opt_random[i] = np.random.choice(len(factors), m, p = weights.ravel(),replace = replace)
    return opt_random

###未投入使用的实验性代码
def Monte_Carlo_randomized_trial_plus(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,n,m):
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
    min_gdop = np.inf
    opt_idx = None
    opt_random_idx = Monte_Carlo_Select(NormalStation_factors,NormalStation_weights,n,m)

    for idx_random in opt_random_idx:
        # 计算组合的gdop值
        st_val = NormalStation_factors[idx_random]
        Gdop = gdop(np.vstack((a_core,st_val)))
        # 如果小于最小gdop值，就更新最小gdop值和最优测站组合
        if Gdop < min_gdop:
            min_gdop = Gdop
            opt_idx = idx_random
    #globals()["Gdop_grid"].append(min_gdop) 
    # 返回最小gdop值和最优测站组合
    opt_st_comb = [NormalStation_names[i] for i in opt_idx]
    return min_gdop, opt_st_comb

### 测站组合从小到大重排
def sort_dop_idx(gdop_list,opt_random_idx):
    idx_list = np.argsort(gdop_list,axis=0)     ### 返回按照GDOP值大小排列后，按照原始位置排列的GDOP值大小索引顺序
    gdop_list = gdop_list[idx_list]             ### 令GDOP列表按照从小到大顺序重排
    opt_random_idx = opt_random_idx[idx_list]   ### 令蒙特卡洛抽取组合成的测站列表按照从小到大顺序重排
    return gdop_list,opt_random_idx

#初始化n个m长度的种群
def initialize_species(NormalStation_factors,NormalStation_weights,a_core,n,m):
    opt_random_idx = Monte_Carlo_Select(NormalStation_factors,NormalStation_weights,n,m)
    gdop_list = np.empty(n,dtype=float)
    for i in range(n):
        # 计算组合的gdop值
        st_val = NormalStation_factors[opt_random_idx[i]]
        gdop_list[i] = gdop(np.vstack((a_core,st_val)))
    #numpy.argsort(a, axis=-1, kind=None, order=None)
    gdop_list, opt_random_idx = sort_dop_idx(gdop_list,opt_random_idx)  ### 测站组合从小到大重排
    return gdop_list,opt_random_idx

### 未投入使用的实验性代码（简单的线性拉伸，用于计算“适应度”）
def inversed_linear_stretch(gdop_list, max_out = 255, min_out = 0):
    min_val = gdop_list[0]
    max_val = gdop_list[-1]
    linear_stretch_lst = np.empty(gdop_list.shape)
    # 遍历原列表中的每个元素，按照公式进行转换，并添加到新列表中
    for i in range(len(gdop_list)):
        #Big is smaller, small is better
        linear_stretch_lst[i] = (max_val - gdop_list[i]) / (max_val - min_val) * (max_out - min_out) + min_out 
        '''
        Originally this code is designed for linear-stretch in gray images
        gray[gray < min_out] = min_out
        gray[gray > max_out] = max_out
        if(max_out <= 255):
            gray = np.uint8(gray)
        elif(max_out <= 65535):
            gray = np.uint16(gray)
        '''
    linear_stretch_lst /= np.sum(linear_stretch_lst)
    return linear_stretch_lst

# Gdop转适应度
def Gdop_fitness_translate(gdop_list):
    # 获取列表的长度
    n = len(gdop_list)
    # 获取列表的最小元素和最大元素
    #min_val = gdop_list[0]
    #max_val = gdop_list[-1]
    # 计算等差数列的公差
    d = (0.5 - 1) / (n - 1)
    # 创建一个新的列表，用于存储转换后的元素
    new_lst = np.empty(gdop_list.shape)
    # 遍历原列表中的每个元素，按照公式进行转换，并添加到新列表中
    for i in range(n):
        # 转换公式为：new_val = 1 + i * d
        new_lst[i] = 1 + i * d
    new_lst /= np.sum(new_lst)
    # 返回归一化之后的新列表
    return new_lst

###模拟遗传过程中的“染色体”配对过程
def mating(population,fitness_ratio, normal_dict_len,crossing_rate = 0.7, mutation_rate = 0.05):
    #一对个体进行交配，产生一对新的子代(与真实染色体不同，这里的个体只有一组而非两组数据，可以理解随机长度“染色体”自由组合)
    roulette_idx = np.random.choice(len(population), 2, p = fitness_ratio.ravel(),replace = False)
    [entity1, entity2] = population[roulette_idx]
    if np.random.random() < crossing_rate:
        # PMX算子+单点交叉算法
        entity1 , entity2 = Partial_single_point_cross_matching(entity1, entity2)
    if np.random.random() < mutation_rate:
        entity1 = Random_single_point_variation(entity1, normal_dict_len)
    if np.random.random() < mutation_rate:
        entity2 = Random_single_point_variation(entity2, normal_dict_len)
    return entity1,entity2 

# 遗传算法，轮盘赌概率的选择函数
def roulette_selection(population, fitness_ratio, a_core, normal_dict_len, NormalStation_factors, crossing_rate, mutation_rate):
    """
    population: 一个二维列表，表示初始种群，每一行代表一个个体
    fitness: 一个一维列表，表示每个个体的适应度值
    num: 一个整数，表示要选择出的个体数量
    """
    mating_times = int(len(population)/2)
    filial_lst = np.empty((mating_times*2,len(population[0])),dtype=int)
    filial_gdop = np.empty(mating_times*2, dtype=float)
    # 计算每个个体的累积适应度比例
    #cumulative_ratio = np.cumsum(fitness_ratio, axis=0)
    for i in range(mating_times):
        filial_lst[2*i], filial_lst[2*i-1] = mating(population,fitness_ratio, normal_dict_len, crossing_rate, mutation_rate)

    for i in range(len(filial_lst)):
        st_val = NormalStation_factors[filial_lst[i]]
        filial_gdop[i] = gdop(np.vstack((a_core,st_val)))
    filial_gdop, filial_lst = sort_dop_idx(filial_gdop,filial_lst)  ### 测站组合从小到大重排
    return filial_gdop, filial_lst

###遍历得到两测站列表中第一个相同元素出现的位置
def find_first_idx(parent,same_st_lst,cross_idx):
    same_idx = np.array([parent.index(each) for each in same_st_lst if(parent.index(each) < cross_idx)],dtype=int)
    return same_idx

###对测站列表中的相同元素进行交换（经过两次交换后，能保证该元素回到原来的位置，从而避免同一列表中出现相同测站）
def array_crossing(arr1,arr2,same_idx):
    #print(arr1-arr2)
    tmp = arr2[same_idx]
    arr2[same_idx] = arr1[same_idx]
    arr1[same_idx] = tmp
    #print(arr1-arr2)
    return arr1,arr2

# PMX算子+单点交叉算法
def Partial_single_point_cross_matching(parent1, parent2):
    # 随机选择一个交叉位置
    cross_idx = np.random.randint(0,len(parent1))
    #c3 = list(set(a) & set(b))
    ### 先找出父母列表中有哪些重复项目
    same_st_lst = list(set(parent1).intersection(set(parent2)))
    #print(parent1)
    #print(cross_idx)
    ### 分别标记父母测站列表在需要交换的区间内所有重复元素第一次出现的位置（根据测站抽取规则，不可能出现重复元素出现两次的情况，故不考虑）
    same_idx_1 = find_first_idx(list(parent1),same_st_lst,cross_idx)
    same_idx_2 = find_first_idx(list(parent2),same_st_lst,cross_idx)
    ### 消除2测站列表中与1测站重复元素的索引位置，防止发生第三次交换
    same_idx_2 = np.array(list(set(same_idx_2).difference(set(same_idx_1))),dtype=int)

    ### 对父母测站列表的交换段进行重复元素的预先交换
    if(len(same_idx_1)>0):
        parent1, parent2 = array_crossing(parent1, parent2, same_idx_1)
    #print(same_idx_1)
    if(len(same_idx_2)>0):
        parent2, parent1 = array_crossing(parent2,parent1,same_idx_2)

    #print(parent1-tmp1)
    #print(parent2-tmp2)
    ### 正式交换，生成子代
    child1 = np.hstack((parent1[:cross_idx], parent2[cross_idx:]))
    child2 = np.hstack((parent2[:cross_idx], parent1[cross_idx:]))
    '''
    ### 以下代码用于检校交换后的子代列表是否存在重复元素
    if(len(set(child1))<len(child1)):
        print('error')
    if(len(set(child2))<len(child2)):
        print('error')
    '''
    #print(child1-tmp2)
    #print(child2-tmp1)
    return child1,child2

def Random_single_point_variation(entity ,normal_dict_len):
    # 随机选择一个变异位置
    idx_mutation = np.random.randint(len(entity))
    idx_all_gene = list(set(np.arange(normal_dict_len)) - set(entity))
    if(len(idx_all_gene)>0):
        idx_new_gene = np.random.choice(idx_all_gene) 
        entity[idx_mutation] = idx_new_gene
    #print(idx_mutation)
    return entity