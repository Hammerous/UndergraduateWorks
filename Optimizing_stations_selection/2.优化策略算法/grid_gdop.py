import time
import data_tools as data
import drawing_tools as draw
import algorithums as alg
import numpy as np

### 生成加权格网
def Gird_Generate(all_grid_dict,dict1):
    all_grid_lst =  [[sub_sub_lst for sub_sub_lst in sub_lst if sub_sub_lst] for sub_lst in all_grid_dict if any(sub_sub_lst for sub_sub_lst in sub_lst)]

    NormalStation_names = []
    NormalStation_factors = []
    AllStation_weights = []
    bool_remove = []

    a_core = data.dict2aMat(dict1)      ### 生成核心测站组成的坐标矩阵
    
    for grid_lst in all_grid_lst:
        for st_lst in grid_lst:
            length = len(st_lst)    # 计算每个值所在列表的长度（单个格网内测站数量）
            weight = 1 / length     # 计算每个值的权重，即列表长度的倒数（每个测站分到的权重，核心站与非核心站平权）
            for each in st_lst:
                #if data.is_in_list(str(each['Site-Name'])[:4] , core_list_omit)!=True:
                    #for element in sub_sub_lst:               
                # 把元素和权重合起来形成一个列表
                if(each['is_core_station']):
                    bool_remove.append(False)       # False时会消除核心测站，核心测站不参与最后的归一化操作
                else:
                    x = float(each['X(m)'])
                    y = float(each['Y(m)'])
                    z = float(each['Z(m)'])
                    dist = alg.euclidean_distance(x,y,z,0,0,0)
                    a_line = np.array([x / dist, y / dist,z / dist, 1])
                    # 把这个权列表添加到列表中
                    NormalStation_names.append(each['Site-Name'])
                    NormalStation_factors.append(a_line)
                    bool_remove.append(True)
                AllStation_weights.append(weight)
    NormalStation_factors = np.array(NormalStation_factors)
    NormalStation_weights = np.array(AllStation_weights)[np.array(bool_remove)]     # 消除核心测站
    NormalStation_weights /= np.sum(NormalStation_weights)                          # 归一化
    ### 以上操作的目的是适当降低格网内核心站较多的非核心站其权重，为核心站少的非核心站加权
    return NormalStation_factors,NormalStation_weights,NormalStation_names,a_core

### 格网加权法
def Grid_operations_weight(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1,trial_times=10000):
    """
    通过格网法加权选取测站
    :param normal_grid_dict: 非核心测站网格列表
    :param dict1: 核心测站字典列表
    :param n：纬度方向划分的格网数
    :return: 优化后的测站列表
    """
    print('Monte Carlo Sampling')
    min_gdop, opt_st_comb = alg.Monte_Carlo_randomized_trial_plus(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,trial_times,i1)
    return min_gdop, opt_st_comb

if __name__ == '__main__':
    start = time.time()
    #core_list_omit=[]
    #定义两个全局变量列表，用于存储每添加一个非核心测站后的gdop值
    Gdop_grid=[]

    # 从数据库中读取全部测站数据
    data_dict = data.database()

    #core_dict_omit=[] # 简略版的核心测站数据

    core_dict, normal_dict, bool_dict = data.split_dict(data_dict,data.core_list)

    list_station_path = 'D:\\2023秋\\大创\\list\\4be7e800e5b54629b582840ec7ba7e54_c6a2afaf9d238128dc15b9dda107ba3f_8'
    list = data.list_readin(list_station_path)
    # 将列表转换成集合，以提高查找效率
    core_dict = data.filter_dict_by_list(core_dict, list)
    normal_dict = data.filter_dict_by_list(normal_dict, list)
    #core_dict_omit=split_dict_omit(core_dict)

    num_interval = 10
    max_num = int((len(normal_dict))/num_interval+1)*num_interval
    #格网纬度方向分成的块数
    i2=72
    grid_size_y = 180 / i2
    grid_size_x = grid_size_y
    all_grid_dict = data.station_grid_dict(data_dict,bool_dict,grid_size_x,grid_size_y,i2)

    last = start
    opt_st_comb = []
    method_name = 'Grid Gdop'
    NormalStation_factors,NormalStation_weights,NormalStation_names,a_core = Gird_Generate(all_grid_dict,core_dict)
    for i1 in range(num_interval,max_num,num_interval):
        print('Calculating {0} with step interval {1}'.format(i1,num_interval))
        min_gdop, opt_st_comb = Grid_operations_weight(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1)
        draw.single_draw(method_name,normal_dict,core_dict,opt_st_comb,i1)
        Gdop_grid.append(min_gdop)
        this = time.time()
        execution_time = this - last
        last = this
        print(f"task cost: {execution_time} sec\n")
    #print(Gdop_grid)
    draw.point_draw_GDOP(method_name,Gdop_grid,max_num,num_interval)
    Gdop_grid.clear()

    end = time.time()
    # 计算并打印运算使用的时间，单位是秒
    execution_time = end - start
    print(f"运算使用的时间是：{execution_time}秒")