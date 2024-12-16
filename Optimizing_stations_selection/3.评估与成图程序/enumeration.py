import time
import data_tools as data
import drawing_tools as draw
import algorithums as alg
import numpy as np
import random,math
from itertools import combinations
import step_gdop as partdop
import grid_gdop as griddop
import Genetic_gdop as genedop
import scipy

### 枚举法
def gdop_enumeration(core_dict,normal_dict,n):
    a_core = data.dict2aMat(core_dict)
    a_normal = data.dict2aMat(normal_dict)
    opt_idx = []
    gdop = np.inf
    idx_list = list(range(len(a_normal)))
    comb = combinations(idx_list, n)
    print("Possibilities: {0:.0f}".format(scipy.special.comb(len(a_normal), n, exact=False)))
    #for comb_lst in comb:
    idx=0
    for comb_lst in comb:
        idx+=1
        if(idx%10000==0):
            print('\rPossibility Running: {0}           '.format(idx),end="")
        opt_st_comb = a_normal[np.array(comb_lst,dtype=int)]
        gdop_tmp = alg.gdop(np.vstack((a_core,opt_st_comb)))
        if(gdop_tmp < gdop):
            gdop = gdop_tmp
            opt_idx = comb_lst
    print()
    opt_st_comb = [normal_dict[i]['Site-Name'] for i in opt_idx]
    return gdop,opt_st_comb

if __name__ == '__main__':
    start = time.time()
    #core_list_omit=[]
    #定义两个全局变量列表，用于存储每添加一个非核心测站后的gdop值
    Gdop_gdop=[]

    # 从数据库中读取全部测站数据
    data_dict = data.database()

    #core_dict = [] # 核心测站数据

    core_num = 50
    normal_num = 30

    simu_dict=random.sample(data_dict,k=core_num+normal_num) # 简略版的核心测站数据\
    simu_list_core = [each['Site-Name'][:4] for each in simu_dict[:core_num]]
    simu_list_normal = [each['Site-Name'][:4] for each in simu_dict[core_num:]]

    core_dict, normal_dict, bool_dict = data.split_dict(simu_dict,simu_list_core)

    #list_station_path = 'D:\\2023秋\\大创\\list\\4be7e800e5b54629b582840ec7ba7e54_c6a2afaf9d238128dc15b9dda107ba3f_8'
    #data_list = data.list_readin(list_station_path)
    # 将列表转换成集合，以提高查找效率
    #core_dict = data.filter_dict_by_list(core_dict, data_list)
    #normal_dict = data.filter_dict_by_list(normal_dict, data_list)
    #core_dict_omit=split_dict_omit(core_dict)

    i2=72
    grid_size_y = 180 / i2
    grid_size_x = grid_size_y
    all_grid_dict = data.station_grid_dict(simu_dict,bool_dict,grid_size_x,grid_size_y,i2)

    last = start
    opt_st_comb = []
    gdop_dict = {'Grid Gdop Simu':[],'Step Gdop Simu':[],'Genetic Gdop Simu':[],'Enum Gdop Simu':[]}
    NormalStation_factors,NormalStation_weights,NormalStation_names,a_core = griddop.Gird_Generate(all_grid_dict,core_dict)

    num_interval = 3
    max_num = int((normal_num)/num_interval+1)*num_interval
    for i1 in range(num_interval,max_num,num_interval):
        print('Calculating {0} with step interval {1}'.format(i1,num_interval))

        min_gdop, opt_st_comb = griddop.Grid_operations_weight(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1)
        draw.single_draw('Grid Gdop Simu',normal_dict,core_dict,opt_st_comb,i1)
        gdop_dict['Grid Gdop Simu'].append(min_gdop)
        #print(opt_st_comb)
        print("Grid:{0}".format(min_gdop))

        min_gdop, opt_st_comb = partdop.best_gdop(core_dict,normal_dict,i1)
        draw.single_draw('Step Gdop Simu',normal_dict,core_dict,opt_st_comb,i1)
        gdop_dict['Step Gdop Simu'].append(min_gdop)
        #print(opt_st_comb)
        print("Step:{0}".format(min_gdop))

        min_gdop, opt_st_comb = genedop.Genetic_dop(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1)
        draw.single_draw('Genetic Gdop Simu',normal_dict,core_dict,opt_st_comb,i1)
        gdop_dict['Genetic Gdop Simu'].append(min_gdop)
        #print(opt_st_comb)
        print("Genetic:{0}".format(min_gdop))

        min_gdop, opt_st_comb = gdop_enumeration(core_dict,normal_dict,i1)
        draw.single_draw('Enum Gdop Simu',normal_dict,core_dict,opt_st_comb,i1)
        gdop_dict['Enum Gdop Simu'].append(min_gdop)
        #print(opt_st_comb)
        print("Enum:{0}".format(min_gdop))

        this = time.time()
        execution_time = this - last
        last = this
        print(f"task cost: {execution_time} sec\n")
    #print(Gdop_grid)
    draw.draw_dict('Enum',gdop_dict,max_num,num_interval,'gdop',True)
    gdop_dict_compare = {'Grid - Enum':[],'Step - Enum':[],'Genetic - Enum':[]}
    gdop_dict_compare['Grid - Enum'] = np.array(gdop_dict['Grid Gdop Simu'])-np.array(gdop_dict['Enum Gdop Simu'])
    gdop_dict_compare['Step - Enum'] = np.array(gdop_dict['Step Gdop Simu'])-np.array(gdop_dict['Enum Gdop Simu'])
    gdop_dict_compare['Genetic - Enum'] = np.array(gdop_dict['Genetic Gdop Simu'])-np.array(gdop_dict['Enum Gdop Simu'])
    draw.draw_dict('Enum_Compare',gdop_dict_compare,max_num,num_interval,'方法与最佳gdop差值')
    Gdop_gdop.clear()

    end = time.time()
    # 计算并打印运算使用的时间，单位是秒
    execution_time = end - start
    print(f"运算使用的时间是：{execution_time}秒")