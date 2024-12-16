import time
import data_tools as data
import drawing_tools as draw
import algorithums as alg
import numpy as np
import grid_gdop as griddop

### 遗传算法
def Genetic_dop(NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, entity_num, species_num = 300 , crossing_rate = 0.7, mutation_rate = 0.05, threashod = 1e-9):
    opt_idx_lst = []
    gdop_lst = []
    gdop_tmp = np.inf
    gdop_list, population = alg.initialize_species(NormalStation_factors,NormalStation_weights,a_core,species_num,entity_num)
    #fitness_ratio = alg.inversed_linear_stretch(gdop_list,100,20)
    fitness_ratio = alg.Gdop_fitness_translate(gdop_list)
    #Fraction above is also an interesting topic to discuss about

    while(True):
        gdop_list, population = alg.roulette_selection(population, fitness_ratio, a_core, len(NormalStation_names),NormalStation_factors, crossing_rate, mutation_rate)
        ## gdop_list is in the order from max to min
        #fitness_ratio = alg.inversed_linear_stretch(gdop_list,100,20)
        fitness_ratio = alg.Gdop_fitness_translate(gdop_list)
        gdop_lst.append(gdop_list[0])
        idx_tmp = population[0]
        opt_idx_lst.append(idx_tmp)
        if(abs(gdop_tmp - gdop_list[0]) > threashod):
            gdop_tmp = gdop_list[0]
        else:
            break
    gdop_lst = np.array(gdop_lst)
    best_gdop_idx = np.argmin(gdop_lst,axis=0)
    gdop = gdop_lst[best_gdop_idx]
    opt_st_comb = [NormalStation_names[i] for i in opt_idx_lst[best_gdop_idx]]
    return gdop,opt_st_comb

###  参数分析使用的函数，实际计算中不涉及，用于选择参数
def name2idx(param_name):
    '''    
    Spieces_lst = ['Spieces Number', 100, 125, 150]
    loop_lst = ['Loop Number', 25, 100, 175]
    crossing_lst = ['Crossing Rate', 0.1, 0.3, 0.5, 0.7, 0.9]
    mutation_lst = ['Mutation Rate', 0.005, 0.01, 0.05, 0.1, 0.2]
    baseline: 300/100/0.7/0.05
    '''
    if(param_name == 'Spieces Number'):
        return 0
    elif(param_name == 'Crossing Rate'):
        return 1
    elif(param_name == 'Mutation Rate'):
        return 2
    else:
        print('paraments not extracted!')
        return False

### 参数分析使用的函数，实际计算中不涉及，用于评估算法收敛性
def convergence_analysis(num_interval,NormalStation_factors,NormalStation_weights,NormalStation_names,a_core, trial_time, species_num, crossing_rate, mutation_rate):
    max_num = int((normal_st_num)/num_interval+1)*num_interval
    idx_lst = ['Station Number']
    for station_num in range(num_interval,max_num,num_interval):
        idx_lst.append(station_num)
    idx_lst = idx_lst[:int(len(idx_lst)/2)+1]
    result_array = np.empty([len(idx_lst)-1,trial_time])
    for idx_station in range(1,len(idx_lst)):
        print('Analysing {0}/{1}'.format(idx_lst[idx_station] ,max_num))
        for idx in range(trial_time):
            min_gdop, opt_st_comb = Genetic_dop(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,idx_lst[idx_station],species_num,crossing_rate,mutation_rate)
            result_array[idx_station-1][idx] = min_gdop
    draw.box_line_draw(result_array,idx_lst,'Convergence Anaysis')

baseline_const = [300,0.7,0.05]
### 参数分析使用的函数，实际计算中不涉及，用于模拟单参数分析
def single_param_analysis(param_lst, NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,entity_num,trial_time,folder_path):
    baseline = np.copy(baseline_const)
    para_lst_len = len(param_lst)
    anlsys_rslt = np.zeros([para_lst_len - 1,trial_time])
    param_idx1 = name2idx(param_lst[0])
    for idx1 in range(1, para_lst_len):
        print("Starting simulation for {0}:{1};".format(param_lst[0],param_lst[idx1]))
        baseline[param_idx1] = param_lst[idx1]
        [species_num, crossing_rate,mutation_rate] = baseline
        for idx in range(trial_time):
            anlsys_rslt[idx1 - 1][idx],opt_st_comb_lst = Genetic_dop(NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, entity_num, int(species_num), crossing_rate,mutation_rate)
    draw.box_line_draw(anlsys_rslt,param_lst,'Convergence Anaysis under {0} Stations'.format(entity_num),folder_path)
    #draw.gdop_line_draw(anlsys_rslt, param_lst, entity_num)

### 参数分析使用的函数，实际计算中不涉及，用于计算四分位数差值
def interquartile_average(data):
    """
    Calculate the interquartile range of a given list of data.

    Parameters:
    data (list): A list of numerical data.

    Returns:
    float: The interquartile range of the data.
    """
    # Sort the data
    sorted_data = sorted(data)
    
    # Calculate the first quartile (Q1)
    q1_index = int(len(sorted_data) * 0.25)
    q1 = sorted_data[q1_index]
    
    # Calculate the third quartile (Q3)
    q3_index = int(len(sorted_data) * 0.75)
    q3 = sorted_data[q3_index]
    
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    
    return iqr[0],np.average(sorted_data)

### 参数分析使用的函数，实际计算中不涉及，用于模拟双参数分析
def double_param_analysis(A_lst, B_lst, NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,entity_num,trial_time,folder_path):
    num_x = len(A_lst) - 1
    num_y = len(B_lst) - 1
    anlsys_rslt = np.empty([num_x,num_y,3],dtype=float)
    param_idx1 = name2idx(A_lst[0])
    param_idx2 = name2idx(B_lst[0])
    baseline = np.copy(baseline_const)
    for idx1 in range(num_x):
        baseline[param_idx1] = A_lst[idx1+1]
        for idx2 in range(num_y):
            baseline[param_idx2] = B_lst[idx2+1]
            [species_num, crossing_rate,mutation_rate] = baseline
            print("Starting simulation for {0}:{1}; {2}:{3}".format(A_lst[0],A_lst[idx1+1],B_lst[0],B_lst[idx2+1]))
            #species_num, crossing_rate,mutation_rate = param_selection([A_lst,B_lst],[idx1,idx2])
            trial_array = np.empty([trial_time,1])
            for idx in range(trial_time):
                trial_array[idx],opt_st_comb_lst = Genetic_dop(NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, entity_num,int(species_num), crossing_rate,mutation_rate)
            interquartile, average = interquartile_average(trial_array)
            data_score = 1/(interquartile*average)
            ##四分位差越小，数据越集中；均值越小，数据选择效果越好
            anlsys_rslt[idx1][idx2] = np.array([data_score,interquartile, average])
    draw.chart_draw(anlsys_rslt,A_lst,B_lst,entity_num,folder_path)

import os
def mkdir(path):
    # 去除首位空格
    path=path.strip()
    # 去除尾部 \ 符号
    path=path.rstrip("\\")
    # 判断路径是否存在
    isExists=os.path.exists(path)
    # 判断结果
    if not isExists:
        # 如果不存在则创建目录,创建目录操作函数
        '''
        os.mkdir(path)与os.makedirs(path)的区别是,当父目录不存在的时候os.mkdir(path)不会创建，os.makedirs(path)则会创建父目录
        '''
        #此处路径最好使用utf-8解码，否则在磁盘中可能会出现乱码的情况
        os.makedirs(path) 
        return True
    else:
        # 如果目录存在则不创建，并提示目录已存在
        return False

if __name__ == '__main__':
    start = time.time()
    # 从数据库中读取全部测站数据
    data_dict = data.database()

    
    #core_dict = [] # 核心测站数据

    normal_dict = [] # 普通测站数据

    # 将数据库中的测站数据分为核心、非核心
    core_dict, normal_dict, bool_dict = data.split_dict(data_dict,data.core_list)

    list_station_path = 'D:\\2023秋\\大创\\list\\4be7e800e5b54629b582840ec7ba7e54_c6a2afaf9d238128dc15b9dda107ba3f_8'
    list = data.list_readin(list_station_path)
    # 将列表转换成集合，以提高查找效率
    core_dict = data.filter_dict_by_list(core_dict, list)
    normal_dict = data.filter_dict_by_list(normal_dict, list)
    normal_st_num = len(normal_dict)
    #core_dict_omit=split_dict_omit(core_dict)
    

    '''
    core_num = 50
    normal_num = 30

    simu_dict=random.sample(data_dict,k=core_num+normal_num) # 简略版的核心测站数据\
    simu_list_core = [each['Site-Name'][:4] for each in simu_dict[:core_num]]
    simu_list_normal = [each['Site-Name'][:4] for each in simu_dict[core_num:]]
    normal_st_num = len(simu_list_normal)
    core_dict, normal_dict, bool_dict = data.split_dict(simu_dict,simu_list_core)
    '''

    num_interval = 15

    #格网纬度方向分成的块数
    i2=72
    grid_size_y = 180 / i2
    grid_size_x = grid_size_y
    all_grid_dict = data.station_grid_dict(data_dict,bool_dict,grid_size_x,grid_size_y,i2)

    a_core = data.dict2aMat(core_dict)
    NormalStation_factors,NormalStation_weights,NormalStation_names,a_core = griddop.Gird_Generate(all_grid_dict,core_dict)
    
    last = start
    
    '''
    ## these rows of code is used to determine the convergence ability of method
    convergence_analysis(15, NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, 25, 300, 0.7, 0.05)
    this = time.time()
    execution_time = this - last
    last = this
    print(f"task cost: {execution_time} sec\n")
    ##
    '''
    

    Gdop_grid=[]

    '''
    ## these rows of code is used to determine the effects of stations' number adding into the simulation
    method_name = 'Genetic Gdop'
    max_num = int((normal_st_num)/num_interval+1)*num_interval
    for i1 in range(num_interval,max_num,num_interval):
        print('Calculating {0} with step interval {1}'.format(i1,num_interval))
        min_gdop, opt_st_comb = Genetic_dop(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1)
        draw.single_draw(method_name,normal_dict,core_dict,opt_st_comb,i1)
        Gdop_grid.append(min_gdop)
        this = time.time()
        execution_time = this - last
        last = this
        print(f"task cost: {execution_time} sec\n")
    #print(Gdop_grid)
    draw.point_draw_GDOP(method_name,Gdop_grid,max_num,num_interval)
    '''

    ## these rows of code is used to determine the effects of other paraments about the simulation
    Spieces_lst = ['Spieces Number', 100, 200, 300, 400, 500]
    #loop_lst = ['Loop Number', 25, 100, 175]
    crossing_lst = ['Crossing Rate', 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    mutation_lst = ['Mutation Rate', 0.001, 0.01, 0.05, 0.1, 0.2]
    i_lst = [50,100,150] #'Station Number'
    trial_time = 25
    # 获取当前所在文件的路径
    cur_path = os.getcwd()
    analysis_folder_path = cur_path+ '\\Analysis\\{0}\\'.format(list_station_path.split('\\')[-1])

    mkdir(analysis_folder_path)
    for i_num in i_lst:
        single_param_analysis(Spieces_lst, NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, i_num,trial_time,analysis_folder_path)
        single_param_analysis(crossing_lst, NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, i_num,trial_time,analysis_folder_path)
        single_param_analysis(mutation_lst, NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, i_num,trial_time,analysis_folder_path)
        double_param_analysis(Spieces_lst, crossing_lst, NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, i_num, trial_time,analysis_folder_path)
        double_param_analysis(Spieces_lst, mutation_lst, NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, i_num, trial_time,analysis_folder_path)
        double_param_analysis(mutation_lst, crossing_lst, NormalStation_factors, NormalStation_weights, NormalStation_names, a_core, i_num, trial_time,analysis_folder_path)

    '''
    a_core = data.dict2aMat(core_dict)
    NormalStation_factors,NormalStation_weights,NormalStation_names,a_core = griddop.Grid_operations_weight(all_grid_dict,core_dict)

    n = 300
    m = 100
    loop = 100

    best_gdop = np.empty(loop,dtype=float)
    gdop_list, population = alg.initialize_species(NormalStation_factors,NormalStation_weights,a_core,n,m)
    fitness_ratio = alg.inversed_linear_stretch(gdop_list,100,20)
    #fitness_ratio = alg.Gdop_fitness_translate(gdop_list)
    #Fraction above is also an interesting topic to discuss about

    for i in range(loop):
        gdop_list, population = alg.roulette_selection(population, fitness_ratio, a_core, len(normal_dict),NormalStation_factors)
        fitness_ratio = alg.inversed_linear_stretch(gdop_list,100,20)
        #fitness_ratio = alg.Gdop_fitness_translate(gdop_list)
        best_gdop[i] = gdop_list[0]
        st_val = NormalStation_factors[population[0]]
    '''

    #print(best_gdop)
    end = time.time()
    # 计算并打印运算使用的时间，单位是秒
    execution_time = end - start
    print(f"运算使用的时间是：{execution_time}秒")