import time
import data_tools as data
import drawing_tools as draw
import algorithums as alg
import numpy as np

#根据张角大小局部最优化选取
def best_angle(core_dict,normal_dict,n):
    """
    迭代循环,将普通测站依据最邻近指数一步步添加到核心测站中
    :param dict1: 核心测站字典
    :param dict2: 普通测站字典
    :param n:选取普通测站的数目
    :return: 优化后的测站列表
    """
    a_core = data.dict2aMat(core_dict)
    a_normal = data.dict2aMat(normal_dict)
    #start_mean_angle = alg.mean_min_angle(a_core)
    opt_idx = []
    maximise_angle = 0

    for i in range(n):
        idx_tmp = 0
        for j in range(len(a_normal)):
            angle_tmp = alg.mean_min_angle(np.vstack((a_core,a_normal[j])))
            if(angle_tmp > maximise_angle):
                maximise_angle = angle_tmp
                idx_tmp = j
        a_core = np.vstack((a_core,a_normal[idx_tmp]))
        a_normal=np.delete(a_normal,idx_tmp,axis=0)
        opt_idx.append(idx_tmp)
    opt_st_comb = [normal_dict[i]['Site-Name'] for i in opt_idx]
    angle_gdop = alg.gdop(a_core)
    return angle_gdop,opt_st_comb

start = time.time()
#core_list_omit=[]
#定义两个全局变量列表，用于存储每添加一个非核心测站后的gdop值
Gdop_gdop=[]

# 从数据库中读取全部测站数据
data_dict = data.database()

#core_dict = [] # 核心测站数据

core_dict_omit=[] # 简略版的核心测站数据

normal_dict = [] # 普通测站数据

core_dict, normal_dict = data.split_dict(data_dict)

list_station_path = 'D:\\2023秋\\大创\\list\\4be7e800e5b54629b582840ec7ba7e54_c6a2afaf9d238128dc15b9dda107ba3f_8'
list = data.list_readin(list_station_path)
# 将列表转换成集合，以提高查找效率
core_dict = data.filter_dict_by_list(core_dict, list)
normal_dict = data.filter_dict_by_list(normal_dict, list)
#core_dict_omit=split_dict_omit(core_dict)

#step
num_interval=3
max_num = int((len(normal_dict))/num_interval)*num_interval

last = start
opt_st_comb = []
method_name = 'Step MIN Angle'

for i1 in range(num_interval,max_num,num_interval):
    print('Calculating {0} with step interval {1}'.format(i1,num_interval))
    angle_gdop, opt_st_comb = best_angle(core_dict,normal_dict,i1)
    draw.single_draw(method_name,normal_dict,core_dict,opt_st_comb,i1)
    Gdop_gdop.append(angle_gdop)
    this = time.time()
    execution_time = this - last
    last = this
    print(f"task cost: {execution_time} sec\n")
#print(Gdop_grid)
draw.point_draw_GDOP(method_name,Gdop_gdop,max_num,num_interval)
Gdop_gdop.clear()

end = time.time()
# 计算并打印运算使用的时间，单位是秒
execution_time = end - start
print(f"运算使用的时间是：{execution_time}秒")