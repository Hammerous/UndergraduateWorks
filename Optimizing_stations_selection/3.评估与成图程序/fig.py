import os
import re
import numpy as np
import drawing_tools as draw
import csv

def dict_to_csv(data, filename):
    # 从字典键中提取表头
    headers = list(data.keys())
    # 以写入模式打开文件
    with open(filename, 'w', newline='',encoding='gbk') as csvfile:
        # 创建CSV写入对象
        csv_writer = csv.writer(csvfile)
        
        # 将表头写入CSV文件
        csv_writer.writerow(headers)
        
        # 将数据行写入CSV文件
        for i in range(len(data[headers[0]])):
            row = [data[header][i] for header in headers]
            csv_writer.writerow(row)

# 定义一个函数来计算每个自变量的PDOP值的平均值
def calculate_average_pdop(path):
    gdop_dict_avg = {'格网加权':[],'局部优化':[],'遗传算法':[]}
    # 字典用于存储每个自变量的PDOP值的总和及其计数    
    # 正则表达式匹配PDOP值的行
    # 遍历当前目录下的所有文件
    step_interval = 5
    for filename in os.listdir(path):
        # 检查文件是否为.txt文件
        if filename.endswith('.txt'):
            gdop_dict_tmp = {'Grid Gdop 10k':[],'Step Gdop':[],'Genetic Gdop':[]}
            # 打开文件并读取行
            file_path = os.path.join(path,filename)
            with open(file_path, 'r') as file:
                for line in file:
                    if(len(line)>5):
                        seg = re.split(':',line)
                        #step = int(seg[0][:2])
                        mission_type = re.split('_',seg[0])[-1]
                        PDOP_value = float(seg[-1])
                        gdop_dict_tmp[mission_type].append(PDOP_value)
            gdop_dict_avg['格网加权'].append(gdop_dict_tmp['Grid Gdop 10k'])
            gdop_dict_avg['局部优化'].append(gdop_dict_tmp['Step Gdop'])
            gdop_dict_avg['遗传算法'].append(gdop_dict_tmp['Genetic Gdop'])

    gdop_dict_fnl = {'格网加权':[],'局部优化':[],'遗传算法':[]}
    for key,value in gdop_dict_avg.items():
        final_lst = []
        idx_all = 0
        while(True):
            add_times = 0
            add_value = 0
            for each in value:
                if(idx_all < len(each)):
                    add_times+=1
                    add_value+=each[idx_all]
            idx_all+=1
            if(add_times==0):
                break
            #final_lst.append(np.log(add_value/add_times))
            final_lst.append(add_value/add_times)
        gdop_dict_fnl[key] = final_lst
    dict_to_csv(gdop_dict_fnl,path+r'\CSVDATA.csv')
    
    draw.draw_dict('Real_AvgLOG',gdop_dict_fnl,step_interval*idx_all,step_interval,'Log(GDOP)',path,-1)

# 调用函数计算平均PDOP值
#folderpath = r"D:\2023秋\大创\ava_site\tjr\Simplified Stations"
folderpath = r"D:\2023秋\大创\ava_site\tjf\Simplified Stations"
calculate_average_pdop(folderpath)