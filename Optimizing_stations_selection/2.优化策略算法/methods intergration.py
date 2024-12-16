import time,os
import data_tools as data
import drawing_tools as draw
import step_gdop as partdop
import grid_gdop as griddop
import Genetic_gdop as genedop
import numpy as np

def new_folder(current_dir):
    pht_dir = os.path.join(current_dir, "Simplified Stations")
    if not os.path.exists(pht_dir):
        os.mkdir(pht_dir)
    return pht_dir

def find_files(path,iterate_able = False):
    files = []
    for file in os.listdir(path):
        file_path = os.path.join(path, file)
        if os.path.isdir(file_path):
            if(iterate_able):
                files.extend(find_files(file_path))
        else:
            #files.append(file_path)
            files.append(file)
    return files

require_map_movie = False
activate_methods = [True,True,True]     ### 激活的三种方法：['格网加权','局部优化','遗传算法']
#list_folder_path = "D:\\2023秋\\大创\\ava_site\\tjf"
list_folder_path = "D:\\2023秋\\大创\\ava_site\\tjr"
#list_folder_path = "D:\\2023秋\\大创\\ava_site\\test"
store_folder = new_folder(list_folder_path)
step_interval = 5

if __name__ == '__main__':
    start = time.time()
    #core_list_omit=[]
    data_dict = data.database()         # 从数据库中读取全部测站数据
    core_dict, normal_dict, bool_dict = data.split_dict(data_dict,data.core_list)   # 加载核心测站与非核心测站字典以及bool判断字典（True为核心，False为非核心）

    #list_folder_path = 'D:\\2023秋\\大创\\list\\'
    dir_list = find_files(list_folder_path)
    #list_station_path = 'D:\\2023秋\\大创\\list\\4be7e800e5b54629b582840ec7ba7e54_c6a2afaf9d238128dc15b9dda107ba3f_8'
    gdop_dict_avg = {'格网加权':[],'局部优化':[],'遗传算法':[]}
    for idx in range(len(dir_list)):
        # 打开文件，使用写入模式
        #idx = len(dir_list)-2
        lst_filename = '{0}_best_lst.txt'.format(dir_list[idx])
        lst_filepath = os.path.join(store_folder,lst_filename)
        file = open(lst_filepath, "w")

        list_station_path = os.path.join(list_folder_path,dir_list[idx])
        data_list = data.list_readin(list_station_path)                     # 将列表转换成集合，以提高查找效率
        core_dict = data.filter_dict_by_list(core_dict, data_list)          # 筛选列表中存在的核心测站
        normal_dict = data.filter_dict_by_list(normal_dict, data_list)      # 筛选列表中存在的非核心测站
        #core_dict_omit=split_dict_omit(core_dict)

        max_num = int((len(normal_dict))/step_interval+1)*step_interval
        #格网纬度方向分成的块数
        i2=72
        grid_size_y = 180 / i2
        grid_size_x = grid_size_y
        all_grid_dict = data.station_grid_dict(data_dict,bool_dict,grid_size_x,grid_size_y,i2)      #生成各个格网下的测站字典
        NormalStation_factors,NormalStation_weights,NormalStation_names,a_core = griddop.Gird_Generate(all_grid_dict,core_dict)
        ### 利用格网对非核心测站加权，详见本函数

        last = start
        opt_st_comb = []

        gdop_dict = {'格网加权':[],'局部优化':[],'遗传算法':[]}
        time_consume = {'格网加权':[],'局部优化':[],'遗传算法':[]}

        #gdop_dict = {'Grid Gdop 10k':[],'Grid Gdop 100k':[],'Step Gdop':[],'Genetic Gdop':[]}
        #time_consume = {'Grid Gdop 10k':[],'Grid Gdop 100k':[],'Step Gdop':[],'Genetic Gdop':[]}
        for i1 in range(step_interval,max_num,step_interval):
            print('Calculating {0} with step interval {1}'.format(i1,step_interval))
            '''Grid Gdop with 100k'''
            '''
            time_start = time.time()
            min_gdop, opt_st_comb = griddop.Grid_operations_weight(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1,100000)
            time_end = time.time()
            draw.single_draw('Grid Gdop 100k',normal_dict,core_dict,opt_st_comb,i1)
            gdop_dict['Grid Gdop 100k'].append(min_gdop)
            time_consume['Grid Gdop 100k'].append(time_end - time_start)

            line = 'day{0}_{1}:'.format(idx,'Grid Gdop 100k')
            line += " ".join([str(item) for item in opt_st_comb])
            # 将字符串写入文件
            line +='    PDOP_value:{0:.10f}\n'.format(min_gdop)
            file.write(line)
            '''

            if(activate_methods[0]):
                '''Grid Gdop with 10k'''
                time_start = time.time()
                min_gdop, opt_st_comb = griddop.Grid_operations_weight(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1)
                time_end = time.time()
                #fig_folderpath = os.path.join(store_folder,dir_list[idx])
                #draw.single_draw(fig_folderpath,'Grid Gdop 10k',normal_dict,core_dict,opt_st_comb,i1)
                gdop_dict['格网加权'].append(min_gdop)
                time_consume['格网加权'].append(time_end - time_start)

                line = '{0}stations_{1}:['.format(i1,'Grid Gdop 10k')
                line += " ".join([str(item) for item in opt_st_comb])
                line +=']    PDOP_value: {0:.10f}\n'.format(min_gdop)
                # 将字符串写入文件
                file.write(line)

            if(activate_methods[1]):
                '''Step Gdop'''
                time_start = time.time()
                min_gdop, opt_st_comb = partdop.best_gdop(core_dict,normal_dict,i1)
                time_end = time.time()
                time_consume['局部优化'].append(time_end - time_start)
                #fig_folderpath = os.path.join(store_folder,dir_list[idx])
                #draw.single_draw(fig_folderpath,'Step Gdop',normal_dict,core_dict,opt_st_comb,i1)
                gdop_dict['局部优化'].append(min_gdop)

                line = '{0}stations_{1}:['.format(i1,'Step Gdop')
                line += " ".join([str(item) for item in opt_st_comb])
                line +=']    PDOP_value: {0:.10f}\n'.format(min_gdop)
                # 将字符串写入文件
                file.write(line)

            if(activate_methods[2]):
                '''Genetic Gdop'''
                time_start = time.time()
                min_gdop, opt_st_comb = genedop.Genetic_dop(NormalStation_factors,NormalStation_weights,NormalStation_names,a_core,i1)
                time_end = time.time()
                time_consume['遗传算法'].append(time_end - time_start)
                #fig_folderpath = os.path.join(store_folder,dir_list[idx])
                #draw.single_draw(fig_folderpath,'Genetic Gdop',normal_dict,core_dict,opt_st_comb,i1)
                gdop_dict['遗传算法'].append(min_gdop)

                line = '{0}stations_{1}:['.format(i1,'Genetic Gdop')
                line += " ".join([str(item) for item in opt_st_comb])
                line +=']    PDOP_value: {0:.10f}\n'.format(min_gdop)
                # 将字符串写入文件
                line +='\n\n'
                file.write(line)
            '''Data Write in'''

            this = time.time()
            execution_time = this - last
            last = this
            print(f"task cost: {execution_time} sec\n")

        #print(Gdop_grid)
        for key,value in gdop_dict_avg.items():
            value.append(gdop_dict[key])
        draw.draw_dict('Real',gdop_dict,max_num,step_interval,'GDOP',store_folder,idx,require_map_movie)
        gdop_dict.clear()
        file.close()
        draw.draw_dict('Real',time_consume,max_num,step_interval,'算法时间消耗(s)',store_folder,idx)

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
            final_lst.append(add_value/add_times)
        gdop_dict_fnl[key] = final_lst

    draw.draw_dict('Real_Avg',gdop_dict_fnl,step_interval*idx_all,step_interval,'GDOP',store_folder,-1)

    end = time.time()
    # 计算并打印运算使用的时间，单位是秒
    execution_time = end - start
    print(f"运算使用的时间是：{execution_time}秒")

'''
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
'''