import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap  #地图底图库
import numpy as np
import os,itertools

def create_folder(save_folder):
    if(os.path.exists(save_folder)==False):
        os.makedirs(save_folder)
    return save_folder+'/'

def point_draw_GDOP(save_folder,result_lst,max_num,num_interval):
    # 设置字体为宋体，防止中文标签不显示
    plt.rcParams['font.sans-serif'] = ['STzhongsong']
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

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
    x = list(range(num_interval,max_num,num_interval))

    # 二维：调用plot函数绘制两条折线图，并设置不同的颜色和标签
    plt.plot(x, result_lst, color='red', label='Gdop值随非核心测站增量变化')
 
    # 二维：设置x轴和y轴的标签
    plt.xlabel('引入的非核心测站数')
    plt.ylabel('gdop值')

    # 二维：设置图表的标题
    plt.title('gdop值图')

    # 非阻塞地显示第一个图像
    plt.savefig(create_folder(save_folder)+'gdop.jpg',dpi=800)
    plt.close()
    #plt.show(block=False)

def single_draw(save_folder,figure_type,normal_dict,core_dict,opt_st_comb,st_num):
    # 设置字体为宋体，防止中文标签不显示
    plt.rcParams['font.sans-serif'] = ['STzhongsong']
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
    """
    绘制三维选取完后的测站散点图

    data_dict ：未添加测站的测站数据
    data_dict1：法1添加完测站后的测站数据
    data_dict2：法2添加完测站后的测站数据    
    st_num：添加测站数量
    """

    """
    导出测站经纬度数据
    lons_0，lats_0:核心测站经纬度列表
    lons_1，lats_1：法1添加的测站经纬度列表
    """
    comb_dict = [each for each in normal_dict if each['Site-Name'] in opt_st_comb]

    lons_0 = [float(d['Longitude']) for d in core_dict]
    lats_0 = [float(d['Latitude']) for d in core_dict]

    lons_1 = [float(d['Longitude']) for d in comb_dict]
    lats_1 = [float(d['Latitude']) for d in comb_dict]

    lons_2 = [float(d['Longitude']) for d in normal_dict]
    lats_2 = [float(d['Latitude']) for d in normal_dict]

    # 创建图形对象
    fig = plt.figure(figsize=(8,4))
    ax = fig.add_subplot(111)

    # 生成两个子图
    #ax2 = fig2.add_subplot(121)
    #ax3 = fig2.add_subplot(122)
    
    # 创建一个二维世界地图对象
    m1 = Basemap(projection='cyl', llcrnrlat=-90, urcrnrlat=90, llcrnrlon=-180, urcrnrlon=180, ax=ax)

    x01, y01 = m1(lons_0, lats_0) # 将经纬度转换为地图坐标
    x1, y1 = m1(lons_1, lats_1)
    x2, y2 = m1(lons_2, lats_2)

    # 绘制地图的边界，海岸线，国家
    m1.drawmapboundary(linewidth=0.25)
    m1.drawcoastlines(linewidth=0.25)
    m1.drawcountries(linewidth=0.25)

    # 在地图上添加散点
    m1.scatter(x2, y2, s=5, c='none', marker='o',alpha=0.3,edgecolors = 'gray',label='未选测站')
    m1.scatter(x01, y01, s=30, c='red', marker='^',alpha=0.8,edgecolors = 'red',label='核心测站') # 绘制散点
    m1.scatter(x1, y1, s=20, c='green', marker='o',alpha=0.8,edgecolors = 'green',label='已选测站')

    # 设置图表的标题
    ax.set_title('{0}:{1}测站分布'.format(figure_type,st_num))
    plt.legend(loc="lower left")
    # 非阻塞地显示第二个图像
    #plt.show(block=False)
    plt.savefig(create_folder(os.path.join(save_folder,figure_type))+'{0}_station_scatter.png'.format(st_num),dpi=400)
    plt.close()
    # 保持所有图像打开
    #plt.show()

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
    plt.savefig('nearest_ditribute.png',dpi=400)

    # 保持所有图像打开
    #plt.show()

# 定义一个函数，接受一个文件名列表作为参数
def sort_by_number(file_list):
    # 定义一个辅助函数，用于提取文件名中的数字部分
    def get_number(file_name):
        # 用下划线分割文件名，返回第一个元素（数字）转换为整数
        return int(str(file_name).split("\\")[-1].split("_")[0])
    # 使用sorted函数，按照get_number函数的返回值对文件名列表进行排序
    # 返回排序后的列表
    return sorted(file_list, key=get_number)

def del_file(filepath):
    """
    删除某一目录下的所有文件或文件夹
    :param filepath: 路径
    :return:
    """
    del_list = os.listdir(filepath)
    for f in del_list:
        file_path = os.path.join(filepath, f)
        if os.path.isfile(file_path):
            os.remove(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)

import imageio,glob,shutil
def video_stacking(folder,day_seiral=0):
    # 获取当前文件夹下的所有 PNG 文件名
    filenames = glob.glob(folder+'/*.png')
    # 对文件名进行排序
    filenames = sort_by_number(filenames)
    # 遍历每个文件名
    if(os.path.exists('day{0}_{1}.mp4'.format(day_seiral,folder))):
        os.remove('day{0}_{1}.mp4'.format(day_seiral,folder))
    video_path = 'day{0}_{1}.mp4'.format(day_seiral,folder)
    images = []
    for filename in filenames:
        # 打开图片并添加到列表中
        images.append(imageio.v3.imread(filename))
    # 将列表中的第一张图片保存为 GIF，并指定其他图片作为动画帧
    imageio.mimsave(video_path,images,fps=6)
    shutil.rmtree(folder)
    #images[0].save('day{0}_{1}.gif'.format(day_seiral,folder), format='GIF', append_images=images[1:], save_all=True, duration=150, loop=1)

def find_roots(x, y):
    s = np.abs(np.diff(np.sign(y))).astype(bool)
    return x[:-1][s] + np.diff(x)[s] / (np.abs(y[1:][s] / y[:-1][s]) + 1)

def draw_dict(mission_type,mydict,max_num,num_interval,y_value_name,folder_path,day_seiral = 0,have_video = False):
    # 设置字体为宋体，防止中文标签不显示
    plt.rcParams['font.sans-serif'] = ['STzhongsong']
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
     # 设置字体为宋体，防止中文标签不显示
     # 二维：定义横坐标列表
    x = list(range(num_interval,max_num,num_interval))

    fig1 = plt.figure()
    ax1 = fig1.subplots()
    #ax.set_ylim(0, 0.5)
    key_lst = []
    
     # 二维：设置x轴和y轴的标签
    plt.xlabel('引入的非核心测站数')
    plt.ylabel(y_value_name)
    
    if(day_seiral<0):
        plt.title('{0}步长下{1}随非核心站平均变化'.format(num_interval,y_value_name))
    else:
        # 二维：设置图表的标题
        plt.title('{0}步长下{1}随非核心站引入变化'.format(num_interval,y_value_name))

    # 遍历字典的每个键值对
    for key, value in mydict.items():
        # 调用 gif_stacking 函数，将键对应的文件夹下的图片合成一张 GIF，并保存在当前目录下
        key_lst.append(key)
        if(have_video and day_seiral > 0):
            video_stacking(key, day_seiral)
        ax1.plot(x, value, label=key)
    combinations_lst = list(itertools.combinations(key_lst, 2))
    for each in combinations_lst:
        # Get the common range, from `max(x1[0], x2[0])` to `min(x1[-1], x2[-1])`   
        y1 = np.array(mydict[each[0]])
        y2 = np.array(mydict[each[1]])

        delta_y = np.sign(y1 - y2)
        for i in range(len(delta_y)-1):
            is_intersect = delta_y[i] * delta_y[i + 1]
            if(is_intersect < 1):
                coord_x = x[i] + 0.5
                coord_y = (y1[i]+y2[i])/2
                ax1.plot(coord_x, coord_y, 'r', ms=3)
                #plt.text(x,y,s,font,fontsize,style,color)
                ax1.text(coord_x,coord_y,'({0},{1:.3f})'.format(int(coord_x), coord_y))
    # 设置图例位置为最佳
    plt.legend(loc="best")
    # 保存图形在当前目录下，文件名为 "plot.png"
    if(day_seiral<0):
        fig_path = folder_path + "\\{0}_{1}_plot.jpg".format(mission_type,y_value_name)
    else:
        fig_path = folder_path + "\\{0}_{1}_day{2}_plot.jpg".format(mission_type,y_value_name,day_seiral)
    plt.savefig(fig_path, dpi = 800)
    plt.close()

import matplotlib.colors as mcolors
def chart_draw(array,A_lst,B_lst,entity_num,folder_path=""):
    fig, ax = plt.subplots()
    plt.title('{0} with {1} Analysis under {2} Stations'.format(B_lst[0],A_lst[0],entity_num))

    # 添加横轴标注  
    ax.set_xlabel(B_lst[0])
    ticks = np.array(list(range(array.shape[1])))+0.5
    ax.set_xticks(ticks)
    ax.set_xticklabels(B_lst[1:])

    # 添加纵轴标注
    ax.set_ylabel(A_lst[0])
    ticks = np.array(list(range(array.shape[0])))+0.5
    ax.set_yticks(ticks)  
    ax.set_yticklabels(A_lst[1:])

    chart_array = array[:,:,0]  ##只选择有score的一栏
    max_val = max(map(max, chart_array))
    min_val = min(map(min, chart_array))
    val_range = max_val - min_val
    ftsize = int(30 / max(array.shape))
    for i in range(chart_array.shape[0]):
        for j in range(chart_array.shape[1]):
            percent_to_maxmin = (chart_array[i][j] - min_val)/val_range
            color = mcolors.to_rgba('blue', percent_to_maxmin)
            ax.text(j + 0.5, i + 0.5, 'GAvg:{0:.2e}\n Quat:{1:.2e}'.format(array[i][j][2],array[i][j][1]), fontsize=ftsize , ha="center", va="center", color='r' if color[3] == 1 else ('w' if color[3] > 0.5 else 'k'))
            ax.add_patch(plt.Rectangle((j, i), 1, 1, facecolor=color))

    ax.set_xlim(0, chart_array.shape[1])
    ax.set_ylim(0, chart_array.shape[0])
    ax.set_aspect('equal')
    #plt.show()
    fig.savefig(folder_path + '{0}_with_{1}_Analysis_{2}Stations.png'.format(B_lst[0],A_lst[0],entity_num), dpi=300, bbox_inches='tight')

def gdop_line_draw(rslt_array, idx_lst, entity_num):
    # 设置字体为宋体，防止中文标签不显示
    plt.rcParams['font.sans-serif'] = ['STzhongsong']
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
     # 设置字体为宋体，防止中文标签不显示
     # 二维：定义横坐标列表

    fig1 = plt.figure()
    ax1 = fig1.subplots()
    #ax.set_ylim(0, 0.5)
    
     # 二维：设置x轴和y轴的标签
    plt.xlabel(idx_lst[0])
    plt.ylabel('GDOP')

    # 二维：设置图表的标题
    plt.title('{0} change with GDOP under {1} stations'.format(idx_lst[0], entity_num))
    for rslt_lst in rslt_array:
        ax1.plot(idx_lst[1:], rslt_lst, label=rslt_lst[0])
    #plt.text(x,y,s,font,fontsize,style,color)
    #ax1.text(coord_x,coord_y,'({0},{1:.3f})'.format(int(coord_x), coord_y))
    # 设置图例位置为最佳
    plt.legend(loc="best")
    # 保存图形在当前目录下，文件名为 "plot.png"
    #plt.show()
    plt.savefig("{0}_change with_GDOP_plot_under{1}stations.jpg".format(idx_lst[0], entity_num), dpi = 800)
    plt.close()

def box_line_draw(rslt_array, idx_lst,Analysis_type,folder_path=""):
    plt.rcParams['font.sans-serif'] = ['STzhongsong']
    # 设置正常显示负号
    plt.rcParams['axes.unicode_minus'] = False
     # 设置字体为宋体，防止中文标签不显示
     # 二维：定义横坐标列表

    fig1 = plt.figure()
    ax1 = fig1.subplots()
    #ax.set_ylim(0, 0.5)

    box_lst = []
    for idx_line in range(rslt_array.shape[0]):
        #print(rslt_array[idx_line,:])
        box_lst.append(rslt_array[idx_line,:])

     # 二维：设置图表的标题
    plt.title('{0}: {1} change with Stations Adding in'.format(Analysis_type,idx_lst[0]))
    plt.grid(True)  # 显示网格
    plt.boxplot(box_lst,
            medianprops={'color': 'red', 'linewidth': '1.5'},
            meanline=True,
            showmeans=True,
            meanprops={'color': 'blue', 'ls': '--', 'linewidth': '1.5'},
            flierprops={"marker": "x", "color": "red", "markersize": 5})
    #plt.text(x,y,s,font,fontsize,style,color)
    #ax1.text(coord_x,coord_y,'({0},{1:.3f})'.format(int(coord_x), coord_y))
    # 设置图例位置为最佳
    # 保存图形在当前目录下，文件名为 "plot.png"
    #plt.show()
    
    # 二维：设置x轴和y轴的标签
    ax1.set_xlabel(idx_lst[0])
    ax1.set_ylabel('GDOP')

    ticks = list(range(1,rslt_array.shape[0]+1))
    ax1.set_xticks(ticks)
    labels = [str(each) for each in idx_lst[1:]]
    ax1.set_xticklabels(labels)

    '''
    plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
    plt.margins(0, 0)
    保留图片大小的设置
    '''
    #print([item.get_text() for item in ax1.get_xticklabels(which='both')])
    plt.savefig(folder_path+"{0}_{1}_change with_Stations_Adding_in.jpg".format(Analysis_type,idx_lst[0]), dpi = 800, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    folder_lst = ['Grid Gdop','Step Gdop']
    for each in folder_lst:
        video_stacking(each,each)